"""Check that Uplift playlists move the way they are meant to.

For each negative emotion (sad, angry, fear, disgust) this prints one example
playlist with its waypoints, then builds RUNS playlists and reports how the
songs' valence and arousal change from song 1 to song 5.

Usage (from the project root):
    python scripts/evaluate_uplift.py             # all English songs
    python scripts/evaluate_uplift.py --previews  # only songs with a saved preview,
                                                  # which is what the app can play
    python scripts/evaluate_uplift.py --language Hindi [--previews]
    python scripts/evaluate_uplift.py --check     # the playlist check below
    python scripts/evaluate_uplift.py --tables    # only rebuild docs/results/ from the last --check

Songs are drawn from one language (default English). Bucket centres and the
distance scale always come from the English (Muse) songs, as in the app.

--check: the playlist check. Readings are not hand-picked: they are the
app's fused readings (app.py combine_emotions, HSEmotion's settings) of the
test pairs of scripts/evaluate_fusion.py (FER2013 PrivateTest x GoEmotions
test), keeping those that Uplift applies to (sad + angry + fear + disgust
>= 50%), grouped by the start emotion Uplift picks, and up to CHECK_READINGS
per start emotion sampled with a fixed seed. For each reading and each
language the 5-song path is built by the app's own build_queue in Uplift
mode (cache-only: only songs with a saved preview), and compared with two
baselines built by the app's own functions: 5 random playable songs of the
same language (player.playable_songs) and 5 songs of the start emotion's
bucket (player.resonance_queue, i.e. a Resonance playlist). The intended
progression is the planned path's waypoints: valence rises at every step;
arousal falls at every step for angry / fear / disgust (towards calm), and
for sad it falls, then rises (sad -> neutral -> happy, and happy songs are
more energetic than sad ones). Telugu and Malayalam: the app does not offer
Uplift (build_queue falls back to Resonance); the path player.uplift_queue
would build is reported to show why. Valence and arousal are the dataset's
values (Muse listener tags; Spotify audio features mapped by percentile),
not validated by people; this checks the playlist's structure, not any
effect on a listener.

Reads   data/songs/songs_clean.csv  (and data/songs/previews.csv with --previews)
Writes  models/uplift/report.txt    (report_previews.txt with --previews);
        for another language report_<language>.txt / report_previews_<language>.txt;
        --check: models/uplift/playlist_check.txt, playlist_check.csv,
        playlist_check_paths.csv, playlist_check_steps.png; and docs/results/:
        uplift_by_emotion / _by_language / _by_emotion_language (.csv, .md),
        uplift_by_emotion.png, uplift_by_language.png, uplift_wording.md

The intended progression (progression_tables): sad paths should get
brighter (valence rises at each step); angry, fear and disgust paths should
get calmer (arousal falls at each step; their valence change is reported
too). Each is measured as the share of steps in that direction, all 4 steps
in it, and song 5 against song 1, for the Uplift path and for the random and
Resonance baselines built from the same readings with the same seed.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uplift  # noqa: E402

RUNS = 500
SEED = 42
MIXED = {"sad": 0.45, "angry": 0.30, "neutral": 0.25}
# A waypoint is "thin" if its 10th-nearest candidate is further than this
# (in valence/arousal standard deviations) or it has fewer than 10 candidates.
THIN_DISTANCE = 0.5


def run_many(songs, fused, emotions, rng, pool=None):
    rows = []
    for _ in range(RUNS):
        playlist, _ = uplift.build_playlist(songs, fused, emotions, rng, pool=pool)
        rows.append({"valence": playlist.valence_tags.values, "arousal": playlist.arousal_tags.values,
                     "first_bucket": playlist.bucket.iloc[0] if len(playlist) else None,
                     "last_bucket": playlist.bucket.iloc[-1] if len(playlist) else None})
    return pd.DataFrame(rows)


def preview_coverage(songs, in_language, pool, log):
    """How many songs per bucket can be played, and how well each waypoint is served."""
    import previews
    language = songs[in_language]
    checked = language.lastfm_url.map(lambda url: previews.cached(url) is not None)
    table = pd.DataFrame({"songs": language.groupby("bucket").size(),
                          "checked": checked.groupby(language.bucket).sum(),
                          "with_preview": pd.Series(pool[in_language], index=language.index).groupby(language.bucket).sum()})
    table["hit_rate"] = (table.with_preview / table.checked.where(table.checked > 0)).map(
        lambda x: "-" if pd.isna(x) else f"{x:.0%}")
    log("SONGS WITH A SAVED PREVIEW, BY BUCKET")
    log(table.to_string())
    log(f"total: {int(pool.sum()):,} of {len(language):,} songs have a preview; {int(checked.sum()):,} have been checked")

    points = songs[["valence_tags", "arousal_tags"]].values
    scale = uplift.reference_scale(songs)
    centres = uplift.bucket_centres(uplift.reference_songs(songs))
    log(f"\nCANDIDATES PER WAYPOINT (journey buckets only). 'd10' is the distance to the 10th-nearest candidate in\n"
        f"standard deviations; a waypoint is thin if it has fewer than {uplift.NEAREST_K} candidates or d10 > {THIN_DISTANCE}.")
    for emotion in uplift.NEGATIVE_EMOTIONS:
        journey = uplift.plan_journey({emotion: 1.0}, [emotion], centres, scale)
        in_buckets = songs.bucket.isin({emotion, *journey["targets"]}).values & in_language
        log(f"-- {emotion} -> {' -> '.join(journey['targets'])}")
        for step, waypoint in enumerate(journey["waypoints"], 1):
            cells = []
            for label, mask in [("all songs", in_buckets), ("with preview", in_buckets & pool)]:
                distance = np.sort(np.linalg.norm((points[mask] - waypoint) / scale, axis=1))
                d10 = distance[uplift.NEAREST_K - 1] if len(distance) >= uplift.NEAREST_K else float("inf")
                cells.append((label, int((distance <= THIN_DISTANCE).sum()), d10))
            thin = cells[1][2] > THIN_DISTANCE
            log(f"   waypoint {step} ({waypoint[0]:.2f}, {waypoint[1]:.2f}): "
                + " | ".join(f"{label}: {near} within {THIN_DISTANCE}, d10 {d10:.2f}" for label, near, d10 in cells)
                + ("   <-- THIN" if thin else ""))


CHECK_READINGS = 200
CHECK_LANGUAGES = ["English", "Hindi", "Tamil", "Korean", "Telugu", "Malayalam"]
METHODS = [("uplift", "Uplift path (app)"), ("random", "5 random songs, same language"),
           ("resonance", "5 songs of the start bucket (Resonance)")]


class _SeededRng:
    """While active, each np.random.default_rng() call without a seed gets the next fixed seed, so the
    app's own unseeded picks (build_queue, uplift_queue) are repeatable. Seeded calls are unchanged."""

    def __init__(self, first):
        self.real, self.next = np.random.default_rng, first

    def __enter__(self):
        def seeded(*args, **kw):
            if args or kw:
                return self.real(*args, **kw)
            self.next += 1
            return self.real(self.next)
        np.random.default_rng = seeded
        return self

    def __exit__(self, *exc):
        np.random.default_rng = self.real


def path_metrics(rows, waypoints, scale, start_emotion, reading_emotions):
    """Structure of one 5-song path (rows: the songs in play order) against the planned waypoints."""
    m = {"songs": len(rows), "complete": len(rows) == uplift.N_SONGS,
         "distinct_links": rows.lastfm_url.nunique() == len(rows),
         "distinct_songs": (rows.title.str.casefold() + "|" + rows.artist.str.casefold()).nunique() == len(rows)}
    if not len(rows):
        return m
    v, a = rows.valence_tags.values.astype(float), rows.arousal_tags.values.astype(float)
    start = waypoints[0]
    m["first_distance"] = float(np.linalg.norm((np.array([v[0], a[0]]) - start) / scale))
    m["first_in_start_bucket"] = rows.bucket.iloc[0] == start_emotion
    m["first_in_reading"] = rows.bucket.iloc[0] in reading_emotions
    if m["complete"]:
        want_v, want_a = np.sign(np.diff(waypoints[:, 0])), np.sign(np.diff(waypoints[:, 1]))
        got_v, got_a = np.sign(np.diff(v)), np.sign(np.diff(a))
        m.update({"valence_steps_right": float(np.mean(got_v == want_v)), "valence_all_steps": bool(np.all(got_v == want_v)),
                  "arousal_steps_right": float(np.mean(got_a == want_a)), "arousal_all_steps": bool(np.all(got_a == want_a)),
                  "d_valence": v[-1] - v[0], "d_arousal": a[-1] - a[0]})
        m.update({f"V{i + 1}": v[i] for i in range(5)})
        m.update({f"A{i + 1}": a[i] for i in range(5)})
    return m


def playlist_check():
    import eval_common as ec
    import face_models
    import mood
    import player
    import previews
    import app_code
    assert not previews.LIVE_LOOKUPS, "run without LIVE_LOOKUPS: the check uses the saved previews only, as the app does"
    out_dir = os.path.join("models", "uplift")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    songs = pd.read_csv(os.path.join("data", "songs", "songs_clean.csv"), keep_default_na=False)
    by_link = songs.set_index("lastfm_url")
    app = app_code.load(["build_queue"], FUSION_CONFIG=face_models.fusion_config(ec.PRIMARY))
    setup = ec.Setup(ec.PRIMARY)
    centres, scale = uplift.bucket_centres(uplift.reference_songs(songs)), uplift.reference_scale(songs)
    playable = songs.lastfm_url.isin(previews.playable_urls()).values

    # Readings: the app's fused readings of the test pairs, those Uplift applies to, by start emotion.
    pairs = ec.test_pairs(setup.face_y, setup.text_y)
    unique = sorted({(int(f), int(t)) for p in pairs for f, t in zip(p[0], p[1])})
    readings = []
    for f, t in unique:
        emotions, fused = setup.combine(setup.face_raw[f], setup.text_raw[t])
        if uplift.should_uplift(fused):
            readings.append({"face": f, "text": t, "emotions": emotions, "fused": fused,
                             "start": max(uplift.NEGATIVE_EMOTIONS, key=lambda e: fused[e])})
    rng = np.random.default_rng(SEED)
    sample = {}
    for e in uplift.NEGATIVE_EMOTIONS:
        pool = [r for r in readings if r["start"] == e]
        sample[e] = [pool[i] for i in sorted(rng.choice(len(pool), min(CHECK_READINGS, len(pool)), replace=False))]

    log("UPLIFT PLAYLIST CHECK (structure of the 5-song path; not an effect on any listener)")
    log(f"Readings: app.py combine_emotions with HSEmotion's fusion settings on the {len(unique):,} distinct (face, comment) test pairs of")
    log(f"scripts/evaluate_fusion.py; {len(readings):,} have sad + angry + fear + disgust >= {uplift.MIN_NEGATIVE_SHARE:.0%} (Uplift applies).")
    log("Readings per start emotion (available -> sampled, seed 42): "
        + ", ".join(f"{e} {sum(r['start'] == e for r in readings)} -> {len(sample[e])}" for e in uplift.NEGATIVE_EMOTIONS))
    log("Each reading's playlist mix (emotions at >= 15%), its start point and waypoints come from the app's code unchanged.")
    log("Songs: only those with a saved preview (data/songs/previews.csv), as the app plays them by default.\n")

    log("PLAYABLE SONGS PER LANGUAGE AND BUCKET")
    log(pd.crosstab(songs.language[playable], songs.bucket[playable]).reindex(CHECK_LANGUAGES).to_string())
    log("")
    log("PLANNED PATH (waypoints, valence / arousal; the same for every language)")
    for e in uplift.NEGATIVE_EMOTIONS:
        j = uplift.plan_journey({e: 1.0}, [e], centres, scale)
        log(f"  {e:8} -> {' -> '.join(j['targets']):15} " + "  ".join(f"({v:.2f}, {a:.2f})" for v, a in j["waypoints"]))
    log("  (a mixed reading starts between the centres of its emotions, weighted by their share)\n")

    paths = []
    with _SeededRng(SEED * 100000):
        for language in CHECK_LANGUAGES:
            in_language = (songs.language == language).values
            buckets = {b: rows for b, rows in songs[in_language].groupby("bucket")}
            for e in uplift.NEGATIVE_EMOTIONS:
                for n, r in enumerate(sample[e]):
                    journey = uplift.plan_journey(r["fused"], r["emotions"], centres, scale)
                    base = {"language": language, "start_emotion": e, "reading": n, "face_idx": r["face"],
                            "text_idx": r["text"], "reading_emotions": " ".join(r["emotions"])}
                    queue = app["build_queue"](songs, r["emotions"], r["fused"], "Uplift", language)
                    offered = queue["mode"] == "Uplift"
                    if offered:
                        links = [t["lastfm"] for t in queue["tracks"] if t["section"] == "journey"]
                        waypoints = np.array(queue["journey"]["waypoints"])
                        assert np.allclose(waypoints, journey["waypoints"], atol=0.006)
                    elif language not in app["UPLIFT_LANGUAGES"]:
                        tracks, _ = player.uplift_queue(songs, r["fused"], r["emotions"], pool=in_language)
                        links = [t["lastfm"] for t in tracks or [] if t["section"] == "journey"]
                    else:
                        links = []      # Uplift offered for this language, but no playable song on the path
                    rows = by_link.loc[links].reset_index()
                    # The words the app shows with this path (path card title and subtitle, the mood card's sentence).
                    words = ({"title": player.path_title(queue["journey"]), "subtitle": queue["subtitle"],
                              "message": mood.card(r["fused"], "Uplift", queue["journey"])["message"]} if offered else {})
                    paths.append({**base, "method": "uplift", "offered_by_app": offered, "note": queue["note"] or "", **words,
                                  **path_metrics(rows, journey["waypoints"], scale, e, r["emotions"])})
                    picked = player.playable_songs(songs[in_language], uplift.N_SONGS, np.random.default_rng())
                    rows = by_link.loc[[s.lastfm_url for s, _ in picked]].reset_index()
                    paths.append({**base, "method": "random", **path_metrics(rows, journey["waypoints"], scale, e, r["emotions"])})
                    tracks = player.resonance_queue(buckets, [e])[:uplift.N_SONGS]
                    rows = by_link.loc[[t["lastfm"] for t in tracks]].reset_index()
                    paths.append({**base, "method": "resonance", **path_metrics(rows, journey["waypoints"], scale, e, r["emotions"])})
    P = pd.DataFrame(paths)

    def summary(g):
        c = g[g.complete]
        return pd.Series({
            "paths": len(g), "complete_5": g.complete.mean(), "distinct_links": g.distinct_links.mean(),
            "distinct_title_artist": g.distinct_songs.mean(),
            "first_distance_mean": g.first_distance.mean(), "first_distance_median": g.first_distance.median(),
            "first_within_0.5sd": (g.first_distance <= 0.5).mean(),
            "first_in_start_bucket": g.first_in_start_bucket.astype(float).mean(),
            "first_in_reading_mix": g.first_in_reading.astype(float).mean(),
            "valence_steps_right": c.valence_steps_right.mean(), "valence_all_4_steps": c.valence_all_steps.astype(float).mean(),
            "arousal_steps_right": c.arousal_steps_right.mean(), "arousal_all_4_steps": c.arousal_all_steps.astype(float).mean(),
            "d_valence_mean": c.d_valence.mean(), "d_valence_std": c.d_valence.std(),
            "d_arousal_mean": c.d_arousal.mean(), "d_arousal_std": c.d_arousal.std(),
            "last_brighter": (c.d_valence > 0).mean(), "last_calmer": (c.d_arousal < 0).mean()})

    S = P.groupby(["language", "start_emotion", "method"], sort=False).apply(summary, include_groups=False).reset_index()
    S.to_csv(os.path.join(out_dir, "playlist_check.csv"), index=False, float_format="%.4f")
    P.to_csv(os.path.join(out_dir, "playlist_check_paths.csv"), index=False, float_format="%.4f")

    cols = ["paths", "complete_5", "distinct_title_artist", "first_distance_median", "first_in_start_bucket", "first_in_reading_mix",
            "valence_steps_right", "valence_all_4_steps", "arousal_steps_right", "arousal_all_4_steps",
            "d_valence_mean", "d_valence_std", "d_arousal_mean", "d_arousal_std", "last_brighter", "last_calmer"]
    short = {"paths": "n", "complete_5": "5 songs", "distinct_title_artist": "distinct", "first_distance_median": "d1 med",
             "first_in_start_bucket": "1st=start", "first_in_reading_mix": "1st in mix", "valence_steps_right": "V steps", "valence_all_4_steps": "V all4",
             "arousal_steps_right": "A steps", "arousal_all_4_steps": "A all4", "d_valence_mean": "dV", "d_valence_std": "sd",
             "d_arousal_mean": "dA", "d_arousal_std": "sd ", "last_brighter": "brighter", "last_calmer": "calmer"}
    shares = {"complete_5", "distinct_title_artist", "first_in_start_bucket", "first_in_reading_mix", "valence_steps_right", "valence_all_4_steps",
              "arousal_steps_right", "arousal_all_4_steps", "last_brighter", "last_calmer"}

    def table(df, index):
        t = df.set_index(index)[cols].rename(columns=short)
        for k, v in short.items():
            if k == "paths":
                t[v] = t[v].astype(int)
            elif k in shares:
                t[v] = t[v].map(lambda x: "-" if pd.isna(x) else f"{x:.0%}")
            else:
                t[v] = t[v].map(lambda x: "-" if pd.isna(x) else f"{x:+.2f}" if k.startswith("d_") else f"{x:.2f}")
        return t.to_string()

    log("COLUMNS: n paths | 5 songs = path has 5 songs | distinct = 5 different title+artist | d1 med = median distance of song 1")
    log("from the start point (in Muse standard deviations of valence / arousal) | 1st=start = song 1 is in the start emotion's")
    log("bucket | 1st in mix = song 1 is in the bucket of one of the reading's emotions (>= 15%) | V steps / A steps = share of the 4 steps that move in the planned direction | V all4 / A all4 = all 4 steps do")
    log("(strict) | dV, dA = song 5 minus song 1 (mean, sd) | brighter = song 5 has higher valence than song 1 | calmer = lower")
    log("arousal. Step and change columns use complete paths only. For sad the planned arousal goes down, then up, and the")
    log("last song is meant to be MORE energetic (happy), so 'calmer' is not the aim there.\n")

    supported = P.language.isin(app["UPLIFT_LANGUAGES"])
    for label, mask in [("ALL SUPPORTED LANGUAGES (English, Hindi, Tamil, Korean)", supported)]:
        T = P[mask].groupby(["start_emotion", "method"], sort=False).apply(summary, include_groups=False).reset_index()
        log(f"== {label}: by start emotion")
        log(table(T, ["start_emotion", "method"]))
        log("")
    T = P[supported].groupby(["language", "method"], sort=False).apply(summary, include_groups=False).reset_index()
    log("== BY LANGUAGE (all four start emotions)")
    log(table(T, ["language", "method"]))
    log("")
    for language in CHECK_LANGUAGES:
        sub = S[S.language == language]
        note = "" if language in app["UPLIFT_LANGUAGES"] else \
            "  -- NOT OFFERED BY THE APP (build_queue falls back to Resonance); the Uplift rows are what player.uplift_queue would build"
        log(f"== {language.upper()}{note}")
        log(table(sub, ["start_emotion", "method"]))
        log("")

    # mean valence / arousal by step, Uplift paths in supported languages
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True)
    for col, e in enumerate(uplift.NEGATIVE_EMOTIONS):
        j = uplift.plan_journey({e: 1.0}, [e], centres, scale)
        for row, (k, name) in enumerate([("V", "valence"), ("A", "arousal")]):
            ax = axes[row, col]
            ax.plot(range(1, 6), j["waypoints"][:, row], "k--", lw=1, label="planned (pure reading)")
            for language in sorted(app["UPLIFT_LANGUAGES"], key=CHECK_LANGUAGES.index):
                c = P[(P.method == "uplift") & (P.language == language) & (P.start_emotion == e) & P.complete]
                if len(c):
                    ax.plot(range(1, 6), [c[f"{k}{i}"].mean() for i in range(1, 6)], "-o", ms=3, label=language)
            c = P[(P.method == "random") & (P.start_emotion == e) & supported & P.complete]
            ax.plot(range(1, 6), [c[f"{k}{i}"].mean() for i in range(1, 6)], ":", color="grey", label="random baseline")
            ax.set_title(f"{e}: {name}", fontsize=10); ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("song in path")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Uplift paths: mean valence and arousal of song 1..5 (dataset values; supported languages)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "playlist_check_steps.png"), dpi=120)

    # weak spots, from the numbers
    log("== WEAK SPOTS (from the numbers above)")
    up = S[S.method == "uplift"]
    log("- Telugu and Malayalam: the app does not offer Uplift (too few playable songs near the waypoints); for them")
    for _, r in up[~up.language.isin(app["UPLIFT_LANGUAGES"])].iterrows():
        log(f"    {r.language:9} {r.start_emotion:8} complete {r.complete_5:.0%}, song 1 in the start bucket {r.first_in_start_bucket:.0%}, "
            f"median song-1 distance {r.first_distance_median:.2f}, valence steps right {r.valence_steps_right:.0%}")
    weak = up[up.language.isin(app["UPLIFT_LANGUAGES"]) & ((up.complete_5 < 0.95) | (up.first_in_start_bucket < 0.5)
                                                           | (up.first_distance_median > 0.5) | (up.valence_steps_right < 0.75))]
    for _, r in weak.iterrows():
        why = []
        if r.complete_5 < 0.95: why.append(f"only {r.complete_5:.0%} of paths have 5 songs")
        if r.first_in_start_bucket < 0.5: why.append(f"song 1 in the start bucket only {r.first_in_start_bucket:.0%} "
                                                     f"(in one of the reading's emotions {r.first_in_reading_mix:.0%})")
        if r.first_distance_median > 0.5: why.append(f"song 1 median {r.first_distance_median:.2f} sd from the start")
        if r.valence_steps_right < 0.75: why.append(f"valence steps right only {r.valence_steps_right:.0%}")
        log(f"- {r.language} {r.start_emotion}: " + "; ".join(why))
    if weak.empty:
        log("- (no supported language x start emotion below the thresholds: 95% complete, 50% song 1 in start bucket,")
        log("  median song-1 distance 0.5 sd, 75% valence steps right)")
    log("- Per-step direction is strict (any step against the plan counts as wrong, however small). The planned arousal")
    log("  steps are small for disgust (about -0.12 per step) and for the first half of sad (about -0.14, then +1.0), well")
    log("  within the spread of one bucket, so 'A all4' is low there by construction; dA is the better measure.")
    log("- A reading with several heavy emotions starts between their centres, so song 1 can come from another emotion of")
    log("  the reading's mix: this is how angry's '1st=start' is low while '1st in mix' stays high.")
    log("- The 4 negative buckets overlap in valence / arousal (data/README.md), so 'close to the start mood' is measured")
    log("  in valence / arousal and by bucket; neither says the song sounds sad, angry, fearful or disgusted to a person.")
    log("- Valence / arousal are dataset values: Muse listener tags (English) and Spotify audio features mapped onto")
    log("  the Muse scale by percentile (other languages). They were not checked by people. This checks the playlist's")
    log("  structure, not any effect on a listener.")
    with open(os.path.join(out_dir, "playlist_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {out_dir}")
    progression_tables(P, songs, playable)


# --- The intended progression of each Uplift path, against random and Resonance playlists ---------------
# sad -> neutral -> happy is meant to get BRIGHTER: valence rises at every step.
# angry / fear / disgust -> neutral are meant to get CALMER: arousal falls at every step (their valence
# change is reported too). These match the planned waypoints (uplift.plan_journey), which rise in valence
# for every start emotion and fall in arousal for angry, fear and disgust.
INTENDED = {"sad": ("valence", 1, "brighter"), "angry": ("arousal", -1, "calmer"),
            "fear": ("arousal", -1, "calmer"), "disgust": ("arousal", -1, "calmer")}
RESULTS_DIR = os.path.join("docs", "results")
MIN_PATHS = 30      # a cell with fewer complete paths than this is marked too small to read
SMALL_POOL = 30     # fewer playable songs than this in the start emotion's bucket (that language): marked
BOOTSTRAP = 2000    # paired bootstrap over readings, for the 95% interval of Uplift minus random
METHOD_NAMES = {"uplift": "Uplift", "random": "Random", "resonance": "Resonance"}
MEASURES = [("steps", "steps in the intended direction"), ("all4", "all 4 steps in the intended direction"),
            ("end", "song 5 vs song 1 in the intended direction"), ("dv", "valence change, song 5 minus song 1 (calming paths)")]


def _intended_measures(P):
    """One row per complete path: its measures on the intended axis of its start emotion."""
    P = P[P.complete.astype(bool)].copy()
    axis = P.start_emotion.map(lambda e: INTENDED[e][0])
    sign = P.start_emotion.map(lambda e: INTENDED[e][1]).to_numpy()
    V = P[[f"V{i}" for i in range(1, 6)]].to_numpy(float)
    A = P[[f"A{i}" for i in range(1, 6)]].to_numpy(float)
    X = np.where((axis == "valence").to_numpy()[:, None], V, A)
    right = np.sign(np.diff(X, axis=1)) == sign[:, None]
    P["steps"] = right.mean(axis=1)                  # share of the path's 4 steps
    P["all4"] = right.all(axis=1).astype(float)
    P["end"] = (np.sign(X[:, -1] - X[:, 0]) == sign).astype(float)
    P["dv"] = np.where(P.start_emotion == "sad", np.nan, V[:, -1] - V[:, 0])
    return P


def _compare(G, rng):
    """Uplift, random and Resonance on one group of paths: a list of rows, one per measure."""
    rows = []
    key = ["language", "start_emotion", "reading"]
    by = {m: G[G.method == m].set_index(key) for m in METHOD_NAMES}
    n = {m: len(by[m]) for m in METHOD_NAMES}
    for measure, label in MEASURES:
        vals = {m: by[m][measure].dropna() for m in METHOD_NAMES}
        if not len(vals["uplift"]):
            continue
        paired = pd.concat([vals["uplift"].rename("u"), vals["random"].rename("r")], axis=1).dropna()
        diffs = (paired.u - paired.r).to_numpy()
        boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(BOOTSTRAP)] if len(diffs) else [np.nan]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pct = measure != "dv"
        scale = 100 if pct else 1
        row = {"measure": label, "unit": "%" if pct else "valence points",
               **{METHOD_NAMES[m]: vals[m].mean() * scale for m in METHOD_NAMES},
               "Uplift - Random": (vals["uplift"].mean() - vals["random"].mean()) * scale,
               "95% interval of Uplift - Random": f"{lo * scale:+.1f} to {hi * scale:+.1f}",
               "Uplift - Resonance": (vals["uplift"].mean() - vals["resonance"].mean()) * scale,
               "n paths": len(vals["uplift"]),
               "n paths (random, Resonance)": f"{len(vals['random'])}, {len(vals['resonance'])}",
               "n steps": 4 * len(vals["uplift"]) if measure == "steps" else ""}
        rows.append(row)
    return rows


def _markdown(df, title, notes):
    def cell(v, col):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:+.1f}" if col.startswith("Uplift -") else f"{v:.1f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    body = "".join("| " + " | ".join(cell(v, c) for c, v in zip(df.columns, r)) + " |\n" for r in df.itertuples(index=False))
    return f"# {title}\n\n" + "".join(f"{n}\n" for n in notes) + "\n" + head + body


def progression_tables(P, songs, playable):
    """docs/results/: the intended-progression tables (CSV and Markdown), two plots and the app's wording."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    M = _intended_measures(P[P.language.isin(["English", "Hindi", "Tamil", "Korean"])])
    languages = ["English", "Hindi", "Tamil", "Korean"]
    emotions = list(INTENDED)
    pool = {(lang, e): int(((songs.language == lang).values & (songs.bucket == e).values & playable).sum())
            for lang in languages for e in emotions}
    notes = ["Paths: the app's own Uplift path (`app.build_queue`, saved previews only) for 200 sampled fused readings per start",
             "emotion (HSEmotion + DistilBERT on FER2013 PrivateTest x GoEmotions test pairs), in each language. Random: 5 playable",
             "songs drawn at random from the same language (`player.playable_songs`); Resonance: the first 5 songs of a Resonance",
             "playlist of the start emotion's bucket (`player.resonance_queue`). Same readings and the same seeded generator for all three.",
             "",
             "Intended direction: sad paths should get brighter (valence rises at each step); angry, fear and disgust paths should",
             "get calmer (arousal falls at each step). Steps are strict: a step with no change counts as not in the direction.",
             "Chance for a random path: 50% of steps, 0.8% for all 4 steps (1 in 120 orders), 50% for song 5 vs song 1.",
             "Difference columns are percentage points (valence points for the valence row). The 95% interval is a paired",
             f"bootstrap over readings ({BOOTSTRAP} resamples). Valence and arousal are dataset values (Muse tags; Spotify",
             "features mapped by percentile), not checked by people; this measures playlist structure, not any effect on a listener."]
    tables = {}
    rows = []
    for e in emotions:
        for r in _compare(M[M.start_emotion == e], rng):
            rows.append({"start emotion": e, "intended": INTENDED[e][2] + f" ({INTENDED[e][0]} {'up' if INTENDED[e][1] > 0 else 'down'})", **r})
    tables["by_emotion"] = (pd.DataFrame(rows), "Uplift paths by start emotion (English, Hindi, Tamil and Korean together)")
    rows = []
    for lang in languages:
        for r in _compare(M[M.language == lang], rng):
            rows.append({"language": lang, **r})
    tables["by_language"] = (pd.DataFrame(rows), "Uplift paths by language (all four start emotions; each path on its own intended axis)")
    rows = []
    for e in emotions:
        for lang in languages:
            G = M[(M.start_emotion == e) & (M.language == lang)]
            flags = []
            if (G.method == "uplift").sum() < MIN_PATHS:
                flags.append(f"too small to read (fewer than {MIN_PATHS} paths)")
            if pool[(lang, e)] < SMALL_POOL:
                flags.append(f"small song pool: {pool[(lang, e)]} playable {e} songs")
            for r in _compare(G, rng):
                rows.append({"start emotion": e, "language": lang, **r, "playable songs in start bucket": pool[(lang, e)],
                             "flag": "; ".join(flags)})
    tables["by_emotion_language"] = (pd.DataFrame(rows), "Uplift paths by start emotion and language")

    for name, (df, title) in tables.items():
        df.to_csv(os.path.join(RESULTS_DIR, f"uplift_{name}.csv"), index=False, float_format="%.2f")
        extra = [] if name != "by_emotion_language" else [
            "", f"Flags: a cell is too small to read below {MIN_PATHS} complete paths (none is: every cell has 200); a small song",
            f"pool (fewer than {SMALL_POOL} playable songs in the start bucket in that language) means the paths reuse few songs."]
        with open(os.path.join(RESULTS_DIR, f"uplift_{name}.md"), "w", encoding="utf-8") as f:
            f.write(_markdown(df, title, notes + extra))
    print(f"tables written to {RESULTS_DIR}")
    _plots(tables)
    _wording(P)


def _plots(tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"Uplift": "#257de0", "Random": "#f47230", "Resonance": "#00b894"}   # categorical slots 1-3, validated
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "axes.labelcolor": "#191919",
                         "xtick.color": "#666666", "ytick.color": "#666666"})
    for name, key, ticks in [("by_emotion", "start emotion", lambda g: f"{g}\n{INTENDED[g][2]}\n({INTENDED[g][0]} {'up' if INTENDED[g][1] > 0 else 'down'})"),
                             ("by_language", "language", lambda g: g)]:
        df = tables[name][0]
        groups = list(dict.fromkeys(df[key]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, measure in zip(axes, ["steps in the intended direction", "all 4 steps in the intended direction"]):
            d = df[df.measure == measure].set_index(key)
            x = np.arange(len(groups)); w = 0.26
            for i, m in enumerate(colours):
                vals = [d.loc[g, m] for g in groups]
                bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=colours[m], label=m, edgecolor="white", linewidth=2)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#191919")
            ax.set_xticks(x, [ticks(g) for g in groups]); ax.set_ylim(0, 110); ax.set_yticks(range(0, 101, 25))
            ax.set_ylabel(f"% of {'steps' if measure.startswith('steps') else 'paths'}")
            n = int(d["n paths"].iloc[0])
            ax.set_title(f"{measure[0].upper() + measure[1:]}\n(n = {n} paths per {key}" + (f", {4 * n} steps)" if measure.startswith("steps") else ")"),
                         fontsize=10, color="#191919")
            ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle(f"Uplift vs random vs Resonance, by {key}", fontsize=12, color="#191919")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(RESULTS_DIR, f"uplift_{name}.png"), dpi=150)
        plt.close(fig)
    print(f"plots written to {RESULTS_DIR}")


def _wording(P):
    """The words the app shows with each start emotion's Uplift path (English paths), and any 'brighter' / 'hopeful'."""
    import ui
    U = P[(P.method == "uplift") & (P.language == "English") & P.offered_by_app.astype(bool)]
    lines = ["# What the app says with an Uplift path, by start emotion", "",
             "From the app's own code on the English Uplift paths of the playlist check (200 readings per start emotion).",
             "Other languages add \" in <language>\" to the subtitle. Nothing here was changed.", "",
             "Shown with every Uplift path, whatever the start emotion:", "",
             f"- choice card (button): \"{ui.CHOICES['Uplift'][2]}\", with the text \"{ui.CHOICES['Uplift'][3]}\"",
             "- path card label: \"Your path\"; list headings \"Your path, in play order\" and \"Keep listening\"",
             "- path card play button: no text, labelled \"Play\" for screen readers", ""]
    flagged = []
    for e in INTENDED:
        S = U[U.start_emotion == e]
        lines += [f"## {e} ({len(S)} paths; intended: {INTENDED[e][2]})", ""]
        for col, label in [("title", "Path card title"), ("subtitle", "Subtitle"), ("message", "Mood card sentence (chosen by the reading's top emotion)")]:
            counts = S[col].value_counts()
            lines.append(f"- {label}: " + "; ".join(f"\"{t}\" ({c})" for t, c in counts.items()))
            for t in counts.index:
                for word in ("brighter", "hopeful", "lighter"):
                    if word in t.lower() and INTENDED[e][2] == "calmer":
                        flagged.append(f"{e}: {label.split(' (')[0].lower()} \"{t}\" says \"{word}\"")
        lines.append("")
    lines += ["## \"brighter\", \"hopeful\" or \"lighter\" on a calming path (angry, fear, disgust)", ""]
    lines += [f"- {f}" for f in flagged] or ["- none in the path card title, subtitle or mood card sentence"]
    lines += [f"- the choice card's text \"{ui.CHOICES['Uplift'][3]}\" is the same for every start emotion, calming ones included",
              "- About (shell.py): \"a five-song path from a low mood towards a lighter one\", for every start emotion"]
    with open(os.path.join(RESULTS_DIR, "uplift_wording.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--previews", action="store_true",
                        help="only use songs that have a saved preview (data/songs/previews.csv)")
    parser.add_argument("--language", default="English", help="language to draw songs from (default English)")
    parser.add_argument("--check", action="store_true", help="the playlist check (see the top of this file)")
    parser.add_argument("--tables", action="store_true",
                        help="only rebuild docs/results/ from models/uplift/playlist_check_paths.csv (written by --check)")
    args = parser.parse_args()
    if args.check:
        return playlist_check()
    if args.tables:
        import previews
        songs = pd.read_csv(os.path.join("data", "songs", "songs_clean.csv"), keep_default_na=False)
        P = pd.read_csv(os.path.join("models", "uplift", "playlist_check_paths.csv"), keep_default_na=False,
                        na_values={c: [""] for c in ["V1", "V2", "V3", "V4", "V5", "A1", "A2", "A3", "A4", "A5"]})
        P["complete"] = P.complete.astype(str) == "True"
        P["offered_by_app"] = P.offered_by_app.astype(str) == "True"
        return progression_tables(P, songs, songs.lastfm_url.isin(previews.playable_urls()).values)

    songs = pd.read_csv(os.path.join("data", "songs", "songs_clean.csv"), keep_default_na=False)
    os.makedirs(os.path.join("models", "uplift"), exist_ok=True)
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    # All songs stay in `songs` so the centres come from the English ones; the pool limits the picks.
    in_language = (songs.language == args.language).values
    if not in_language.any():
        parser.error(f"no songs in language {args.language!r}")
    pool = in_language
    if args.language != "English":
        log(f"LANGUAGE: {args.language} ({int(in_language.sum()):,} songs)\n")
    if args.previews:
        import previews
        pool = in_language & songs.lastfm_url.isin(previews.playable_urls()).values
        preview_coverage(songs, in_language, pool, log)
        log()

    centres = uplift.bucket_centres(uplift.reference_songs(songs))
    log("BUCKET CENTRES (valence, arousal)")
    for bucket, (v, a) in sorted(centres.items()):
        log(f"  {bucket:9} {v:.2f}, {a:.2f}")

    cases = [(e, {e: 1.0}, [e]) for e in uplift.NEGATIVE_EMOTIONS]
    cases.append(("mixed: sad 45%, angry 30%, neutral 25%", MIXED, list(MIXED)))

    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 44)
    for name, fused, emotions in cases:
        rng = np.random.default_rng(SEED)
        playlist, journey = uplift.build_playlist(songs, fused, emotions, rng, pool=pool)
        log(f"\n== {name.upper()}: start ({journey['start'][0]:.2f}, {journey['start'][1]:.2f}), "
            f"target {' -> '.join(journey['targets'])}")
        table = playlist[["step", "waypoint_valence", "waypoint_arousal", "title", "artist",
                          "valence_tags", "arousal_tags", "bucket"]].round(2)
        table.columns = ["step", "wp_V", "wp_A", "title", "artist", "V", "A", "bucket"]
        log(table.to_string(index=False))

    log(f"\n== {RUNS} PLAYLISTS PER CASE")
    for name, fused, emotions in cases:
        runs = run_many(songs, fused, emotions, np.random.default_rng(SEED), pool)
        log(f"\n-- {name}")
        full = runs[runs.valence.map(len) == uplift.N_SONGS]
        if args.previews:
            log(f"playlists with all {uplift.N_SONGS} songs:               {len(full) / len(runs):.1%}")
        if full.empty:
            log("no complete playlists: not enough songs with previews along this path")
            continue
        v, a = np.stack(full.valence), np.stack(full.arousal)
        steps_up = (np.diff(v, axis=1) > 0)
        log(f"valence rises at every step (all 4):   {steps_up.all(axis=1).mean():.1%}")
        log(f"valence rises at 3 or more of 4 steps: {(steps_up.sum(axis=1) >= 3).mean():.1%}")
        log(f"last song more positive than first:    {(v[:, -1] > v[:, 0]).mean():.1%}")
        log(f"single steps that rise:                {steps_up.mean():.1%}")
        log("mean valence by song 1..5: " + "  ".join(f"{x:.2f}" for x in v.mean(axis=0)))
        log("mean arousal by song 1..5: " + "  ".join(f"{x:.2f}" for x in a.mean(axis=0)))
        log(f"song 1 bucket: {full.first_bucket.value_counts(normalize=True).round(2).to_dict()} | "
            f"song 5 bucket: {full.last_bucket.value_counts(normalize=True).round(2).to_dict()}")

    if not args.previews:
        log("\n== WHY THE SONG POOL IS RESTRICTED: bucket of song 1 without the restriction")
        uplift.RESTRICT_TO_JOURNEY_BUCKETS = False
        for name, fused, emotions in cases[:4]:
            runs = run_many(songs, fused, emotions, np.random.default_rng(SEED), pool)
            log(f"{name:8} {runs.first_bucket.value_counts(normalize=True).round(2).to_dict()}")
        uplift.RESTRICT_TO_JOURNEY_BUCKETS = True

    report = "report_previews" if args.previews else "report"
    report += ".txt" if args.language == "English" else f"_{args.language.lower()}.txt"
    with open(os.path.join("models", "uplift", report), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
