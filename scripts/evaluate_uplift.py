"""Check that Uplift playlists move the way they are meant to.

For each negative emotion (sad, angry, fear, disgust) this prints one example
playlist with its waypoints, then builds RUNS playlists and reports how the
songs' valence and arousal change from song 1 to song 5.

Usage (from the project root):
    python scripts/evaluate_uplift.py             # all English songs
    python scripts/evaluate_uplift.py --previews  # only songs with a saved preview,
                                                  # which is what the app can play
    python scripts/evaluate_uplift.py --language Hindi [--previews]

Songs are drawn from one language (default English). Bucket centres and the
distance scale always come from the English (Muse) songs, as in the app.

Reads   data/songs/songs_clean.csv  (and data/songs/previews.csv with --previews)
Writes  models/uplift/report.txt    (report_previews.txt with --previews);
        for another language report_<language>.txt / report_previews_<language>.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--previews", action="store_true",
                        help="only use songs that have a saved preview (data/songs/previews.csv)")
    parser.add_argument("--language", default="English", help="language to draw songs from (default English)")
    args = parser.parse_args()

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
