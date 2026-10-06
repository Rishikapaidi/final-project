"""Check that Uplift playlists move the way they are meant to.

For each negative emotion (sad, angry, fear, disgust) this prints one example
playlist with its waypoints, then builds RUNS playlists and reports how the
songs' valence and arousal change from song 1 to song 5.

Usage (from the project root):
    python scripts/evaluate_uplift.py

Reads   data/songs/songs_clean.csv
Writes  models/uplift/report.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uplift  # noqa: E402

RUNS = 500
SEED = 42
MIXED = {"sad": 0.45, "angry": 0.30, "neutral": 0.25}


def run_many(songs, fused, emotions, rng):
    rows = []
    for _ in range(RUNS):
        playlist, _ = uplift.build_playlist(songs, fused, emotions, rng)
        rows.append({"valence": playlist.valence_tags.values, "arousal": playlist.arousal_tags.values,
                     "first_bucket": playlist.bucket.iloc[0], "last_bucket": playlist.bucket.iloc[-1]})
    return pd.DataFrame(rows)


def main():
    songs = pd.read_csv(os.path.join("data", "songs", "songs_clean.csv"), keep_default_na=False)
    os.makedirs(os.path.join("models", "uplift"), exist_ok=True)
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    centres = uplift.bucket_centres(songs)
    log("BUCKET CENTRES (valence, arousal)")
    for bucket, (v, a) in sorted(centres.items()):
        log(f"  {bucket:9} {v:.2f}, {a:.2f}")

    cases = [(e, {e: 1.0}, [e]) for e in uplift.NEGATIVE_EMOTIONS]
    cases.append(("mixed: sad 45%, angry 30%, neutral 25%", MIXED, list(MIXED)))

    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 44)
    for name, fused, emotions in cases:
        rng = np.random.default_rng(SEED)
        playlist, journey = uplift.build_playlist(songs, fused, emotions, rng)
        log(f"\n== {name.upper()}: start ({journey['start'][0]:.2f}, {journey['start'][1]:.2f}), "
            f"target {' -> '.join(journey['targets'])}")
        table = playlist[["step", "waypoint_valence", "waypoint_arousal", "title", "artist",
                          "valence_tags", "arousal_tags", "bucket"]].round(2)
        table.columns = ["step", "wp_V", "wp_A", "title", "artist", "V", "A", "bucket"]
        log(table.to_string(index=False))

    log(f"\n== {RUNS} PLAYLISTS PER CASE")
    for name, fused, emotions in cases:
        runs = run_many(songs, fused, emotions, np.random.default_rng(SEED))
        v, a = np.stack(runs.valence), np.stack(runs.arousal)
        steps_up = (np.diff(v, axis=1) > 0)
        log(f"\n-- {name}")
        log(f"valence rises at every step (all 4):   {steps_up.all(axis=1).mean():.1%}")
        log(f"valence rises at 3 or more of 4 steps: {(steps_up.sum(axis=1) >= 3).mean():.1%}")
        log(f"last song more positive than first:    {(v[:, -1] > v[:, 0]).mean():.1%}")
        log(f"single steps that rise:                {steps_up.mean():.1%}")
        log("mean valence by song 1..5: " + "  ".join(f"{x:.2f}" for x in v.mean(axis=0)))
        log("mean arousal by song 1..5: " + "  ".join(f"{x:.2f}" for x in a.mean(axis=0)))
        log(f"song 1 bucket: {runs.first_bucket.value_counts(normalize=True).round(2).to_dict()} | "
            f"song 5 bucket: {runs.last_bucket.value_counts(normalize=True).round(2).to_dict()}")

    log("\n== WHY THE SONG POOL IS RESTRICTED: bucket of song 1 without the restriction")
    uplift.RESTRICT_TO_JOURNEY_BUCKETS = False
    for name, fused, emotions in cases[:4]:
        runs = run_many(songs, fused, emotions, np.random.default_rng(SEED))
        log(f"{name:8} {runs.first_bucket.value_counts(normalize=True).round(2).to_dict()}")
    uplift.RESTRICT_TO_JOURNEY_BUCKETS = True

    with open(os.path.join("models", "uplift", "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
