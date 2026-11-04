"""Fill the preview cache (data/songs/previews.csv) ahead of time.

The iTunes Search API allows roughly 20 requests a minute, far fewer than the
app needs for one scan, so this looks songs up slowly in the background. It is
safe to stop and rerun: songs already in the cache are skipped.

Songs are checked in a useful order: first the ones Uplift is most likely to
pick (nearest to each journey's waypoints), then the rest, taking turns between
buckets. Waypoints always come from the English songs' bucket centres, as
in the app.

Without --language only the English songs are looked up. With --language, up
to PER_BUCKET songs per bucket of each named language are taken in that same
order, except that every song of a language in LOOK_UP_EVERY_SONG is taken;
copies of a song already taken (same title and artist) are skipped.

Usage (from the project root):
    python scripts/fetch_previews.py              # every English song (about 9 hours)
    python scripts/fetch_previews.py --limit 1500 # the first 1,500 unchecked songs
    python scripts/fetch_previews.py --bucket angry  # only one bucket
    python scripts/fetch_previews.py --language Hindi Tamil Korean Telugu Malayalam
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import previews  # noqa: E402
import uplift  # noqa: E402

SECONDS_BETWEEN_REQUESTS = 3.5
FIRST_WAIT = 120      # seconds to wait after the API refuses a request,
LONGEST_WAIT = 1800   # doubling each time up to this
MAX_ERRORS_PER_SONG = 3
SKIP = object()
NEAR_WAYPOINT = 30
PER_BUCKET = 30   # with --language: songs per language and bucket
# Too few songs for Uplift, so every song is looked up for Resonance.
LOOK_UP_EVERY_SONG = {"Telugu", "Malayalam"}


def lookup_order(songs, language="English"):
    """Positions of one language's songs: nearest to Uplift waypoints first, then round-robin by bucket."""
    points = songs[["valence_tags", "arousal_tags"]].values
    scale = uplift.reference_scale(songs)
    centres = uplift.bucket_centres(uplift.reference_songs(songs))
    in_language = (songs.language == language).values
    first = []
    for emotion in uplift.NEGATIVE_EMOTIONS:
        journey = uplift.plan_journey({emotion: 1.0}, [emotion], centres, scale)
        pool = np.flatnonzero(songs.bucket.isin({emotion, *journey["targets"]}).values & in_language)
        for waypoint in journey["waypoints"]:
            distance = np.linalg.norm((points[pool] - waypoint) / scale, axis=1)
            first.extend(pool[np.argsort(distance)[:NEAR_WAYPOINT]])
    rng = np.random.default_rng(0)
    per_bucket = [list(rng.permutation(np.flatnonzero((songs.bucket == b).values & in_language)))
                  for b in sorted(songs.bucket[in_language].unique())]
    rest = [group[i] for i in range(max(map(len, per_bucket))) for group in per_bucket if i < len(group)]
    return list(dict.fromkeys(list(first) + rest))


def language_sample(songs, language):
    """Up to PER_BUCKET songs per bucket (all in LOOK_UP_EVERY_SONG), skipping repeated title+artist."""
    taken, seen, per_bucket = [], set(), {}
    for i in lookup_order(songs, language):
        key = (songs.title[i].strip().lower(), songs.artist[i].strip().lower())
        bucket = songs.bucket[i]
        if key in seen or (language not in LOOK_UP_EVERY_SONG and per_bucket.get(bucket, 0) >= PER_BUCKET):
            continue
        seen.add(key)
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
        taken.append(i)
    return taken


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="stop after this many lookups")
    parser.add_argument("--bucket", default=None, help="only look up songs from this bucket, e.g. angry")
    parser.add_argument("--language", nargs="+", default=None,
                        help=f"look up a sample of these languages' songs ({PER_BUCKET} per bucket) instead of English")
    args = parser.parse_args()

    songs = pd.read_csv(os.path.join("data", "songs", "songs_clean.csv"), keep_default_na=False)
    if args.language:
        missing = set(args.language) - set(songs.language)
        if missing:
            parser.error(f"no songs in {sorted(missing)}")
        order = [i for language in args.language for i in language_sample(songs, language)]
    else:
        order = lookup_order(songs)
    todo = [i for i in order if previews.cached(songs.lastfm_url[i]) is None]
    if args.bucket:
        todo = [i for i in todo if songs.bucket[i] == args.bucket]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(songs) - len(todo):,} songs already checked or not requested; looking up {len(todo):,}", flush=True)

    found = 0
    wait = FIRST_WAIT
    for n, i in enumerate(todo, 1):
        song = songs.iloc[i]
        errors = 0
        while True:
            try:
                result = previews.search_itunes(song.title, song.artist)
                wait = FIRST_WAIT
                break
            except previews.RateLimited:
                # Retrying too soon seems to prolong the block, so wait longer each time.
                print(f"rate limited, waiting {wait // 60} min", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, LONGEST_WAIT)
            except Exception as error:  # network hiccup: wait and retry a few times
                errors += 1
                print(f"error ({type(error).__name__}), waiting 20s", flush=True)
                time.sleep(20)
                if errors >= MAX_ERRORS_PER_SONG:
                    result = SKIP
                    break
        if result is SKIP:  # left unchecked, so a later run tries it again
            continue
        previews.remember(song.lastfm_url, result)
        found += result is not None
        if n % 25 == 0 or n == len(todo):
            print(f"{n:,}/{len(todo):,} looked up | {found:,} with a preview ({found / n:.0%})", flush=True)
        time.sleep(SECONDS_BETWEEN_REQUESTS)


if __name__ == "__main__":
    main()
