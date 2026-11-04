"""Uplift mode: a short playlist that moves from the detected mood towards a better one.

Resonance mode recommends songs that match the detected emotion. Uplift mode
instead starts where the listener is and steps towards a target mood:

    sad                   -> neutral (calm) -> happy
    angry, fear, disgust  -> neutral (calm)

The start point is the probability-weighted average of the valence/arousal
centres of the detected emotions' song buckets. N_SONGS waypoints are spaced
evenly from the start to the target, and each one gets the nearest unused song,
picked at random among the NEAREST_K closest so playlists vary between runs.

Songs are only drawn from the journey's own buckets (the detected emotions
plus the targets). Valence and arousal alone cannot tell sad, angry, fear and
disgust songs apart, so without this a sad listener would mostly open with
"disgust" songs and an angry one with "fear" songs.

Nothing here needs the webcam or Streamlit: pass in the song table and the
fused emotion probabilities.
"""
import numpy as np

NEGATIVE_EMOTIONS = ["sad", "angry", "fear", "disgust"]
TARGETS = {
    "sad": ["neutral", "happy"],
    "angry": ["neutral"],
    "fear": ["neutral"],
    "disgust": ["neutral"],
}
N_SONGS = 5
NEAREST_K = 10
# With a `playable` check, how far down the nearest-songs list to look before
# giving up on a waypoint (checked NEAREST_K songs at a time).
MAX_CANDIDATES = 40
# Uplift only runs when the negative emotions hold at least this much of the
# fused probability; otherwise the caller should fall back to Resonance.
MIN_NEGATIVE_SHARE = 0.5
# Draw songs only from the detected emotions' buckets plus the target buckets.
RESTRICT_TO_JOURNEY_BUCKETS = True


def negative_share(fused):
    return sum(fused.get(e, 0.0) for e in NEGATIVE_EMOTIONS)


def should_uplift(fused):
    return negative_share(fused) >= MIN_NEGATIVE_SHARE


def bucket_centres(songs):
    """Mean (valence, arousal) of each bucket's songs."""
    means = songs.groupby("bucket")[["valence_tags", "arousal_tags"]].mean()
    return {bucket: row.values for bucket, row in means.iterrows()}


def reference_songs(songs):
    """The songs bucket centres and the distance scale come from: the English (Muse) ones.

    Their buckets come from listeners' mood words; the other languages' songs
    were placed by distance to these same centres (scripts/prepare_songs.py).
    """
    return songs[songs.language == "English"] if "language" in songs else songs


def reference_scale(songs):
    """(valence std, arousal std) of the reference songs."""
    return reference_songs(songs)[["valence_tags", "arousal_tags"]].values.std(axis=0)


def plan_journey(fused, emotions, centres, scale):
    """Work out the start point, targets and waypoints.

    fused:    {emotion: probability} from fusion
    emotions: the detected emotions used for the playlist (those at 15% or more)
    scale:    (valence std, arousal std), so both count equally in distances
    """
    weights = np.array([fused[e] for e in emotions])
    start = np.average([centres[e] for e in emotions], axis=0, weights=weights)
    start_emotion = max(NEGATIVE_EMOTIONS, key=lambda e: fused.get(e, 0.0))
    targets = TARGETS[start_emotion]

    # Evenly spaced along each leg, so every target is itself a waypoint.
    corners = [start] + [centres[t] for t in targets]
    steps_per_leg = (N_SONGS - 1) // len(targets)
    waypoints = [start]
    for a, b in zip(corners[:-1], corners[1:]):
        waypoints += [a + (b - a) * k / steps_per_leg for k in range(1, steps_per_leg + 1)]
    return {"start": start, "start_emotion": start_emotion, "targets": targets,
            "waypoints": np.array(waypoints), "scale": np.asarray(scale)}


def build_playlist(songs, fused, emotions, rng=None, playable=None, pool=None):
    """Return (playlist, journey), or (None, None) if Uplift does not apply.

    songs needs the columns bucket, valence_tags and arousal_tags. The playlist
    is N_SONGS rows of `songs` in play order, with a `step` column (1..N_SONGS)
    and the waypoint each song was picked for.

    pool: optional boolean array, one per row of `songs`; only True rows may be
    picked. The start point, targets and waypoints still come from the
    reference (English) songs, whatever the pool.

    playable: optional function that takes row positions in `songs` and returns
    the ones that can be played. Each waypoint then gets a song from the
    NEAREST_K nearest playable candidates; if none of those is playable the next
    NEAREST_K are tried, and a waypoint with no playable song nearby is skipped.
    """
    if not emotions or not should_uplift(fused):
        return None, None
    rng = rng or np.random.default_rng()

    points = songs[["valence_tags", "arousal_tags"]].values
    scale = reference_scale(songs)
    journey = plan_journey(fused, emotions, bucket_centres(reference_songs(songs)), scale)

    allowed = np.ones(len(songs), dtype=bool)
    if RESTRICT_TO_JOURNEY_BUCKETS:
        buckets = set(emotions) | {journey["start_emotion"]} | set(journey["targets"])
        allowed = songs.bucket.isin(buckets).to_numpy(copy=True)
    if pool is not None:
        allowed &= np.asarray(pool, dtype=bool)

    chosen, used_waypoints = [], []
    for waypoint in journey["waypoints"]:
        candidates = rng.permutation(np.flatnonzero(allowed))  # random order breaks ties
        distance = np.linalg.norm((points[candidates] - waypoint) / scale, axis=1)
        nearest = candidates[np.argsort(distance, kind="stable")]
        if not len(nearest):
            continue
        if playable is None:
            pick = rng.choice(nearest[:NEAREST_K])
        else:
            pick = None
            for first in range(0, MAX_CANDIDATES, NEAREST_K):
                ok = playable(nearest[first:first + NEAREST_K])
                if len(ok):
                    pick = rng.choice(ok)
                    break
            if pick is None:
                continue
        chosen.append(pick)
        used_waypoints.append(waypoint)
        allowed[pick] = False

    playlist = songs.iloc[chosen].copy().reset_index(drop=True)
    playlist.insert(0, "step", range(1, len(playlist) + 1))
    playlist["waypoint_valence"] = [w[0] for w in used_waypoints]
    playlist["waypoint_arousal"] = [w[1] for w in used_waypoints]
    return playlist, journey


def trajectory_figure(playlist, journey):
    """Small valence/arousal chart: the planned path and the songs picked along it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    way = journey["waypoints"]
    ax.plot(way[:, 0], way[:, 1], "--", color="grey", lw=1, label="planned path")
    ax.plot(playlist.valence_tags, playlist.arousal_tags, "-o", color="tab:blue", label="songs")
    for _, song in playlist.iterrows():
        ax.annotate(str(song.step), (song.valence_tags, song.arousal_tags),
                    textcoords="offset points", xytext=(6, 6), fontsize=10, weight="bold")
    ax.set_xlabel("valence (negative → positive)")
    ax.set_ylabel("arousal (calm → energetic)")
    ax.set_title(f"{journey['start_emotion']} → {' → '.join(journey['targets'])}")
    ax.margins(0.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
