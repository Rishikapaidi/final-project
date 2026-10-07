# Uplift mode

The app has two modes. By default it picks one automatically (Uplift when the
mood is low, otherwise Resonance); the settings popover can force either:

- **Resonance (match my mood):** recommends songs from the bucket of the
  detected emotion. This is the original behaviour.
- **Uplift (gently lift my mood):** builds a five-song playlist that starts at
  the detected mood and steps towards a better one.

The logic is in `uplift.py` and needs neither the webcam nor Streamlit.

## Design

1. **When it applies.** Uplift only runs when sad, angry, fear and disgust
   together hold at least 50% of the fused face + text probability. Otherwise
   (mostly happy, surprised or neutral) the app says so and falls back to
   Resonance.
2. **Start point.** The probability-weighted average of the valence/arousal
   centres of the detected emotions' song buckets (the emotions at 15% or
   more).
3. **Target**, chosen by the strongest negative emotion:

   | Start | Path |
   |---|---|
   | sad | → neutral (calm) → happy |
   | angry, fear, disgust | → neutral (calm) |

4. **Waypoints.** Five points evenly spaced from the start to the target. On
   the sad path, two steps go to neutral and two more to happy, so song 3 sits
   at the calm point.
5. **Songs.** Each waypoint gets the nearest unused song in valence/arousal
   (both scaled by their spread), picked at random among the 10 nearest so
   playlists differ between runs.
6. **Display.** The five songs are listed in play order, numbered 1 to 5, with
   a collapsible "Mood journey" chart of the planned path and the songs picked.
7. **After the five songs.** The app keeps playing songs from the bucket the
   journey ended in (happy after sad; neutral after angry, fear or disgust), so
   the music does not stop. These are ordinary songs from that bucket, not
   further steps of the journey.

In the app, only songs with a playable preview can be picked (see "Song
previews" in `data/README.md`): each waypoint's song comes from its 10 nearest
candidates that have one.

### One departure from "nearest song in the whole file"

Songs are only drawn from the journey's own buckets: the detected emotions
plus the targets. Valence and arousal cannot tell sad, angry, fear and disgust
songs apart, so with the whole file as the pool the opening song was usually
from the wrong bucket:

| Detected | Bucket of song 1, whole file as pool | With the restriction |
|---|---|---|
| sad | 83% disgust, 17% sad | 100% sad |
| angry | 100% fear | 100% angry |
| fear | 83% fear, 9% disgust, 8% angry | 100% fear |
| disgust | 80% disgust, 20% sad | 100% disgust |

`RESTRICT_TO_JOURNEY_BUCKETS` in `uplift.py` switches this off.

## Check

`python scripts/evaluate_uplift.py` builds 500 playlists per emotion and writes
`models/uplift/report.txt`.

| Start | Valence rises at every step | Last song more positive than first | Mean valence, song 1 → 5 | Mean arousal, song 1 → 5 |
|---|---|---|---|---|
| sad | 100.0% | 100% | 3.43 → 7.33 | 3.50 → 3.22 (song 3) → 5.22 |
| angry | 97.4% | 100% | 3.95 → 5.72 | 5.21 → 3.22 |
| fear | 100.0% | 100% | 3.56 → 5.72 | 4.78 → 3.22 |
| disgust | 100.0% | 100% | 2.96 → 5.71 | 3.76 → 3.22 |

Read this as "the code does what it was designed to do", not as evidence that
it lifts anyone's mood: songs are picked to sit close to waypoints whose
valence rises, so a rising sequence is nearly guaranteed. Angry is slightly
lower because its bucket is small (421 songs), so the nearest songs sit
further from the waypoints.

## Limits

- **The targets are design decisions, not findings.** Sad → calm → happy and
  angry/fear/disgust → calm follow the general idea of meeting a listener where
  they are and moving gradually, but these particular targets, the five steps
  and the 50% threshold were chosen for this project. Nothing here has been
  tested with listeners.
- **Valence and arousal come from listener tags, not audio.** They are averages
  of word scores for the mood tags people gave each song (see
  `data/README.md`). A song's position on the chart says how it was described,
  not how it sounds, and a quarter of the songs carry a single word's score.
- **The start point is a bucket average**, so two people who are sad in
  different ways start at the same place.
- **Variety is limited.** Each position draws from its 10 nearest songs, and
  the angry, fear and disgust paths all end at the same calm point, so the
  same closing songs come up often.
- **It inherits the earlier weaknesses**: the emotion detection itself
  (`models/README.md`, `models/fusion/report.txt`) and the song buckets
  (angry is built from proxy words; see `data/README.md`).
- **Song titles are only screened by a word list** (profanity, slurs and
  explicit sexual words; see `data/README.md`), so an occasional crude or oddly
  fitting title can still appear.

## With playable songs only

The app can only play songs that have a saved preview (1,180 so far; see
"Song previews" in `data/README.md`). `python scripts/evaluate_uplift.py
--previews` repeats the check on those songs alone and writes
`models/uplift/report_previews.txt`:

| Start | Valence rises at every step: all songs → playable only | Rises at 3 or more of 4 steps | Last song more positive than first |
|---|---|---|---|
| sad | 100.0% → 100.0% | 100% | 100% |
| angry | 97.4% → 51.6% | 99.4% | 100% |
| fear | 100.0% → 100.0% | 100% | 100% |
| disgust | 100.0% → 100.0% | 100% | 100% |

Every playlist still has all five songs and ends more positive than it began.
Sad, fear and disgust are unaffected: each waypoint still has 50 or more
playable songs close by.

**The angry path is weaker in the app than in the full-file check.** Only 42%
of angry songs have a preview (176 of 421, with the whole bucket checked), so
the nearest playable songs sit further from the first three waypoints, and
songs 2 and 3 overlap (mean valence 4.56 and 4.74). About half of angry
playlists therefore have one step where valence dips slightly instead of
rising, although the overall direction (4.02 to 5.72) holds.
