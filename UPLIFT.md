# Uplift mode

The app has two modes. By default it picks one automatically (Uplift when the
mood is low, otherwise Resonance). Under "What do you need from music?" the
page shows a card for each, tags the one it would pick "Recommended", and
lets the listener choose either. A chosen card stays when the text or the
language changes, and goes back to Auto on a new scan ("Tap to scan again" or
a new photo):

- **Meet me here (Resonance):** recommends songs from the bucket of the
  detected emotion. This is the original behaviour.
- **Lift my mood (Uplift):** builds a five-song playlist that starts at the
  detected mood and steps towards a better one.

The mood card's "X% low" gauge shows the number this choice is made on: the
share of sad, angry, fear and disgust in the fused probabilities, with a
band: **Light** under 25%, **In between** from 25% to 50%, **Low** from 50%
(where Auto picks Uplift). The band is about that share only; the headline's
"Mixed: X and Y" is about the top two emotions, so the band avoids the word.

The logic is in `uplift.py` and needs neither the webcam nor Streamlit.

In the player, the path keeps its order: shuffle is greyed out until the
path has finished and never moves its songs, and songs added with Play next,
Add to queue, or from Explore or the Library go after the current song
without changing the path.

The top bar has **Songs in** (English, Hindi, Tamil, Telugu, Malayalam,
Korean). Uplift is offered for English, Hindi, Tamil and Korean;
see "Other languages" below.

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

The app can only play songs that have a saved preview (1,180 English songs so far; see
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

## Other languages

Songs in Hindi, Tamil and Korean come from a Spotify dataset and were put on
the same valence/arousal scale and into buckets by their distance to the
English buckets' centres (see "Songs in other languages" in
`data/README.md`). Uplift plans every journey with the **English** bucket
centres and spread, whatever the language, and then picks only songs of the
chosen language that have a saved preview; it never borrows songs from
another language.

`python scripts/evaluate_uplift.py --language <language> --previews` repeats
the 500-playlist check and writes `models/uplift/report_previews_<language>.txt`.

The first lookups (up to 30 songs per bucket; see "Song previews" in
`data/README.md`) left thin spots: waypoints with fewer than 10 playable songs
within 0.5 standard deviations, mostly the calm end point every journey but
sad's shares (the neutral centre, 5.71, 3.24: 8 Hindi, 7 Tamil and 2 Korean
songs) and the middle of the angry and fear paths (0 songs near the angry
path's third waypoint in all three languages). `python
scripts/fetch_previews.py --language Hindi Tamil Korean --thin` then looked up
up to 40 more songs near each distinct thin waypoint (480 lookups, 398 with a
preview). Uplift itself and its thresholds were not changed; only the set of
playable songs grew:

| | Hindi | Tamil | Korean |
|---|---|---|---|
| playable songs | 169 → 337 | 180 → 282 | 166 → 294 |
| thin waypoints | 8 → 0 | 6 → 0 | 7 → 0 |
| playable songs near the neutral centre | 8 → 41 | 7 → 43 | 2 → 28 |
| near the angry path's third waypoint | 0 → 103 | 0 → 66 | 0 → 101 |

Results before → after the extra lookups (English for comparison):

| Start | | English | Hindi | Tamil | Korean |
|---|---|---|---|---|---|
| sad | rises at every step | 100.0% | 81.8% → 100.0% | 96.6% → 100.0% | 83.2% → 100.0% |
| | mean valence, song 1 → 5 | 3.44 → 7.33 | 3.39 → 7.41 (unchanged) | 3.45 → 7.33 (unchanged) | 3.43 → 7.35 (unchanged) |
| angry | rises at every step | 51.6% | 21.0% → 85.4% | 13.4% → 47.8% | 19.4% → 72.4% |
| | rises at 3 or more of 4 steps | 99.4% | 73.8% → 100% | 65.2% → 94.0% | 71.0% → 100% |
| | last song's mean valence | 5.72 | 5.49 → 5.75 | 5.38 → 5.73 | 4.86 → 5.66 |
| fear | rises at every step | 100.0% | 35.4% → 97.2% | 46.8% → 77.2% | 22.0% → 90.4% |
| | rises at 3 or more of 4 steps | 100% | 92.6% → 100% | 97.6% → 100% | 72.4% → 100% |
| | last song's mean valence | 5.72 | 5.49 → 5.75 | 5.42 → 5.73 | 4.90 → 5.66 |
| disgust | rises at every step | 100.0% | 49.8% → 60.4% | 76.8% → 83.4% | 45.8% → 73.6% |
| | rises at 3 or more of 4 steps | 100% | 90.6% → 98.4% | 99.4% → 99.6% | 88.6% → 97.8% |
| | last song's mean valence | 5.72 | 5.51 → 5.75 | 5.43 → 5.74 | 4.90 → 5.67 |
| mixed (sad 45%, angry 30%, neutral 25%) | rises at every step | 100.0% | 66.2% → 100.0% | 85.4% → 100.0% | 59.8% → 100.0% |

In every language, before and after, every playlist has all five songs and
ends more positive than it began (100%). Limits that remain:

- **Disgust and Tamil angry still dip more often than in English** (disgust
  rises at every step in 60–84% of playlists, Tamil angry in 48%). No
  waypoint is thin any more by the 0.5-sd rule, so this is about where the
  playable songs sit inside that radius, not how many there are; the overall
  direction holds (last song's mean valence 5.7 in all three).
- **About 280–340 playable songs per language**, against 1,180 for English,
  so the same songs come up more often, especially at the shared calm end.
- **Telugu and Malayalam are Resonance only** (240 and 193 playable songs in
  all, as few as 6 in a bucket). If Uplift would apply, the app plays
  Resonance and says "Uplift needs more songs than we have in <language>".
- The song placements themselves are weaker than for English (an audio-based
  score mapped onto a tag-based scale; see `data/README.md`), so a song's spot
  on the journey is a rougher guess.
