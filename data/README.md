# Data

The dataset files in this folder are **not committed to git** (see `.gitignore`).
They are generated from public sources by the scripts in `scripts/`.

The one exception is `data/songs/songs_clean.csv`, which is committed so the app
runs straight after cloning.

## Text emotion: `data/text_emotion/`

Built from [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
(Reddit comments labelled with 28 emotions, "simplified" config) by:

```
python scripts/prepare_go_emotions.py
```

Run it from the project root with the virtual environment active. It needs an
internet connection and takes under a minute.

The script:

1. **Downloads** the train, validation and test splits from Hugging Face.
2. **Writes `go_emotions.csv`:** all 54,263 comments, with columns `text` and
   `emotion`. Comments with several labels have them comma-joined
   (e.g. `desire,optimism`). This file is kept as a backup of the full dataset.
3. **Filters** to the 45,446 comments that have exactly one label.
4. **Maps** each of the 28 GoEmotions labels to one of the 7 FER2013 categories
   used by the face model (table below).
5. **Writes `go_emotions_single.csv`** with these columns:

| Column | Meaning |
|---|---|
| `text` | The Reddit comment |
| `emotion` | Original GoEmotions label (one of 28) |
| `split` | Official Hugging Face split: `train`, `validation` or `test` |
| `emotion_mapped` | FER2013 category (one of 7) |

Use `train` for training, `validation` for tuning, and keep `test` untouched
for final evaluation.

### 28 → 7 mapping

Based on the GoEmotions authors' Ekman grouping, except that **approval** and
**disapproval** go to `neutral` instead of `happy` / `angry`.

| FER2013 | GoEmotions labels |
|---|---|
| angry | anger, annoyance |
| disgust | disgust |
| fear | fear, nervousness |
| happy | admiration, amusement, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| sad | disappointment, embarrassment, grief, remorse, sadness |
| surprise | confusion, curiosity, realization, surprise |
| neutral | neutral, approval, disapproval |

### Resulting class counts

| emotion_mapped | train | validation | test |
|---|---|---|---|
| neutral | 16,098 | 2,062 | 2,037 |
| happy | 11,047 | 1,410 | 1,367 |
| surprise | 3,553 | 435 | 449 |
| angry | 2,476 | 273 | 325 |
| sad | 2,121 | 241 | 259 |
| fear | 515 | 66 | 77 |
| disgust | 498 | 61 | 76 |

**Weak categories:** `fear` and `disgust` have only about 500 training and 76
test examples each. Use class weights or oversampling, and treat their test
scores as approximate (about ±11 percentage points). Only 20% of `surprise` is
the original "surprise" label; the rest is curiosity, confusion and realization.

### Checking a regenerated copy

If the Hugging Face source hasn't changed, the script produces byte-identical
files. SHA-256 of the versions used in this project:

```
ef147d8d290a27e7dd2bd392a3eec875e054a4bde24c8ce2294b6e78b81c31ae  go_emotions.csv
9c5fff0866b603f0f5180663bae2a9b7757ac55003d71564e29b540bf38c9646  go_emotions_single.csv
```

## Face emotion: `data/face_emotion/raw/`

FER2013 (48×48 grayscale faces, 7 emotions) from the Hugging Face copy
[Aaryan333/fer2013_train_publicTest_privateTest](https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest),
which keeps the official splits:

```
python scripts/prepare_fer2013.py
```

The script stops with an error unless the copy matches official FER2013:
35,887 images; Training 28,709 / PublicTest 3,589 / PrivateTest 3,589; and
class counts Angry 4,953, Disgust 547, Fear 5,121, Happy 8,989, Sad 6,077,
Surprise 4,002, Neutral 6,198. Its PublicTest and PrivateTest sets were also
checked image for image against a second upload (`AutumnQiu/fer2013`).

Known FER2013 quirks, left as-is: 1,853 exact duplicate images (57 duplicate
groups with conflicting labels), 13 near-blank images, and 288 PrivateTest
images that are identical to Training images.

## Fusion cache: `data/fusion/`

Face- and text-model probabilities on the validation and test splits, written
by `scripts/evaluate_fusion.py` so reruns skip model inference. Delete the
folder to recompute.

## Songs: `data/songs/songs_clean.csv`

The song list the app recommends from: 10,901 English songs from Muse and
25,815 songs in Hindi, Tamil, Telugu, Malayalam and Korean from a Spotify
dataset (see "Songs in other languages" below). It is committed to git
(7.4 MB), so there is nothing to run after cloning. To rebuild it from
`muse_v3.csv` (kept untouched in the project root) and
`data/songs/raw/spotify_tracks.csv` (not committed; without it only the
English songs are written):

```
python scripts/prepare_songs.py
```

Columns: `title`, `artist`, `valence_tags` (positivity, about 0.2–8.5),
`arousal_tags` (energy, about 0.1–7.3), `seeds` (mood words; empty for the
Spotify songs), `bucket`, `lastfm_url` (the song's link: Last.fm for Muse
songs, Spotify for the others; the name is kept because the app and the
preview cache look songs up by it), `link_source` ("Last.fm" or "Spotify"),
`language`, `artwork_url` (Spotify songs only), and `spotify_valence` /
`spotify_energy` (the original 0–1 Spotify values, Spotify songs only).

The sections below up to "Songs in other languages" describe the English
(Muse) songs.

### Why the original buckets were wrong

The app used to sort `muse_v3.csv` by `number_of_emotion_tags` and then
valence, and cut it into five slices of 18,000 rows called sad, fear, angry,
neutral and happy. Two problems:

- **The slices were sorted by tag count, not emotion.** "Happy" was simply the
  songs with six or more tags, and "melancholy" and "sad" were among its most
  common mood words. Four of the five slices had almost the same average
  valence and arousal.
- **About 64% of the rows have overwritten titles.** 57,955 of the 90,001 rows
  had their title, artist and link replaced (mostly with Hindi film and rap
  songs, with Spotify links), while the mood words, valence, arousal and genre
  of the original song were left in place. Those rows contain only 2,162
  distinct songs, each repeated up to 174 times with different mood values, so
  their mood data says nothing about the song named. Only the 32,046 rows that
  still have a Last.fm link are intact, and only those are used.

### How the buckets are built

A song goes into the bucket whose mood words appear most often in its `seeds`
(ties are dropped), and it must also pass that bucket's valence/arousal check.
Songs with none of the listed words are dropped; assigning them from valence
and arousal alone put songs in the wrong bucket. The word lists and checks are
in `scripts/prepare_songs.py`.

| Bucket | Songs | Built from | Valence/arousal check |
|---|---|---|---|
| happy | 2,055 | happy, fun, playful, positive, cheerful, … | V ≥ 6.0 and A ≥ 4.0 |
| sad | 1,857 | gloomy, lonely, bleak, sad, melancholy, … | V ≤ 5.0 and A ≤ 4.5 |
| angry | 421 | harsh, urgent, gritty, defiant, feral, … | V ≤ 5.0 and A ≥ 4.0 |
| fear | 2,483 | scary, eerie, spooky, paranoid, ominous, … | V ≤ 5.0 |
| disgust | 1,170 | sarcastic, sardonic, cynical, grim, … | V ≤ 4.5 |
| surprise | 1,029 | eccentric, exotic, quirky, whimsical, … | V ≥ 5.0 and A ≥ 4.5 |
| neutral | 1,886 | soothing, meditative, relaxed, gentle, calm, … | 4.0 ≤ V ≤ 6.8 and A ≤ 4.2 |

10,901 songs in total. They are mostly Western; the Hindi songs were all in
the overwritten rows.

### Profanity filter

Songs whose title or artist contains strong profanity, a slur, the words
rape, raped or porn, or an explicit anatomy or sex-act word (penis, anal,
blowjob, …) are dropped (103 songs: 30 from disgust, 30 fear, 19 happy,
8 angry, 7 sad, 6 surprise, 3 neutral). The word list (`PROFANITY` in the
script) is matched on whole words only, so names like "Cockburn",
"Shub-Niggurath" and "Shitao" are kept. Whole-word matching misses compounds,
so the ones found in the data ("Goatwhore", "Shitloads" and four others) are
listed explicitly.

Deliberately not on the list: milder or ambiguous words (hell, damn, ass,
bastard, dick, cock, piss, crap, prick) and broader sexual words (sex, sexy,
sexual, naked, lust, virgin, bondage, hooker, balls), which in this data are
mostly band names and ordinary titles such as "Sexual Healing" or John Lee
Hooker. Run `python scripts/prepare_songs.py --list-dropped` to see exactly
which songs were removed.

### Known weaknesses

- **Angry is the weakest bucket.** Every song tagged angry, aggressive,
  hostile or fierce is in the overwritten rows, so the bucket is built from
  proxy words only ("bitter" was tried and removed: resentful, not angry).
  Valence and arousal cannot separate angry from fear at all (a classifier on
  those two numbers scores no better than a coin flip); only the mood words do.
- **Surprise is really "quirky/unusual".** Nearly all of it comes from two
  words, eccentric and exotic. It is unexpected-sounding music, not music that
  expresses being startled.
- **Disgust is contempt.** It is built from sardonic, cynical and sarcastic
  songs, the closest thing the data has.
- **Neutral means calm.** No mood word means "neutral", so calm, unobtrusive
  music stands in for it.
- **Valence and arousal come from listener tags, not audio.** For songs with a
  single emotion tag (about a quarter) the values are just that word's score.

### Possible future work

About 1,870 overwritten rows tagged angry, aggressive, hostile or outraged
still carry the original song's `mbid` or `spotify_id`. Their real titles and
artists could be looked up through MusicBrainz (free, no account) to give the
angry bucket genuinely angry songs. This has not been tried.

### Songs in other languages

**Source:** "Spotify Tracks Dataset (Updated Every Week)" by gauthamvijayaraj on
Kaggle, licensed Apache 2.0, downloaded 2026-10-28 to
`data/songs/raw/spotify_tracks.csv` (62,317 rows, not committed).

**Which rows:** languages Hindi, Tamil, Telugu, Malayalam and Korean.
Skipped: language "Unknown" (13,005 rows), rows with -1 for valence or energy
(12 in these languages), repeated `track_id`s (75), and 18 songs caught by the
same profanity filter as above. 25,815 songs are kept: Tamil 12,604,
Korean 6,873, Hindi 5,735, Telugu 321, Malayalam 282. "English" stays the Muse
songs.

**Conversion to the Muse scale.** Spotify's valence and energy run from 0 to 1
and are computed from the audio; the Muse values are averages of listener mood
tags. Each Spotify song gets the Muse value at the same percentile: a song at
the 40th percentile of Spotify valence (across all 25,815 added songs) gets the
40th percentile of the Muse songs' `valence_tags`, and energy is mapped onto
`arousal_tags` the same way. Differences between languages are kept: Tamil
songs end up more positive on average and Korean songs more energetic.

**Buckets.** These songs have no mood words, so each goes to the nearest
bucket centre: the mean valence/arousal of the Muse songs in that bucket, with
distance scaled by the Muse songs' spread (the same distance Uplift uses).

One rule on top: a song may sit in sad, angry, fear or disgust only if its
Spotify valence is below 0.5 (`NEGATIVE_MAX_VALENCE` in
`scripts/prepare_songs.py`: the midpoint of Spotify's 0–1 scale, not tuned);
otherwise it goes to the nearest of happy, surprise and neutral. Without it,
mid-valence, high-energy songs landed in angry, because the Muse angry centre
(valence 4.01) maps to Spotify valence of about 0.5: 53–83% of each language's
angry songs had Spotify valence of 0.5 or more, including much of the
energetic K-pop (Korean angry had 2,153 songs; now 1,007). The rule moved
4,675 songs, mostly into surprise and neutral. It applies to the added songs
only; the Muse buckets are unchanged.

| Language | happy | sad | angry | fear | disgust | surprise | neutral |
|---|---|---|---|---|---|---|---|
| Hindi | 697 | 951 | 220 | 538 | 1,026 | 1,196 | 1,107 |
| Tamil | 2,101 | 1,585 | 535 | 601 | 964 | 2,124 | 4,694 |
| Telugu | 45 | 48 | 24 | 32 | 52 | 69 | 51 |
| Malayalam | 33 | 38 | 10 | 23 | 43 | 78 | 57 |
| Korean | 775 | 471 | 1,007 | 815 | 723 | 2,343 | 739 |

**Weaknesses.**

- **The two scales only loosely agree.** 45 songs appear in both Muse and the
  Spotify file; on those, Spotify valence correlates with Muse valence at
  r = 0.50 and Spotify energy with Muse arousal at r = 0.28. That is too few
  songs to fit a better mapping, and the conversion is an assumption.
- **The negative buckets are ambiguous.** Valence and arousal cannot tell sad,
  angry, fear and disgust apart (see "Known weaknesses" above); 43% of the
  added songs in those buckets were almost as close to a second bucket centre
  (measured before the valence rule above). Angry in particular now means
  "energetic and not cheerful" rather than angry: Korean angry is mostly
  high-energy dance and live tracks with Spotify valence 0.2–0.5.
- **"Neutral" means calm and mildly positive**, the same centre as for the Muse
  songs (valence 5.71, arousal 3.24). It includes some clearly happy
  low-energy songs (for example Ilaiyaraaja's "Jealous", Spotify valence 0.97,
  energy 0.04), and the valence rule sends more songs there.
- **The dataset's language labels are not verified.** Some are wrong: for
  example "Yemainado" (a Telugu song by Thaman S) is labelled Hindi, and
  "Ninthare Naanu" (a Kannada song) is labelled Malayalam.
- **Repeated songs.** 5,995 added songs (23%) have the same title and artist as
  another added song under a different `track_id` (the same recording on
  several albums or compilations); for 219 songs the copies sit in different buckets.
  They are kept.
- **Telugu and Malayalam are small** (321 and 282 songs), which is too few for
  Uplift journeys.
- **The profanity filter is an English word list.** It does not cover slurs or
  profanity in Hindi, Tamil, Telugu, Malayalam or Korean, written in Latin
  script or otherwise.

## Song previews: `data/songs/previews.csv`

The song file only has Last.fm and Spotify links, which cannot be played. The app looks
each song up in the free iTunes Search API (no key) to get a 30-second preview
clip and album art, and stores the answer here, including "no preview found",
so each song is only ever looked up once. The file is committed, so the app
can play music straight after cloning without calling the API.

Columns: `lastfm_url`, `preview_url`, `artwork_url`, `itunes_title`,
`itunes_artist` (the last four are empty when no match was found).

A result is accepted only if the title loosely matches (ignoring case, accents
and punctuation, or sharing a word) and so does the artist: the whole artist
credit, or failing that its first-listed artist (for credits such as
"G. V. Prakash, Saindhavi", which iTunes spells "G.V. Prakash Kumar, Saindhavi
& ...").

So far 1,721 of the 10,901 English songs have been checked and 1,180 have a preview:

| Bucket | Songs | Checked | With preview | Hit rate |
|---|---|---|---|---|
| happy | 2,055 | 201 | 153 | 76% |
| sad | 1,857 | 198 | 149 | 75% |
| angry | 421 | 421 | 176 | 42% |
| fear | 2,483 | 232 | 162 | 70% |
| disgust | 1,170 | 210 | 162 | 77% |
| surprise | 1,029 | 153 | 111 | 73% |
| neutral | 1,886 | 306 | 267 | 87% |

The songs checked first were the ones nearest each Uplift waypoint, then
others in turn from every bucket. The angry bucket has been checked in full;
fewer than half its songs are on iTunes.

For the other languages, `python scripts/fetch_previews.py --language Hindi
Tamil Korean Telugu Malayalam` checked up to 30 songs per bucket of Hindi,
Tamil and Korean (nearest the Uplift waypoints first) and every Telugu and
Malayalam song, skipping repeated copies of a song. 1,168 lookups, 948 with a
preview (81%):

| Language | Checked | happy | sad | angry | fear | disgust | surprise | neutral | With preview | Hit rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Hindi | 210 | 22 | 17 | 25 | 28 | 26 | 25 | 26 | 169 | 80% |
| Tamil | 210 | 29 | 23 | 24 | 26 | 28 | 24 | 26 | 180 | 86% |
| Korean | 210 | 25 | 24 | 27 | 18 | 23 | 24 | 25 | 166 | 79% |
| Telugu | 298 | 31 | 39 | 17 | 25 | 39 | 48 | 41 | 240 | 81% |
| Malayalam | 240 | 22 | 26 | 6 | 16 | 32 | 50 | 41 | 193 | 80% |

(The bucket columns are songs with a preview.) The app's "Songs in" setting
only plays songs in the chosen language, so these are the songs it has to
work with; a queue is never padded with another language, so it can be
shorter than 30 (for example 6 Malayalam angry songs).

**The API is rate-limited** to roughly 20 requests a minute, and a burst of
about 150 requests got this machine's searches refused (HTTP 403) for hours.
One scan can need 50 or more lookups, so:

- **the app is cache-only by default**: it only plays songs already in this
  file and makes no API calls. Resonance and Uplift both pick only among
  songs with a saved preview.
- `python scripts/fetch_previews.py` fills the file slowly in the background
  (about 9 hours for all songs; `--limit N` for fewer). It can be stopped and
  rerun, and checks the songs Uplift is most likely to need first.
- setting the environment variable `LIVE_LOOKUPS=1` before starting the app
  lets it also look up songs it has not seen (in PowerShell:
  `$env:LIVE_LOOKUPS=1; streamlit run app.py`). If the API refuses, it shows
  what it has and says so.
- `python scripts/evaluate_uplift.py --previews` reports how many songs per
  bucket have a preview and whether each Uplift waypoint still has enough
  candidates.

Preview links belong to Apple and can stop working; the player skips a song
whose clip fails to load.
