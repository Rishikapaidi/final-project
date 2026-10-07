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

The song list the app recommends from. It is committed to git (1.6 MB), so
there is nothing to run after cloning. To rebuild it from `muse_v3.csv` (which
is kept untouched in the project root):

```
python scripts/prepare_songs.py
```

Columns: `title`, `artist`, `valence_tags` (positivity, about 0.2–8.5),
`arousal_tags` (energy, about 0.1–7.3), `seeds` (mood words), `bucket` and
`lastfm_url`.

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

## Song previews: `data/songs/previews.csv`

The song file only has Last.fm links, which cannot be played. The app looks
each song up in the free iTunes Search API (no key) to get a 30-second preview
clip and album art, and stores the answer here, including "no preview found",
so each song is only ever looked up once. The file is not committed.

Columns: `lastfm_url`, `preview_url`, `artwork_url`, `itunes_title`,
`itunes_artist` (the last four are empty when no match was found).

A result is accepted only if both the artist and the title loosely match
(ignoring case, accents and punctuation). In a sample of 48 songs, 73% had a
usable preview.

**The API is rate-limited** to roughly 20 requests a minute, and a burst of
about 150 requests got this machine refused (HTTP 403) for well over half an
hour. One scan can need 50 or more lookups, so:

- the app uses songs already in this file first and only looks up new ones
  when it has to; if the API refuses, it shows what it has and says so
- `python scripts/fetch_previews.py` fills the file slowly in the background
  (about 9 hours for all songs; `--limit N` for fewer). It can be stopped and
  rerun, and checks the songs Uplift is most likely to need first.

Preview links belong to Apple and can stop working; the player skips a song
whose clip fails to load.
