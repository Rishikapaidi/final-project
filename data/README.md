# Data

The dataset files in this folder are **not committed to git** (see `.gitignore`).
They are generated from public sources by the scripts in `scripts/`.

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
