# Evaluation facts

The facts behind MOOD TUNE's offline evaluations. Every number below comes
from a file in this repository, named next to it. All evaluations use public
data only (FER2013, GoEmotions, the song datasets). No model was trained or
tuned for them, and none was trained or tuned on anyone's face. The scripts
call the app's own code rather than copies of it:

- `fusion.py`, `uplift.py`, `mood.py` and `ui.py` are imported directly.
- The playlist functions in `player.py` (`uplift_queue`, `resonance_queue`, `playable_songs`) are imported directly.
- `face_models.fusion_config` is imported directly.
- `app.py`'s `combine_emotions`, `face_view`, `text_view` and `build_queue` run from `app.py`'s own source through `scripts/app_code.py`, because importing `app.py` would start the Streamlit page.

The primary face model is **HSEmotion** (the app's default). The CNN
(`model.h5`, inherited from the project this one started from; its training
is not documented) is reported as a secondary row.

| Evaluation | Script | Outputs |
|---|---|---|
| Fusion settings (existing) | `scripts/evaluate_fusion.py [--face hsemotion]` | `models/fusion/report.txt`, `models/fusion/hsemotion/report.txt`, `sweep_*.csv`, `fusion_sweep.png`, `confusion_matrix.png` |
| 1. Signals disagree | `scripts/evaluate_fusion_conflict.py` | `models/fusion/conflict/report.txt`, `by_pair.csv`, `by_confidence.csv`, `polarity.csv`, `conflict_heatmap.png` |
| 2. One signal missing | `scripts/evaluate_fusion_missing.py` | `models/fusion/missing/report.txt`, `summary.csv` |
| 3. Uplift playlist check | `scripts/evaluate_uplift.py --check` | `models/uplift/playlist_check.txt`, `playlist_check.csv`, `playlist_check_steps.png` (and `playlist_check_paths.csv`, not committed: one row per path, 2.9 MB) |

Run them from the project root with `.\venv\Scripts\python.exe`. Items 1 and
2 take about a minute each; item 3 takes about 12 minutes. All three read the
cached model outputs in `data/fusion/`.

---

## 1. Data

### Splits used

| Data | Split | Items | Used for |
|---|---|---|---|
| FER2013 (48×48 grayscale faces) | PublicTest | 3,589 | face validation: temperature, fusion weight |
| FER2013 | PrivateTest | 3,589 | face test: every reported number |
| GoEmotions, single-label comments | validation | 4,548 | text validation: temperature, fusion weight |
| GoEmotions, single-label comments | test | 4,590 | text test: every reported number |

**Class counts in the test splits** (from `data/fusion/*.npz`):

| | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| FER2013 PrivateTest | 491 | 55 | 528 | 879 | 594 | 416 | 626 |
| GoEmotions test (mapped) | 325 | 76 | 77 | 1,367 | 259 | 449 | 2,037 |

The FER2013 copy is checked against the official split sizes and class
counts by `scripts/prepare_fer2013.py` (see `data/README.md`). Known FER2013
quirks are left as they are: 1,853 exact duplicates and 13 near-blank
images. 288 PrivateTest images are identical to Training images.

### GoEmotions → our 7 emotions (`scripts/prepare_go_emotions.py`)

The mapping follows the GoEmotions authors' Ekman grouping, with one
exception: **approval** and **disapproval** go to neutral instead of happy
and angry.

| Ours | GoEmotions labels |
|---|---|
| angry | anger, annoyance |
| disgust | disgust |
| fear | fear, nervousness |
| happy | admiration, amusement, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| sad | disappointment, embarrassment, grief, remorse, sadness |
| surprise | confusion, curiosity, realization, surprise |
| neutral | neutral, approval, disapproval |

**Multi-label comments are dropped.** Only comments with exactly one of the
28 labels are kept: 45,446 of 54,263, so 8,817 are left out. The official
Hugging Face splits are kept. The text model (DistilBERT,
`distilbert-base-uncased`) was trained on the train split only:

- weighted cross-entropy with balanced class weights;
- 3 epochs, learning rate 3e-5, batch size 32, seed 42;
- the best epoch picked by validation macro-F1 (epoch 3, 0.602; `models/text_emotion_distilbert/report.txt`).

Only 20% of the "surprise" class is the original "surprise" label. The rest
is curiosity, confusion and realization.

### Face-model labels → our 7 emotions (`face_models.py`, `fusion.py`)

Every face model's output is put into one order, `fusion.FACE_MODEL_LABELS`
(Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised).
`fusion.align_face` then reorders it by name to the canonical
`fusion.EMOTIONS` order (angry, disgust, fear, happy, sad, surprise,
neutral).

- **HSEmotion** has 8 classes (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise). **Contempt is added to disgust**: its probability is summed into Disgusted.
- **FER+** has the same 8 classes plus contempt, handled the same way (not used in these evaluations).
- **The CNN** already outputs our 7.

The text model outputs the 7 in `EMOTIONS` order (checked against its
`id2label`).

### How faces are fed

In the fusion evaluations each face is the whole 48×48 FER2013 image, with
no webcam and no face detector:

- **HSEmotion:** the image is converted to 3-channel RGB, resized to 224×224 and normalised with ImageNet values, through `HSEmotionModel.prepare`, as in the app.
- **CNN:** the image is scaled to 0–1. The cached CNN outputs (`data/fusion/face_*.npz`) are of one crop. The app also averages each crop with its mirror image, so CNN rows here are slightly different from the app's CNN.

### Synthetic pairs (`scripts/evaluate_fusion.py`)

No dataset has a face and a text from the same moment, so pairs are built.

**Same-emotion pairs** (used to choose the settings, and for item 2):

- one FER2013 face and one GoEmotions comment with the **same** label;
- 50 pairs per emotion × 7 emotions = 350 pairs per pairing, so every pairing is class-balanced;
- 20 pairings;
- within one pairing, faces and comments are drawn without replacement. Across pairings they repeat. FER2013 PrivateTest has only 55 disgust faces, so disgust faces repeat heavily.

The seed is 42, from one generator, drawn in this order:

1. 20 validation pairings (PublicTest × GoEmotions validation);
2. 20 test pairings (PrivateTest × GoEmotions test).

`eval_common.test_pairs` rebuilds exactly these pairs, and item 2
cross-checks them against `sweep_test.csv`; they match to 4 decimals.

**Conflict pairs** (item 1, `scripts/evaluate_fusion_conflict.py`):

- face label A ≠ text label B, covering all 42 ordered pairs;
- 50 pairs per ordered pair per draw, without replacement within a draw;
- 20 draws, so 1,000 pairs per ordered pair and 42,000 per face model;
- sources are PrivateTest × GoEmotions test;
- seed 2026, from one generator, in the order draw → A → B (both in `EMOTIONS` order).

**Independence assumption:** a pair treats the face and the comment as
independent evidence, given the emotion. They come from different people and
different sources, so this is true by construction here. For a real person,
face and words are not independent. The synthetic pairs say nothing about how
correlated they are in real use.

---

## 2. How the fusion settings were chosen

The settings are in `models/fusion/fusion_config.json` (CNN) and
`models/fusion/hsemotion/fusion_config.json` (HSEmotion). The app reads
them through `face_models.fusion_config`.

1. **Temperatures.** Each model gets one temperature T, fitted on **its own validation split only** by minimising negative log-likelihood. The search is bounded on log T ∈ [−3, 3]. Calibration is `softmax(log p / T)`.

   | | T_face | T_text | Face ECE on test (before → after) | Text ECE on test (before → after) |
   |---|---|---|---|---|
   | HSEmotion | 1.448 | 1.306 | 0.093 → 0.027 | 0.119 → 0.043 |
   | CNN | 1.931 | 1.306 | 0.136 → 0.027 | 0.119 → 0.043 |

2. **Rules compared:**
   - `weighted_average`: p = w·p_face + (1−w)·p_text;
   - `log_linear`: log p ∝ w·log p_face + (1−w)·log p_text, renormalised.

3. **Sweep.** Face weight w runs from 0 to 1 in steps of 0.05 (21 values) for both rules, on the calibrated **validation** pairs. The best rule and weight by mean macro-F1 over the 20 pairings are chosen. The test pairs are scored once, after the choice.

4. **Result.** For both face models the chosen settings are `log_linear` with w = 0.55. The best weight was 0.55 for both rules with both models.

   | Macro-F1 on validation pairs | HSEmotion | CNN |
   |---|---|---|
   | `log_linear` | 0.799 ± 0.016 | 0.805 ± 0.019 |
   | `weighted_average` | 0.767 | 0.789 |

5. **Keeping HSEmotion's own settings.** HSEmotion's settings were kept only because they beat the CNN's settings on HSEmotion's validation pairs: 0.799 against 0.789. On test pairs the two are 0.800 and 0.798 (`models/fusion/hsemotion/report.txt`).

**One signal missing.** The app uses the signal it has alone: face weight 1
with a uniform text vector, or face weight 0. The rule then makes no
difference.

**What the app shows from a reading:**

- **Playlist mix:** the emotions at ≥ 15% of the fused probability, at most 5.
- **Headline:** "Mostly X" if the top emotion has ≥ 50%, otherwise "Mixed: X and Y" (`mood.headline`).
- **"X% low":** the share of sad + angry + fear + disgust, banded as Light below 25%, In between from 25% to under 50%, and Low from 50% (`ui.band`).
- **Auto mode:** picks Uplift when that share is ≥ 50% (`uplift.should_uplift`).

---

## 3. Results

### Item 2: one signal missing, and face vs text vs fused (`models/fusion/missing/report.txt`)

The app's single-source path matched the single model's calibrated output on
every test item: HSEmotion 3.3e-16, CNN 2.2e-16, text 2.2e-16. Its top
emotion is identical to the uncalibrated model's. On the full test splits it
gets accuracy:

- face only: HSEmotion 55.7%, CNN 63.2%;
- text only: 67.4%.

**The same test pairs** (20 pairings × 350; mean ± std over pairings):

| Face model | Variant | Accuracy | Macro-F1 | Fused beats the better single model (text) |
|---|---|---|---|---|
| HSEmotion | Face only (no text) | 0.528 ± 0.020 | 0.531 ± 0.021 | |
| HSEmotion | Text only (no face) | 0.688 ± 0.023 | 0.689 ± 0.022 | |
| HSEmotion | Fused, log_linear, w = 0.55 (app) | **0.800 ± 0.015** | **0.800 ± 0.015** | 20 of 20 pairings |
| HSEmotion | Fused, weighted_average, w = 0.55 | 0.764 ± 0.014 | 0.763 ± 0.015 | 20 of 20 |
| CNN | Face only (no text) | 0.614 ± 0.021 | 0.615 ± 0.021 | |
| CNN | Text only (no face) | 0.688 ± 0.023 | 0.689 ± 0.022 | |
| CNN | Fused, log_linear, w = 0.55 | 0.815 ± 0.018 | 0.816 ± 0.018 | 20 of 20 |
| CNN | Fused, weighted_average, w = 0.55 | 0.800 ± 0.018 | 0.799 ± 0.018 | 20 of 20 |

These are same-emotion pairs, so they cover agreement cases only.

### Item 1: the signals disagree (`models/fusion/conflict/report.txt`)

There is no right answer for these pairs, so the results describe behaviour
only. Each model has 42,000 pairs.

| | HSEmotion | CNN |
|---|---|---|
| Fused top = face's label A | 29.5% | 37.5% |
| Fused top = text's label B | 44.2% | 36.5% |
| Fused top = neither | 26.3% | 26.0% |
| Mean fused p(A) / p(B) | 0.249 / 0.297 | 0.300 / 0.268 |
| Headline "Mostly X" / "Mixed: X and Y" | 34.2% / 65.8% | 43.4% / 56.6% |
| Mixed headlines that name exactly A and B | 28.7% | 28.4% |
| Pairs where the two models' own top emotions differ | 89.8% | 90.5% |
| ...of those, fused top = face model's / text model's emotion | 34.5% / 51.1% | 45.7% / 42.3% |

Across the 20 draws, the standard deviation of each share is at most 1.0
point.

- **With HSEmotion the text wins more often than the face.** HSEmotion's calibrated outputs are less confident than the text model's: its top probability is ≥ 0.70 in 21.5% of faces, against 62.7% of comments.
- **With the CNN the two are about even.**
- **Which face labels survive a conflict varies.** For HSEmotion, a happy face survives most often (59% of pairs average over B), and fear the least (10%). Disgust faces are mostly read as angry when the text is angry: B wins 74% (`conflict_heatmap.png`, `by_pair.csv`).

**By confidence (HSEmotion).** These are pairs where the two models' top
emotions differ. Confidence is the calibrated top probability, binned below
0.40, 0.40–0.70, and from 0.70. A face winner means the fused top is the face
model's own emotion; a text winner means it is the text model's
(`by_confidence.csv`).

| Face \ Text confidence | Text low | Text medium | Text high |
|---|---|---|---|
| Face low | face 35% / text 28% | 20% / 55% | 6% / 87% |
| Face medium | 53% / 16% | 40% / 31% | 24% / 64% |
| Face high | 81% / 6% | 78% / 11% | 67% / 30% |

The more confident signal wins. When both are confident, the face wins, at
its 0.55 weight. The CNN shows the same pattern: face high with text low
gives 92% / 3%, and face low with text high gives 13% / 72%.

**Polarity conflicts** (8,000 pairs each way; `polarity.csv`):

| Case | Model | Fused band Light / In between / Low | Face alone | Text alone | Fused top = A / B / neither |
|---|---|---|---|---|---|
| Face happy or neutral, text sad, angry, fear or disgust | HSEmotion | 17.2% / 24.4% / 58.3% | 38.9 / 34.9 / 26.2 | 8.5 / 7.1 / 84.4 | 39.4% / 43.5% / 17.0% |
| Face happy or neutral, text sad, angry, fear or disgust | CNN | 24.1% / 26.5% / 49.4% | 53.4 / 21.7 / 24.9 | 8.5 / 7.1 / 84.4 | 50.1% / 33.0% / 16.9% |
| Face sad, angry, fear or disgust, text happy or neutral | HSEmotion | 12.1% / 43.4% / 44.5% | 1.2 / 12.4 / 86.4 | 80.0 / 8.3 / 11.7 | 20.0% / 41.6% / 38.4% |
| Face sad, angry, fear or disgust, text happy or neutral | CNN | 13.0% / 40.5% / 46.6% | 3.7 / 10.7 / 85.6 | 80.0 / 8.3 / 11.7 | 22.7% / 42.1% / 35.2% |

**Low also means Auto mode picks Uplift.** With HSEmotion:

- **A happy face against heavy words:** the face mostly holds. The band is Low in 25–54% of pairs, depending on the text emotion.
- **A neutral face against heavy words:** the band is mostly Low, 60–89%.
- **A heavy face against light words:** the result lands in between. The fused top is often a third emotion (38%), and the band is In between or Low in 88% of pairs.

### Item 3: Uplift playlist check (`models/uplift/playlist_check.txt`)

**Readings.** The app's fused readings (HSEmotion's settings) of the 6,865
distinct (face, comment) test pairs were computed. Of these, 3,825 have heavy
emotions ≥ 50%, so Uplift applies. Each was grouped by the start emotion
Uplift picks, giving sad 952, angry 1,219, fear 933 and disgust 721. 200 per
start emotion were sampled with seed 42.

**Paths.** For every language, each reading's path is built by the app's
`build_queue` in Uplift mode, using cache-only previews. That is 800 paths
per language per method. There are two baselines, built with the app's own
functions:

- 5 random playable songs of the same language;
- 5 songs of the start emotion's bucket (a Resonance playlist).

**Intended progression** comes from the planned waypoints:

- **Valence** rises at every step for all four start emotions.
- **Arousal** falls at every step for angry, fear and disgust, ending at neutral (calm).
- **Arousal for sad** falls slightly, then rises, because the journey runs sad → neutral → happy and happy songs are more energetic. For sad, the last song is meant to be brighter and more energetic, not calmer.

**Across English, Hindi, Tamil and Korean, by start emotion** (Uplift path vs
the random baseline):

| Start | Song 1 median distance from start (sd) | Song 1 in the reading's mix | Valence steps right | All 4 valence steps right | Arousal steps right | ΔValence, song 5 − song 1 | ΔArousal | Brighter / calmer |
|---|---|---|---|---|---|---|---|---|
| sad | 0.09 (random 0.99) | 93% (27%) | 99% (50%) | 98% (1%) | 95% (48%) | +3.54 ± 0.53 (+0.02 ± 1.76) | +1.57 ± 0.29 (planned up) | 100% / — |
| angry | 0.14 (1.04) | 93% (30%) | 89% (49%) | 62% (1%) | 95% (49%) | +1.44 ± 0.51 (−0.12 ± 1.82) | −1.51 ± 0.44 | 98% / 100% |
| fear | 0.08 (1.05) | 98% (21%) | 96% (49%) | 86% (1%) | 96% (50%) | +2.01 ± 0.42 (−0.05 ± 1.85) | −1.44 ± 0.26 | 100% / 100% |
| disgust | 0.06 (1.06) | 96% (17%) | 94% (51%) | 79% (2%) | 77% (50%) | +2.53 ± 0.45 (+0.12 ± 1.81) | −0.56 ± 0.27 | 100% / 100% |

- **Structure.** Every path in every language had 5 songs, and all were distinct by link and by title + artist (`playlist_check.csv`).
- **Resonance baseline.** Song 1 is always in the start bucket. It gets about 45% of valence steps right and almost never all 4 (0–1%). Its last song is brighter in 41–46% of paths.
- **Per language** (all four start emotions, Uplift):

  | Language | Valence steps right | All 4 right | Song 5 brighter |
  |---|---|---|---|
  | English | 97% | 90% | 100% |
  | Hindi | 95% | 80% | 100% |
  | Tamil | 93% | 74% | 100% |
  | Korean | 94% | 80% | 99% |

**Weak spots, as measured:**

- **Telugu and Malayalam are not offered Uplift by the app**, which falls back to Resonance. Built anyway, their paths are visibly worse:
  - Malayalam angry: valence steps right 63%, all 4 right 10%, song 1 in the start bucket 25%; Malayalam has only 6 playable angry songs.
  - Telugu angry: 72% / 16% / 29%.
- **Angry in Hindi and Korean.** Song 1 is in the angry bucket in only 42% and 46% of paths, but in one of the reading's own emotions in 85% and 88%. All 4 valence steps are right in 71% (Hindi) and 58% (Korean), against 68% in English. Angry has 24–27 playable songs per non-English language.
- **Disgust arousal.** The planned arousal step is only about −0.12 per step, so "all 4 arousal steps right" is low: 16% in Hindi, 20% in Tamil, 29% in Korean and 62% in English. The overall change is still downwards in 100% of paths.
- **The negative buckets overlap in valence and arousal**, so "close to the start mood" is measured in those numbers and by bucket. Neither says how a song sounds to a person.
- **The valence and arousal values are dataset-derived:**
  - English: Muse listener mood tags;
  - other languages: Spotify valence and energy, mapped onto the Muse scale by percentile. On 45 songs in both datasets the agreement is r = 0.50 for valence and 0.28 for arousal (`data/README.md`).
  - They were not validated by people. This checks playlist structure, not any effect on a listener.

---

## 4. Data sources and licences

| Source | Used for | Licence |
|---|---|---|
| FER2013 (Kaggle / ICML 2013 challenge; Hugging Face copy `Aaryan333/fer2013_train_publicTest_privateTest`) | face test and validation data (the inherited CNN's training data is not documented) | **No stated licence**; research challenge data |
| FER+ labels (Microsoft) | `scripts/evaluate_face.py` only | MIT |
| GoEmotions (`google-research-datasets/go_emotions`, Hugging Face) | text model training and evaluation | Dataset card: the GitHub repository holding it is Apache 2.0. The comments are Reddit content. |
| DistilBERT `distilbert-base-uncased` | text model base | Apache 2.0 |
| HSEmotion `enet_b0_8_best_afew` (A. Savchenko) | default face model | code Apache-2.0; trained on AffectNet, whose licence allows **non-commercial research and education only** |
| Muse (`muse_v3.csv`; Akiki and Burghardt 2021, *Journal of Open Humanities Data*, DOI 10.5334/johd.33) | English songs, valence and arousal | CC BY 4.0 |
| "Spotify Tracks Dataset (Updated Every Week)", gauthamvijayaraj, Kaggle | songs in other languages | Apache 2.0 |
| iTunes Search API previews and artwork | playback only | Apple's terms; not redistributed |

---

## 5. What this does NOT show

- **How accurate the app is on webcam faces.** All faces are 48×48 grayscale FER2013 images, with no detector, lighting or real camera. Webcam accuracy was not measured.
- **How the app behaves when one real person's face and words agree or disagree.** Pairs join unrelated faces and comments and assume independence. Real face and text are correlated, and real conflicts (sarcasm, masking) are not represented.
- **Which reading is right when signals conflict.** Item 1 describes behaviour only.
- **Performance on the people who will use the app.** FER2013, AffectNet and Reddit have their own demographic and cultural skews, and no subgroup analysis was done.
- **That the settings are optimal elsewhere.** The temperatures and weight were fitted on synthetic same-emotion pairs (validation split only) and may not transfer.
- **That Uplift improves anyone's mood.** Item 3 checks the order and spread of the songs' dataset values, not any effect on a listener. No user study was done.
- **That a song "sounds" sad, angry, fearful or disgusted.** Buckets come from Muse tags or from nearest valence and arousal centres, which overlap for the negative emotions.
- **Exact CNN behaviour.** CNN rows use one crop, not the app's mirror averaging.
