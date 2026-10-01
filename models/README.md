# Models

## Text emotion

Both models predict one of the 7 FER2013 categories (angry, disgust, fear,
happy, sad, surprise, neutral) from text. They were trained on the same
GoEmotions data, splits and balanced class weights; see `data/README.md`.

| Folder | Model | Test macro-F1 | Role |
|---|---|---|---|
| `text_emotion_distilbert/` | Fine-tuned DistilBERT | 0.603 | **Primary** |
| `text_emotion/` | TF-IDF + logistic regression | 0.591 | Baseline, for comparison |

The trained weights aren't in git. Rebuild them with:

```
python scripts/train_text_emotion_distilbert.py   # ~3 hours on a laptop CPU, resumable
python scripts/train_text_emotion.py              # ~8 minutes
```

Each folder keeps its results in git: `report.txt` (per-class scores),
`confusion_matrix.png` and `test_predictions.csv`.
`text_emotion_distilbert/comparison.txt` compares the two models side by side.

### Decisions

- **DistilBERT is the primary text-emotion model.** It is reliably better than
  the baseline on angry, happy and surprise, and reliably worse on neutral
  (it labels fewer comments neutral). The overall macro-F1 gain (+0.012) is
  within noise.
- **All 7 categories are kept; disgust is not merged into angry.** The
  disgust/angry confusion isn't consistent between the two models: the
  baseline mostly confused disgust with neutral, DistilBERT mostly with angry.

### Known weaknesses

- `fear` and `disgust` have only ~76 test examples each, so their scores are
  approximate (about ±11 percentage points).
- `surprise` mostly learns questions and puzzlement (80% of its training data
  is curiosity, confusion and realization), not shocked reactions.
