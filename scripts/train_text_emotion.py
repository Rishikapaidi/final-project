"""Train a text-emotion classifier on GoEmotions mapped to the 7 FER2013 categories.

Model: TF-IDF (word 1-2 grams + character 2-5 grams) -> logistic regression
with class_weight="balanced", so rare classes (fear, disgust) count as much as
common ones. The regularisation strength C is chosen on the validation split
by macro-F1; the test split is only used once, for the final report.

Reads   data/text_emotion/go_emotions_single.csv
Writes  models/text_emotion/  (model.joblib, report.txt, confusion_matrix.png,
                               test_predictions.csv)

Usage (from the project root):
    python scripts/train_text_emotion.py
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from sklearn.pipeline import make_pipeline, make_union

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
C_GRID = [0.25, 0.5, 1, 2, 4, 8]
SEED = 42


def build_model(C, class_weight):
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                        sublinear_tf=True, max_features=200_000),
    )
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000,
                             random_state=SEED)
    return make_pipeline(features, clf)


def fit_best(train, val, class_weight):
    """Try each C on validation and return (best_C, fitted model, scores)."""
    scores = {}
    best = None
    for C in C_GRID:
        model = build_model(C, class_weight).fit(train.text, train.emotion_mapped)
        scores[C] = f1_score(val.emotion_mapped, model.predict(val.text),
                             labels=LABELS, average="macro")
        if best is None or scores[C] > scores[best[0]]:
            best = (C, model)
    return best[0], best[1], scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join("data", "text_emotion", "go_emotions_single.csv"))
    parser.add_argument("--out-dir", default=os.path.join("models", "text_emotion"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data, keep_default_na=False)
    train, val, test = (df[df.split == s] for s in ["train", "validation", "test"])

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    results = {}
    for name, cw in [("unweighted", None), ("balanced", "balanced")]:
        C, model, scores = fit_best(train, val, cw)
        pred = model.predict(test.text)
        results[name] = (C, model, pred)
        log(f"== {name}: validation macro-F1 by C: "
            + ", ".join(f"C={c}: {s:.3f}" for c, s in scores.items()) + f"  -> best C={C}")

    for name, (C, model, pred) in results.items():
        log(f"\n== TEST, {name} (C={C})")
        log(classification_report(test.emotion_mapped, pred, labels=LABELS, digits=3, zero_division=0))

    C, model, pred = results["balanced"]
    joblib.dump(model, os.path.join(args.out_dir, "model.joblib"))

    cm = confusion_matrix(test.emotion_mapped, pred, labels=LABELS)
    log("== TEST confusion matrix, balanced (rows = true, columns = predicted)")
    log(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Counts")
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=LABELS).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Share of each true class (rows sum to 1)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.suptitle(f"Text emotion, GoEmotions test set (n={len(test):,}), balanced logistic regression, C={C}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)

    out = test[["text", "emotion", "emotion_mapped"]].copy()
    out["predicted"] = pred
    out["unweighted_predicted"] = results["unweighted"][2]
    out.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False, encoding="utf-8")

    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
