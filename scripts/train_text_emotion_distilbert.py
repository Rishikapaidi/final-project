"""Fine-tune DistilBERT on GoEmotions mapped to the 7 FER2013 categories.

Same data, splits and class weighting as scripts/train_text_emotion.py so the
two models can be compared directly:
- trains on the train split only
- weighted cross-entropy with sklearn's "balanced" class weights
  (n_samples / (n_classes * class_count)), the same weights the logistic
  regression baseline uses
- picks the best epoch by validation macro-F1
- evaluates once on the test split

Reads   data/text_emotion/go_emotions_single.csv
        models/text_emotion/test_predictions.csv  (baseline, for comparison)
Writes  models/text_emotion_distilbert/  (model/, report.txt, comparison.txt,
        confusion_matrix.png, test_predictions.csv, train_log.txt)

Usage (from the project root; CPU training takes a few hours):
    python scripts/train_text_emotion_distilbert.py

If training is interrupted, run the same command again: it resumes from the
last checkpoint (models/text_emotion_distilbert/model/checkpoint.pt).
"""
import argparse
import copy
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
BASE_MODEL = "distilbert-base-uncased"
SEED = 42
CHECKPOINT_EVERY = 200  # training steps (~13 min on a laptop CPU)


def encode(tokenizer, df, max_length):
    enc = tokenizer(list(df.text), truncation=True, max_length=max_length)
    labels = [LABEL_TO_ID[label] for label in df.emotion_mapped]
    return [{"input_ids": ids, "attention_mask": mask, "labels": y}
            for ids, mask, y in zip(enc["input_ids"], enc["attention_mask"], labels)]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch.pop("labels")
        preds.append(model(**batch).logits.argmax(-1))
    return [LABELS[i] for i in torch.cat(preds).tolist()]


def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Counts")
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=LABELS).plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Share of each true class (rows sum to 1)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return cm


def comparison_table(y_true, baseline_pred, new_pred):
    rows = []
    b = precision_recall_fscore_support(y_true, baseline_pred, labels=LABELS, zero_division=0)
    d = precision_recall_fscore_support(y_true, new_pred, labels=LABELS, zero_division=0)
    for i, label in enumerate(LABELS):
        rows.append({"class": label, "support": int(b[3][i]),
                     "P_logreg": b[0][i], "P_distilbert": d[0][i],
                     "R_logreg": b[1][i], "R_distilbert": d[1][i],
                     "F1_logreg": b[2][i], "F1_distilbert": d[2][i],
                     "F1_change": d[2][i] - b[2][i]})
    table = pd.DataFrame(rows).set_index("class")
    summary = pd.DataFrame({
        "logreg": [f1_score(y_true, baseline_pred, labels=LABELS, average="macro"),
                   (np.array(y_true) == np.array(baseline_pred)).mean()],
        "distilbert": [f1_score(y_true, new_pred, labels=LABELS, average="macro"),
                       (np.array(y_true) == np.array(new_pred)).mean()],
    }, index=["macro-F1", "accuracy"])
    summary["change"] = summary.distilbert - summary.logreg
    return table, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join("data", "text_emotion", "go_emotions_single.csv"))
    parser.add_argument("--baseline", default=os.path.join("models", "text_emotion", "test_predictions.csv"))
    parser.add_argument("--out-dir", default=os.path.join("models", "text_emotion_distilbert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=None,
                        help="use only the first N train rows (for a quick smoke test)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Training saves a checkpoint every CHECKPOINT_EVERY steps and at the end of
    # each epoch; rerunning the script picks up from it after an interruption.
    ckpt_path = os.path.join(args.out_dir, "model", "checkpoint.pt")
    resume = os.path.exists(ckpt_path)
    log_file = open(os.path.join(args.out_dir, "train_log.txt"), "a" if resume else "w", encoding="utf-8")
    def log(s=""):
        print(s, flush=True)
        log_file.write(s + "\n"); log_file.flush()

    df = pd.read_csv(args.data, keep_default_na=False)
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ["train", "validation", "test"])
    if args.train_limit:
        train = train.head(args.train_limit)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    collate = DataCollatorWithPadding(tokenizer)
    train_ds, val_ds, test_ds = (encode(tokenizer, d, args.max_length) for d in (train, val, test))
    truncated = sum(len(x["input_ids"]) >= args.max_length for x in train_ds)
    log(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,} | "
        f"truncated at {args.max_length} tokens: {truncated} ({truncated / len(train_ds):.2%})")

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ds, batch_size=128, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=collate)

    weights = compute_class_weight("balanced", classes=np.array(LABELS), y=train.emotion_mapped)
    log("class weights: " + ", ".join(f"{l}={w:.2f}" for l, w in zip(LABELS, weights)))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1, best_epoch, best_state = -1.0, None, None
    start_epoch, done_in_epoch, step = 1, 0, 0
    if resume:
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng"])
        start_epoch, done_in_epoch, step = ckpt["epoch"], ckpt["done_in_epoch"], ckpt["step"]
        best_f1, best_epoch, best_state = ckpt["best_f1"], ckpt["best_epoch"], ckpt["best_state"]
        del ckpt
        log(f"== resumed from checkpoint: epoch {start_epoch}, step {done_in_epoch}/{len(train_loader)}")

    def save_checkpoint(epoch, done_in_epoch):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(),
                    "epoch": epoch, "done_in_epoch": done_in_epoch, "step": step,
                    "best_f1": best_f1, "best_epoch": best_epoch, "best_state": best_state}, tmp)
        os.replace(tmp, ckpt_path)

    start = time.time()
    steps_this_run = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        g.manual_seed(SEED + epoch)  # same batch order for an epoch whether or not it was resumed
        skip = done_in_epoch if epoch == start_epoch else 0
        running, n_running = 0.0, 0
        for i, batch in enumerate(train_loader, 1):
            if i <= skip:
                continue
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            steps_this_run += 1
            running += loss.item()
            n_running += 1
            if i % 50 == 0 or i == len(train_loader):
                elapsed = time.time() - start
                eta = elapsed / steps_this_run * (total_steps - step)
                log(f"epoch {epoch} step {i}/{len(train_loader)} | loss {running / n_running:.4f} "
                    f"| elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.0f} min")
                running, n_running = 0.0, 0
            if i % CHECKPOINT_EVERY == 0 and i < len(train_loader):
                save_checkpoint(epoch, i)

        val_pred = predict(model, val_loader)
        val_f1 = f1_score(val.emotion_mapped, val_pred, labels=LABELS, average="macro")
        log(f"== epoch {epoch}: validation macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
        save_checkpoint(epoch + 1, 0)

    log(f"== best epoch {best_epoch} (validation macro-F1 {best_f1:.4f})")
    model.load_state_dict(best_state)
    model.save_pretrained(os.path.join(args.out_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.out_dir, "model"))
    os.remove(ckpt_path)

    pred = predict(model, test_loader)

    lines = [f"DistilBERT ({BASE_MODEL}), best epoch {best_epoch} of {args.epochs}, "
             f"validation macro-F1 {best_f1:.4f}", "",
             "== TEST",
             classification_report(test.emotion_mapped, pred, labels=LABELS, digits=3, zero_division=0)]
    cm = save_confusion_matrix(
        test.emotion_mapped, pred, os.path.join(args.out_dir, "confusion_matrix.png"),
        f"Text emotion, GoEmotions test set (n={len(test):,}), DistilBERT, class-weighted, best epoch {best_epoch}")
    lines += ["== TEST confusion matrix (rows = true, columns = predicted)",
              pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string()]
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))

    out = test[["text", "emotion", "emotion_mapped"]].copy()
    out["predicted"] = pred
    out.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False, encoding="utf-8")

    baseline = pd.read_csv(args.baseline, keep_default_na=False)
    assert (baseline.text.values == test.text.values).all(), "baseline predictions are for different rows"
    table, summary = comparison_table(test.emotion_mapped, baseline.predicted, pred)
    comparison = ("Test set: logistic regression baseline vs DistilBERT (both class-weighted)\n\n"
                  + table.to_string(float_format=lambda x: f"{x:.3f}") + "\n\n"
                  + summary.to_string(float_format=lambda x: f"{x:.3f}") + "\n")
    with open(os.path.join(args.out_dir, "comparison.txt"), "w", encoding="utf-8") as f:
        f.write(comparison)
    log(comparison)


if __name__ == "__main__":
    main()
