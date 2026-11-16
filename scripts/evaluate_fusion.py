"""Evaluate face + text emotion fusion without paired data.

There is no dataset with a face and a text for the same moment, so this builds
synthetic pairs: a FER2013 face and a GoEmotions comment that share the same
true emotion. That assumes face and text are independent evidence once the true
emotion is known, and only covers cases where the two agree.

Steps:
1. Face model (model.h5) probabilities on FER2013 PublicTest and PrivateTest,
   text model (DistilBERT) probabilities on GoEmotions validation and test.
   Cached in data/fusion/.
2. Each model is evaluated on its own test set (supporting evidence).
3. Each model gets a temperature, fitted on its own validation set (NLL).
4. Class-balanced synthetic pairs (PAIRS_PER_CLASS per emotion, REPEATS random
   pairings) are built from the validation sets; face_weight is swept from 0
   to 1 for both fusion rules, and the best rule + weight is chosen by mean
   macro-F1.
5. The chosen settings are reported once on pairs built from the test sets.

The face model is the CNN (model.h5) unless --face names another one of
face_models.py (hsemotion or ferplus); faces are always the whole 48x48
FER2013 images. For another model:
- its probabilities are cached as data/fusion/face_val_<model>.npz /
  face_test_<model>.npz, and its outputs go to models/fusion/<model>/;
- the CNN's settings (models/fusion/fusion_config.json) are also scored on
  this model's validation and test pairs, and the new fit is saved as
  models/fusion/<model>/fusion_config.json ONLY if it beats them on the
  validation pairs (otherwise as fusion_config_candidate.json, unused); the
  app uses a model's own settings when that file is there
  (face_models.fusion_config).

Usage (from the project root):
    python scripts/evaluate_fusion.py                    the CNN
    python scripts/evaluate_fusion.py --face hsemotion   HSEmotion

Reads   data/face_emotion/raw/*.parquet  (FER2013, see data/README.md)
        data/text_emotion/go_emotions_single.csv
        model.h5, models/text_emotion_distilbert/model/
Writes  models/fusion/  (report.txt, fusion_config.json, sweep_validation.csv,
        sweep_test.csv, fusion_sweep.png, confusion_matrix.png)
"""
import argparse
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, log_loss)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion import (EMOTIONS, METHODS, TEXT_MODEL_LABELS, align_face, align_text,  # noqa: E402
                    calibrate, fuse_probs)

FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # dataset label ids
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
PAIRS_PER_CLASS = 50
REPEATS = 20
SEED = 42
# FER2013: human agreement is ~65% and the best published models reach ~73-76%.
# A simple CNN scoring above this is more likely to have seen the test images.
SUSPICIOUS_FACE_ACCURACY = 0.75


def face_probabilities(split, face_model_path):
    from PIL import Image
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    # Same architecture as app.py.
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"), MaxPooling2D((2, 2)),
        Dropout(0.25), Flatten(), Dense(1024, activation="relu"), Dropout(0.5),
        Dense(7, activation="softmax"),
    ])
    model.load_weights(face_model_path)
    df = pd.read_parquet(os.path.join("data", "face_emotion", "raw", f"{split}-0000.parquet"))
    images = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in df.image])
    # The model was trained on pixels scaled to 0-1; raw 0-255 input makes it
    # ~99.7% confident on nearly everything.
    x = images[..., None].astype("float32") / 255.0
    probs = model.predict(x, batch_size=256, verbose=0)
    labels = np.array([FER2013_LABELS[i] for i in df.label])
    return align_face(probs), labels


def text_probabilities(split, text_model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = pd.read_csv(os.path.join("data", "text_emotion", "go_emotions_single.csv"), keep_default_na=False)
    df = df[df.split == split]
    tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(text_model_dir).eval()
    assert [model.config.id2label[i] for i in range(7)] == TEXT_MODEL_LABELS
    out = []
    with torch.no_grad():
        texts = list(df.text)
        for i in range(0, len(texts), 128):
            enc = tokenizer(texts[i:i + 128], truncation=True, max_length=64, padding=True, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, -1).numpy())
    return align_text(np.concatenate(out)), df.emotion_mapped.values


def model_face_probabilities(split, key):
    """A face_models.py model on the whole 48x48 images (the image is the face box), in chunks."""
    from PIL import Image
    import face_models
    model = face_models.load(key)
    assert model.key == key, f"{key} could not be loaded: {model.fallback}"
    df = pd.read_parquet(os.path.join("data", "face_emotion", "raw", f"{split}-0000.parquet"))
    images = [np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in df.image]
    probs = np.concatenate([model.predict([model.prepare(img, img, (0, 0, img.shape[1], img.shape[0])) for img in images[i:i + 256]])
                            for i in range(0, len(images), 256)])
    labels = np.array([FER2013_LABELS[i] for i in df.label])
    return align_face(probs), labels


def load_probabilities(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    suffix = "" if args.face == "cnn" else f"_{args.face}"
    face = (lambda split: face_probabilities(split, args.face_model)) if args.face == "cnn" else \
           (lambda split: model_face_probabilities(split, args.face))
    jobs = {
        f"face_val{suffix}": lambda: face("publicTest"),
        f"face_test{suffix}": lambda: face("privateTest"),
        "text_val": lambda: text_probabilities("validation", args.text_model),
        "text_test": lambda: text_probabilities("test", args.text_model),
    }
    data = {}
    for name, job in jobs.items():
        path = os.path.join(args.cache_dir, f"{name}.npz")
        if not os.path.exists(path):
            probs, labels = job()
            np.savez(path, probs=probs, labels=np.asarray(labels, dtype=str))
        z = np.load(path, allow_pickle=False)
        data[name] = (z["probs"], z["labels"].astype(str))
    return data


def fit_temperature(probs, labels):
    y = np.array([EMOTIONS.index(l) for l in labels])
    nll = lambda log_t: log_loss(y, calibrate(probs, np.exp(log_t)), labels=range(7))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(probs, labels, bins=15):
    """Expected calibration error: gap between confidence and accuracy."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == np.array([EMOTIONS.index(l) for l in labels])
    edges = np.linspace(0, 1, bins + 1)
    return sum(abs(correct[m].mean() - conf[m].mean()) * m.mean()
               for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())


def make_pairings(face_labels, text_labels, rng):
    """REPEATS class-balanced pairings: lists of (face_idx, text_idx, label_idx) arrays."""
    pairings = []
    for _ in range(REPEATS):
        f_idx, t_idx, y = [], [], []
        for k, emotion in enumerate(EMOTIONS):
            f_pool = np.flatnonzero(face_labels == emotion)
            t_pool = np.flatnonzero(text_labels == emotion)
            f_idx.append(rng.choice(f_pool, PAIRS_PER_CLASS, replace=False))
            t_idx.append(rng.choice(t_pool, PAIRS_PER_CLASS, replace=False))
            y.append(np.full(PAIRS_PER_CLASS, k))
        pairings.append((np.concatenate(f_idx), np.concatenate(t_idx), np.concatenate(y)))
    return pairings


def sweep(face_probs, text_probs, pairings):
    rows = []
    for method in METHODS:
        for w in WEIGHTS:
            scores = [f1_score(y, fuse_probs(face_probs[f], text_probs[t], w, method).argmax(1),
                               labels=range(7), average="macro") for f, t, y in pairings]
            rows.append({"method": method, "face_weight": w,
                         "macro_f1_mean": np.mean(scores), "macro_f1_std": np.std(scores),
                         "scores": scores})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--face-model", default="model.h5")
    parser.add_argument("--face", default="cnn", choices=["cnn", "hsemotion", "ferplus"])
    parser.add_argument("--text-model", default=os.path.join("models", "text_emotion_distilbert", "model"))
    parser.add_argument("--cache-dir", default=os.path.join("data", "fusion"))
    parser.add_argument("--out-dir", default=os.path.join("models", "fusion"))
    args = parser.parse_args()
    cnn_config_path = os.path.join(args.out_dir, "fusion_config.json")
    if args.face != "cnn":
        args.out_dir = os.path.join(args.out_dir, args.face)
    os.makedirs(args.out_dir, exist_ok=True)

    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    data = load_probabilities(args)
    suffix = "" if args.face == "cnn" else f"_{args.face}"
    (fv, fv_y), (ft, ft_y) = data[f"face_val{suffix}"], data[f"face_test{suffix}"]
    log(f"Face model: {args.face} (whole 48x48 FER2013 images) | text model: DistilBERT\n")
    (tv, tv_y), (tt, tt_y) = data["text_val"], data["text_test"]

    # --- (a) each model on its own test set
    log("== (a) EACH MODEL ON ITS OWN TEST SET")
    face_acc = float(np.mean(np.array(EMOTIONS)[ft.argmax(1)] == ft_y))
    for name, probs, y in [("Face model, FER2013 PrivateTest", ft, ft_y),
                           ("Text model, GoEmotions test", tt, tt_y)]:
        pred = np.array(EMOTIONS)[probs.argmax(1)]
        log(f"-- {name} (n={len(y):,})")
        log(classification_report(y, pred, labels=EMOTIONS, digits=3, zero_division=0))
    if face_acc > SUSPICIOUS_FACE_ACCURACY:
        log(f"!! WARNING: face accuracy {face_acc:.1%} is above {SUSPICIOUS_FACE_ACCURACY:.0%}, which is "
            "suspiciously high for this CNN on FER2013. The model's training history is unknown and it "
            "may have seen PrivateTest images; treat face results and face-heavy weights with caution.")
    else:
        log(f"Face accuracy {face_acc:.1%} is within the normal range for FER2013 "
            f"(warning threshold {SUSPICIOUS_FACE_ACCURACY:.0%}): no sign of test-set leakage from this number alone.")
    log("Note: 288 PrivateTest images are pixel-identical to FER2013 Training images (a known FER2013 issue).")

    # --- calibration
    t_face, t_text = fit_temperature(fv, fv_y), fit_temperature(tv, tv_y)
    log("\n== CALIBRATION (temperatures fitted on each model's own validation set)")
    for name, t, pv, yv, pt, yt in [("face", t_face, fv, fv_y, ft, ft_y), ("text", t_text, tv, tv_y, tt, tt_y)]:
        log(f"{name}: T = {t:.3f} | test ECE {ece(pt, yt):.3f} -> {ece(calibrate(pt, t), yt):.3f} | "
            f"test mean confidence {pt.max(1).mean():.3f} -> {calibrate(pt, t).max(1).mean():.3f} "
            f"(accuracy {np.mean(np.array(EMOTIONS)[pt.argmax(1)] == yt):.3f})")
    fv_c, ft_c, tv_c, tt_c = calibrate(fv, t_face), calibrate(ft, t_face), calibrate(tv, t_text), calibrate(tt, t_text)

    # --- selection on validation pairs
    rng = np.random.default_rng(SEED)
    val_pairs = make_pairings(fv_y, tv_y, rng)
    test_pairs = make_pairings(ft_y, tt_y, rng)
    val_sweep = sweep(fv_c, tv_c, val_pairs)
    test_sweep = sweep(ft_c, tt_c, test_pairs)
    uncal_val = sweep(fv, tv, val_pairs)

    def at(df, method, w):
        return df[(df.method == method) & np.isclose(df.face_weight, w)].iloc[0]

    best = {m: val_sweep[val_sweep.method == m].sort_values("macro_f1_mean", ascending=False).iloc[0] for m in METHODS}
    chosen = max(best.values(), key=lambda r: r.macro_f1_mean)
    log(f"\n== SELECTION ON VALIDATION PAIRS ({PAIRS_PER_CLASS} pairs x 7 emotions, {REPEATS} random pairings)")
    log(f"face only (w=1.00): macro-F1 {at(val_sweep, 'weighted_average', 1).macro_f1_mean:.3f}")
    log(f"text only (w=0.00): macro-F1 {at(val_sweep, 'weighted_average', 0).macro_f1_mean:.3f}")
    for m in METHODS:
        b = best[m]
        u = uncal_val[uncal_val.method == m].sort_values("macro_f1_mean", ascending=False).iloc[0]
        log(f"{m:17} best face_weight {b.face_weight:.2f}: macro-F1 {b.macro_f1_mean:.3f} +/- {b.macro_f1_std:.3f}"
            f"   (without calibration: best w {u.face_weight:.2f}, {u.macro_f1_mean:.3f})")
    log(f"CHOSEN: {chosen.method}, face_weight {chosen.face_weight:.2f}")

    # --- final report on test pairs
    log(f"\n== FINAL: TEST PAIRS (FER2013 PrivateTest x GoEmotions test; settings chosen on validation)")
    rows = [("Face only", "weighted_average", 1.0), ("Text only", "weighted_average", 0.0)]
    rows += [(f"{m}, w={best[m].face_weight:.2f}", m, best[m].face_weight) for m in METHODS]
    face_scores = np.array(at(test_sweep, "weighted_average", 1).scores)
    text_scores = np.array(at(test_sweep, "weighted_average", 0).scores)
    better_single = face_scores if face_scores.mean() > text_scores.mean() else text_scores
    log(f"{'':28} macro-F1 mean +/- std   min   max   beats better single model in")
    for label, m, w in rows:
        s = np.array(at(test_sweep, m, w).scores)
        beats = "" if label.endswith("only") else f"{np.mean(s > better_single):.0%} of pairings"
        log(f"{label:28} {s.mean():.3f} +/- {s.std():.3f}   {s.min():.3f} {s.max():.3f}   {beats}")

    # per-class F1 on test pairs (all pairings pooled) for singles and chosen rule
    def pooled(method, w):
        y = np.concatenate([p[2] for p in test_pairs])
        pred = np.concatenate([fuse_probs(ft_c[f], tt_c[t], w, method).argmax(1) for f, t, _ in test_pairs])
        return y, pred
    log("\nPer-class F1 on test pairs (all pairings pooled):")
    per = {}
    for label, m, w in rows:
        y, pred = pooled(m, w)
        per[label] = f1_score(y, pred, labels=range(7), average=None)
    log(pd.DataFrame(per, index=EMOTIONS).to_string(float_format=lambda x: f"{x:.3f}"))

    y, pred = pooled(chosen.method, chosen.face_weight)
    cm = confusion_matrix(y, pred, labels=range(7))
    log(f"\nConfusion matrix, chosen rule on test pairs (rows = true, columns = predicted; "
        f"{REPEATS} pairings pooled, {PAIRS_PER_CLASS * REPEATS} pairs per emotion):")
    log(pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_string())

    # --- another face model: the CNN's settings on the same pairs, and keep the new fit only if it beats them
    config_name = "fusion_config.json"
    if args.face != "cnn":
        old = json.load(open(cnn_config_path))
        def score(cfg, f_probs, t_probs, pairs):
            fc, tc = calibrate(f_probs, cfg["face_temperature"]), calibrate(t_probs, cfg["text_temperature"])
            s = [f1_score(y, fuse_probs(fc[f], tc[t], cfg["face_weight"], cfg["method"]).argmax(1), labels=range(7), average="macro")
                 for f, t, y in pairs]
            return np.mean(s), np.std(s)
        new = {"method": chosen.method, "face_weight": float(chosen.face_weight), "face_temperature": t_face, "text_temperature": t_text}
        single = lambda cfg, w, pairs, fp, tp: score({**cfg, "face_weight": w, "method": "weighted_average"}, fp, tp, pairs)
        log(f"\n== THE CNN'S SETTINGS AGAINST THE NEW FIT, ON THIS MODEL'S PAIRS")
        log(f"CNN's settings: {old['method']}, face_weight {old['face_weight']:.2f}, T_face {old['face_temperature']:.3f}, T_text {old['text_temperature']:.3f}")
        log(f"new fit:        {new['method']}, face_weight {new['face_weight']:.2f}, T_face {t_face:.3f}, T_text {t_text:.3f}")
        log(f"{'':16} {'validation pairs':>22} {'test pairs':>22}")
        for name, cfg in (("CNN's settings", old), ("new fit", new)):
            for part, w in (("face only", 1.0), ("text only", 0.0), ("fused", None)):
                v = score(cfg, fv, tv, val_pairs) if w is None else single(cfg, w, val_pairs, fv, tv)
                t = score(cfg, ft, tt, test_pairs) if w is None else single(cfg, w, test_pairs, ft, tt)
                log(f"{name + ', ' + part:30} {v[0]:.3f} +/- {v[1]:.3f}     {t[0]:.3f} +/- {t[1]:.3f}")
        better = score(new, fv, tv, val_pairs)[0] > score(old, fv, tv, val_pairs)[0]
        config_name = "fusion_config.json" if better else "fusion_config_candidate.json"
        log(f"\nThe new fit {'beats' if better else 'does not beat'} the CNN's settings on the validation pairs: "
            + (f"saved as models/fusion/{args.face}/fusion_config.json, which the app uses with this face model."
               if better else f"not used (saved as {config_name} only); the app keeps the CNN's settings."))

    # --- outputs
    for df, name in [(val_sweep, "sweep_validation.csv"), (test_sweep, "sweep_test.csv")]:
        df.drop(columns="scores").to_csv(os.path.join(args.out_dir, name), index=False, float_format="%.4f")
    with open(os.path.join(args.out_dir, config_name), "w") as f:
        json.dump({"method": chosen.method, "face_weight": float(chosen.face_weight),
                   "face_temperature": round(t_face, 4), "text_temperature": round(t_text, 4),
                   **({"face_input_scaling": "pixels / 255"} if args.face == "cnn" else {"face_model": args.face}),
                   "chosen_on": "validation pairs (FER2013 PublicTest x GoEmotions validation)"}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, df, title in [(axes[0], val_sweep, "Validation pairs (used to choose weight)"),
                          (axes[1], test_sweep, "Test pairs (reported, not used for choosing)")]:
        for m, colour in zip(METHODS, ["tab:blue", "tab:orange"]):
            d = df[df.method == m]
            ax.plot(d.face_weight, d.macro_f1_mean, label=m, color=colour)
            ax.fill_between(d.face_weight, d.macro_f1_mean - d.macro_f1_std, d.macro_f1_mean + d.macro_f1_std,
                            color=colour, alpha=0.15)
        ax.axvline(chosen.face_weight, color="grey", ls="--", lw=1)
        ax.set_title(title); ax.set_xlabel("face_weight (text_weight = 1 - face_weight)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (mean +/- std over pairings)"); axes[0].legend()
    fig.suptitle(f"Face + text fusion on synthetic same-emotion pairs (dashed: chosen w={chosen.face_weight:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "fusion_sweep.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=EMOTIONS).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title(f"Fused ({chosen.method}, w={chosen.face_weight:.2f}), test pairs")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)

    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
