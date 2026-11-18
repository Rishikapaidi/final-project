"""Fusion when one signal is missing, and the face / text / fused comparison on the same test pairs.

When there is no text, the app's combine_emotions uses the face alone (face
weight 1, a uniform text vector); when there is no face, the text alone. This
runs that code path (app.py's own combine_emotions, via scripts/app_code.py)
on the test pairs of scripts/evaluate_fusion.py (FER2013 PrivateTest x
GoEmotions test, 50 pairs per emotion x 7 emotions, 20 pairings, seed 42),
checks that it gives exactly the single model's calibrated probabilities,
and then puts face only, text only and both fusion rules in one table, for
HSEmotion (the app's default) and the CNN.

Each model uses its own fusion settings (face_models.fusion_config): both
rules at face weight 0.55, the weight chosen on the validation pairs for
both rules and both models (models/fusion/report.txt and
models/fusion/hsemotion/report.txt). The app uses log_linear;
weighted_average is shown for comparison only.

Nothing is trained or tuned here; the model outputs come from the caches in
data/fusion/ (see scripts/eval_common.py).

Usage (from the project root):
    python scripts/evaluate_fusion_missing.py
Writes  models/fusion/missing/report.txt, models/fusion/missing/summary.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import eval_common as ec
from fusion import EMOTIONS, align_face, align_text, calibrate

OUT = os.path.join(ec.ROOT, "models", "fusion", "missing")


def predictions(setup, pairing, face, text):
    f_idx, t_idx, _ = pairing
    return np.array([setup.fused(f, t, face, text)[1].argmax() for f, t in zip(f_idx, t_idx)])


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("FUSION WHEN ONE SIGNAL IS MISSING (app.py combine_emotions, run from its own source)")
    log("Test pairs: FER2013 PrivateTest x GoEmotions test, same true emotion, 50 pairs x 7 emotions = 350 per pairing,")
    log("20 pairings (seed 42; the same pairs as scripts/evaluate_fusion.py). Mean +/- std over the 20 pairings.\n")
    rows = []
    for key, name in ec.MODELS:
        app = ec.Setup(key)
        wavg = ec.Setup(key, method="weighted_average")
        cfg = app.config
        log(f"== {name}: T_face {cfg['face_temperature']:.4f}, T_text {cfg['text_temperature']:.4f}, "
            f"face weight {cfg['face_weight']:.2f} (from {('models/fusion/' + key + '/') if key != 'cnn' else 'models/fusion/'}fusion_config.json)")

        # 1. the single-source path equals the single model's calibrated output, on every test item
        face_only = np.array([app.fused(i, 0, text=False)[1] for i in range(len(app.face_raw))])
        text_only = np.array([app.fused(0, i, face=False)[1] for i in range(len(app.text_raw))])
        face_cal = calibrate(align_face(app.face_raw), cfg["face_temperature"])
        text_cal = calibrate(align_text(app.text_raw), cfg["text_temperature"])
        d_face, d_text = np.abs(face_only - face_cal).max(), np.abs(text_only - text_cal).max()
        same_face = np.array_equal(face_only.argmax(1), align_face(app.face_raw).argmax(1))
        same_text = np.array_equal(text_only.argmax(1), align_text(app.text_raw).argmax(1))
        acc_f = np.mean(np.array(EMOTIONS)[face_only.argmax(1)] == app.face_y)
        acc_t = np.mean(np.array(EMOTIONS)[text_only.argmax(1)] == app.text_y)
        log(f"No text -> face only, all {len(face_only):,} PrivateTest faces: max |app - calibrated face model| = {d_face:.1e}; "
            f"top emotion identical to the uncalibrated model's: {same_face}; accuracy {acc_f:.1%}")
        log(f"No face -> text only, all {len(text_only):,} GoEmotions test comments: max |app - calibrated text model| = {d_text:.1e}; "
            f"top emotion identical: {same_text}; accuracy {acc_t:.1%}")
        assert d_face < 1e-9 and d_text < 1e-9 and same_face and same_text

        # 2. the same test pairs: face only, text only, both rules
        pairs = ec.test_pairs(app.face_y, app.text_y)
        variants = [("Face only (no text)", app, True, False), ("Text only (no face)", app, False, True),
                    (f"Fused, log_linear, w={cfg['face_weight']:.2f} (app)", app, True, True),
                    (f"Fused, weighted_average, w={cfg['face_weight']:.2f}", wavg, True, True)]
        scores = {}
        for label, setup, face, text in variants:
            acc, f1 = [], []
            for p in pairs:
                pred = predictions(setup, p, face, text)
                acc.append(accuracy_score(p[2], pred))
                f1.append(f1_score(p[2], pred, labels=range(7), average="macro"))
            scores[label] = (np.array(acc), np.array(f1))
        better = max(list(scores)[:2], key=lambda k: scores[k][1].mean())
        log(f"\nSame test pairs ({len(pairs)} pairings x {len(pairs[0][2])} pairs). Better single model: {better.split(' (')[0].lower()}.")
        log(f"{'':44} {'accuracy':>16} {'macro-F1':>16}   fused beats the better single model (macro-F1)")
        for label, (acc, f1) in scores.items():
            beats = np.mean(f1 > scores[better][1]) if label.startswith("Fused") else None
            log(f"{label:44} {acc.mean():.3f} +/- {acc.std():.3f}  {f1.mean():.3f} +/- {f1.std():.3f}   "
                + (f"in {beats:.0%} of pairings ({int(round(beats * len(pairs)))} of {len(pairs)})" if beats is not None else ""))
            rows.append({"face_model": key, "variant": label, "accuracy_mean": acc.mean(), "accuracy_std": acc.std(),
                         "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(),
                         "share_pairings_beating_better_single": beats, "pairings": len(pairs)})

        # 3. cross-check against evaluate_fusion.py's own sweep on these pairs
        sweep = pd.read_csv(os.path.join(ec.ROOT, "models", "fusion", *([key] if key != "cnn" else []), "sweep_test.csv"))
        def swept(method, w):
            return float(sweep[(sweep.method == method) & np.isclose(sweep.face_weight, w)].macro_f1_mean.iloc[0])
        checks = [("face only", swept("weighted_average", 1.0), scores[variants[0][0]][1].mean()),
                  ("text only", swept("weighted_average", 0.0), scores[variants[1][0]][1].mean()),
                  ("log_linear", swept("log_linear", cfg["face_weight"]), scores[variants[2][0]][1].mean()),
                  ("weighted_average", swept("weighted_average", cfg["face_weight"]), scores[variants[3][0]][1].mean())]
        ok = all(abs(a - b) < 5e-4 for _, a, b in checks)
        log("Cross-check with evaluate_fusion.py's sweep_test.csv (macro-F1): "
            + ", ".join(f"{n} {a:.4f} vs {b:.4f}" for n, a, b in checks) + f" -> {'match' if ok else 'DIFFER'}\n")

    log("Notes: with one signal missing the fusion rule does not matter (the missing side gets weight 0), so 'face only' and")
    log("'text only' are the same for both rules. Faces are whole 48x48 FER2013 images (no webcam, no face detector). The CNN's")
    log("cached outputs (data/fusion/face_test.npz) are of one crop, not averaged with its mirror image as in the app;")
    log("HSEmotion's are as in the app. Pairs share a true emotion, so this shows agreement cases only (see models/fusion/conflict/).")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4f")
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
