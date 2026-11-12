"""Compare the face-expression models (face_models.py) on FER2013 PrivateTest.

Every model is fed exactly as the app feeds it: each 48x48 test face is
enlarged into a 640x480 grey frame (as a webcam frame would hold a face),
found with the app's face detection (face.find_faces), and the SAME box is
given to every model, which builds its own crop from the frame (the CNN:
48x48 grayscale and its mirror image; FER+: 64x64 grayscale; HSEmotion:
224x224 colour, here grayscale copied into three channels, as FER2013 has no
colour). Only faces the detector finds are scored, and all models score the
same faces.

Labels:
  original  the FER2013 labels (one per face; known to be noisy).
  FER+      the FER+ relabelling (Microsoft, MIT licence): the majority of 10
            crowd-sourced votes per face, leaving out faces whose majority is
            "unknown" or "not a face"; "contempt" counts as disgust.
            Read from data/face_emotion/raw/fer2013new.csv when it is there
            (python scripts/fetch_face_models.py --labels).

Fair-comparison notes, also printed in the report: the CNN and FER+ were
trained on the FER2013 training images (the CNN with the original labels,
FER+ with the FER+ labels), so this test set is their own kind of picture;
HSEmotion was trained on AffectNet's colour photos and sees grayscale 48-pixel
faces here, which works against it. Nothing is tuned on this test set.

Usage (from the project root):
    python scripts/evaluate_face.py                 every model whose files are here
    python scripts/evaluate_face.py cnn ferplus     only these

Reads   data/face_emotion/raw/privateTest-0000.parquet (and fer2013new.csv),
        model.h5, models/face/onnx/*.onnx
Writes  models/face/report.txt
"""
import io
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, recall_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import face  # noqa: E402
import face_models  # noqa: E402
from fusion import EMOTIONS, align_face  # noqa: E402

FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # dataset label ids
FERPLUS_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
FERPLUS_TO_OURS = {"neutral": "neutral", "happiness": "happy", "surprise": "surprise", "sadness": "sad", "anger": "angry",
                   "disgust": "disgust", "fear": "fear", "contempt": "disgust"}
FRAME_SIZE = (480, 640)   # the frame each face is placed on
FACE_SIZE = 240           # the size the 48x48 face is enlarged to inside that frame
NEGATIVE = ["angry", "disgust", "fear"]
LICENCES = {
    "cnn": "model.h5, this project's own CNN, trained on the FER2013 training images (Kaggle / ICML 2013 challenge data).",
    "ferplus": "emotion-ferplus-8.onnx, ONNX model zoo, MIT licence; trained by Microsoft on the FER2013 training images with the "
               "FER+ crowd-sourced labels (labels MIT licence). The FER2013 images themselves come from a Kaggle research "
               "challenge and carry no stated licence.",
    "hsemotion": "enet_b0_8_best_afew.onnx, HSEmotion (A. Savchenko), code Apache-2.0; EfficientNet-B0 pretrained on VGGFace2 "
                 "and fine-tuned on AffectNet, whose licence allows non-commercial research and education only: these "
                 "weights carry that limit.",
}


def frame_of(img):
    frame = np.full(FRAME_SIZE, 128, np.uint8)
    top, left = (FRAME_SIZE[0] - FACE_SIZE) // 2, (FRAME_SIZE[1] - FACE_SIZE) // 2
    frame[top:top + FACE_SIZE, left:left + FACE_SIZE] = cv2.resize(img, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
    return frame


def scores(true, pred):
    rec = recall_score(true, pred, labels=EMOTIONS, average=None, zero_division=0)
    neg, hap = np.isin(true, NEGATIVE), true == "happy"
    return {"accuracy": np.mean(true == pred), "macro-F1": f1_score(true, pred, labels=EMOTIONS, average="macro", zero_division=0),
            "recall": dict(zip(EMOTIONS, rec)), "neg->happy": np.mean(pred[neg] == "happy"),
            "happy->neutral/sad": np.mean(np.isin(pred[hap], ["neutral", "sad"])), "n": len(true)}


def main():
    split = pd.read_parquet(os.path.join(ROOT, "data", "face_emotion", "raw", "privateTest-0000.parquet"))
    images = [np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in split.image]
    original = np.array([FER2013_LABELS[i] for i in split.label])
    plus_path = os.path.join(ROOT, "data", "face_emotion", "raw", "fer2013new.csv")
    plus = plus_ok = None
    if os.path.exists(plus_path):
        fp = pd.read_csv(plus_path)
        fp = fp[fp.Usage == "PrivateTest"].reset_index(drop=True)
        votes = fp[FERPLUS_COLUMNS].values
        plus = np.array([FERPLUS_TO_OURS[FERPLUS_COLUMNS[i]] for i in votes.argmax(1)])
        plus_ok = votes.max(1) >= fp[["unknown", "NF"]].values.max(1)
        agree = np.mean(plus == original)
    lines = []
    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    detector = face.detector()
    frames = [frame_of(img) for img in images]
    found = [(i, max(b, key=lambda x: x[2] * x[3])) for i, b in ((i, face.find_faces(f, detector)) for i, f in enumerate(frames)) if b]
    kept = np.array([i for i, _ in found])
    wanted = sys.argv[1:] or list(face_models.NAMES)
    log("Face-expression models on FER2013 PrivateTest, through the app's path (detector, the same box for every model)\n")
    log(f"faces found by the app's detector: {len(kept):,} of {len(images):,} ({len(kept) / len(images):.0%}); all models are scored on these")
    if plus is not None:
        log(f"FER+ labels: {int(plus_ok[kept].sum()):,} of these faces have an emotion as the FER+ majority "
            f"(FER+ and the original label agree on {agree:.1%} of the test set)")
    log("")

    results, matrices = {}, {}
    for key in wanted:
        model = face_models.load(key)
        if model.key != key:
            log(f"{face_models.NAMES[key]}: skipped ({model.fallback})\n")
            continue
        t0 = time.time()
        probs = model.predict([model.prepare(frames[i], frames[i], box) for i, box in found])
        ms = (time.time() - t0) / len(found) * 1000
        pred = np.array([EMOTIONS[int(align_face(p).argmax())] for p in probs])
        results[key] = {"name": model.name, "ms": ms, "original": scores(original[kept], pred)}
        if plus is not None:
            m = plus_ok[kept]
            results[key]["FER+"] = scores(plus[kept][m], pred[m])
            matrices[key] = confusion_matrix(plus[kept][m], pred[m], labels=EMOTIONS)
        else:
            matrices[key] = confusion_matrix(original[kept], pred, labels=EMOTIONS)

    label_sets = ["original"] + (["FER+"] if plus is not None else [])
    log("== Summary")
    rows = []
    for key, r in results.items():
        row = {"model": r["name"], "ms / face": round(r["ms"], 1)}
        for ls in label_sets:
            row[f"accuracy ({ls})"] = f"{r[ls]['accuracy']:.1%}"
            row[f"macro-F1 ({ls})"] = f"{r[ls]['macro-F1']:.3f}"
        rows.append(row)
    log(pd.DataFrame(rows).set_index("model").to_string())
    log("")
    for ls in label_sets:
        log(f"== Recall per emotion, {ls} labels")
        log(pd.DataFrame({r["name"]: {e: f"{v:.1%}" for e, v in r[ls]["recall"].items()} for r in results.values()}).to_string())
        log("")
        log(f"== The failures seen on webcams, {ls} labels")
        log(pd.DataFrame({r["name"]: {"angry / disgust / fear read as happy": f"{r[ls]['neg->happy']:.1%}",
                                      "happy read as neutral or sad": f"{r[ls]['happy->neutral/sad']:.1%}"}
                          for r in results.values()}).to_string())
        log("")
    for key, r in results.items():
        log(f"== Confusion matrix, {r['name']} ({'FER+' if plus is not None else 'original'} labels; rows = true, columns = predicted)")
        log(pd.DataFrame(matrices[key], index=EMOTIONS, columns=EMOTIONS).to_string())
        log("")
    log("== What each model was trained on, and its licence")
    for key in results:
        log(f"{face_models.NAMES[key]}: {LICENCES[key]}")
    log("")
    log("Notes: the CNN and FER+ were trained on FER2013 training images, so this test set is their own kind of picture;")
    log("HSEmotion was trained on AffectNet's colour photos and sees grayscale 48-pixel faces here, which works against it.")
    log("The CNN's crop is averaged with its mirror image, as in the app. Nothing was tuned on this test set or on anyone's face.")

    out = os.path.join(ROOT, "models", "face", "report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
