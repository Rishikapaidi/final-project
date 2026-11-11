"""Evaluate the face model (model.h5) on the FER2013 PrivateTest split.

Two ways of feeding the test faces, both with the app's own preprocessing
(face.crop: grayscale, 48x48, pixels scaled to 0-1):

  (A) whole image   each 48x48 test image is already a face crop, so the whole
                    image is used as the face box. This measures the model.
  (B) Haar crop     each face is placed on a larger frame (as a webcam frame
                    would be), found with the app's face detection
                    (face.find_faces: the Haar cascade, then a contrast-boosted
                    copy if it finds nothing) and cropped to that box from the
                    original frame, as the app does. Only faces found are scored.

For each it reports per-class precision, recall and F1, a confusion matrix,
and how often each emotion is predicted compared with how often it is the
true label (to show over- or under-prediction).

Usage (from the project root):
    python scripts/evaluate_face.py

Reads   model.h5, data/face_emotion/raw/privateTest-0000.parquet
Writes  models/face/report.txt
"""
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face  # noqa: E402
from fusion import EMOTIONS  # noqa: E402

FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # dataset label ids
FRAME_SIZE = (480, 640)   # (B): frame the face is placed on
FACE_SIZE = 240           # (B): size the 48x48 face is scaled to inside that frame


def predicted_labels(model, crops):
    probs = face.predict(model, crops)
    return np.array([EMOTIONS[int(face.as_emotions(p).argmax())] for p in probs])


def section(title, true, pred, log):
    log(f"== {title}")
    log(f"faces scored: {len(true):,} | accuracy {np.mean(true == pred):.1%}")
    log(classification_report(true, pred, labels=EMOTIONS, digits=3, zero_division=0))
    cm = confusion_matrix(true, pred, labels=EMOTIONS)
    log("confusion matrix (rows = true, columns = predicted):")
    log(pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_string())
    log("")
    log("how often each emotion is predicted vs how often it is the true label:")
    rows = []
    for e in EMOTIONS:
        t, p = int((true == e).sum()), int((pred == e).sum())
        rows.append({"emotion": e, "true": t, "predicted": p, "true share": f"{t / len(true):.1%}",
                     "predicted share": f"{p / len(pred):.1%}", "predicted / true": f"{p / t:.2f}" if t else "-"})
    log(pd.DataFrame(rows).set_index("emotion").to_string())
    log("")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split = pd.read_parquet(os.path.join(root, "data", "face_emotion", "raw", "privateTest-0000.parquet"))
    images = [np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in split.image]
    true = np.array([FER2013_LABELS[i] for i in split.label])
    model = face.build_model(os.path.join(root, "model.h5"))
    lines = []
    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("Face model (model.h5) on FER2013 PrivateTest, with the app's preprocessing (face.crop)\n")

    # (A) whole 48x48 image as the face box
    whole = predicted_labels(model, [face.crop(img, (0, 0, img.shape[1], img.shape[0])) for img in images])
    section("(A) whole image (the model itself)", true, whole, log)

    # (B) the app's full path: Haar detection on a frame, crop to the Haar box
    detector = face.detector()
    crops, kept = [], []
    top, left = (FRAME_SIZE[0] - FACE_SIZE) // 2, (FRAME_SIZE[1] - FACE_SIZE) // 2
    for i, img in enumerate(images):
        frame = np.full(FRAME_SIZE, 128, np.uint8)
        frame[top:top + FACE_SIZE, left:left + FACE_SIZE] = cv2.resize(img, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
        boxes = face.find_faces(frame, detector)
        if boxes:
            crops.append(face.crop(frame, max(boxes, key=lambda b: b[2] * b[3])))
            kept.append(i)
    kept = np.array(kept)
    haar = predicted_labels(model, crops)
    log(f"Haar found a face in {len(kept):,} of {len(images):,} test images ({len(kept) / len(images):.0%}); "
        f"missed per emotion: " + ", ".join(f"{e} {int(((true == e).sum() - (true[kept] == e).sum()))}" for e in EMOTIONS) + "\n")
    section("(B) Haar crop, as the app does (faces Haar found)", true[kept], haar, log)
    section("(A) whole image, on the same faces as (B), for a like-for-like comparison", true[kept], whole[kept], log)

    out = os.path.join(root, "models", "face", "report.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
