"""Two contact sheets of FER2013 PrivateTest faces (public benchmark images), to look at.

  docs/fer2013_examples.png  one row per emotion, 8 faces each, with the dataset's label
  docs/fer2013_misreads.png  one row per emotion, 6 faces the app's default face model (the CNN)
                             got wrong through the app's own path (detector, crop, mirror average),
                             each with "true: X" and "CNN: Y"

The faces are picked with a fixed random seed, so the sheets come out the same each time.
FER2013 comes from the ICML 2013 "Challenges in Representation Learning" facial-expression
challenge (Goodfellow et al., 2013, arXiv:1307.0414); it is meant for research use.

Usage (from the project root):
    python scripts/make_fer2013_sheet.py

Reads   data/face_emotion/raw/privateTest-0000.parquet, model.h5
Writes  docs/fer2013_examples.png, docs/fer2013_misreads.png
"""
import io
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import face  # noqa: E402
import face_models  # noqa: E402
from fusion import EMOTIONS, align_face  # noqa: E402

FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # dataset label ids
ROWS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SEED = 2013
TILE, GAP, LEFT, CAPTION, TOP = 96, 10, 96, 34, 40
FOOT = "FER2013 PrivateTest (Goodfellow et al., 2013, arXiv:1307.0414), research use"


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheet(title, rows, columns, path):
    """rows: [(row label, [(48x48 face, caption lines), ...]), ...] -> a PNG at path."""
    width = LEFT + columns * (TILE + GAP)
    height = TOP + len(rows) * (TILE + CAPTION + GAP) + 24
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((10, 10), title, fill="black", font=font(16))
    for r, (label, cells) in enumerate(rows):
        y = TOP + r * (TILE + CAPTION + GAP)
        draw.text((10, y + TILE // 2 - 8), label, fill="black", font=font(15))
        for c, (img, lines) in enumerate(cells):
            x = LEFT + c * (TILE + GAP)
            page.paste(Image.fromarray(img).resize((TILE, TILE), Image.BICUBIC).convert("RGB"), (x, y))
            for k, line in enumerate(lines):
                draw.text((x, y + TILE + 3 + k * 15), line, fill="black" if k == 0 else "#b00020", font=font(12))
    draw.text((10, height - 20), FOOT, fill="#555555", font=font(11))
    page.save(path, optimize=True)
    print(f"{os.path.relpath(path, ROOT)}: {page.width} x {page.height} px, {os.path.getsize(path) / 1024:.0f} KB")


def main():
    split = pd.read_parquet(os.path.join(ROOT, "data", "face_emotion", "raw", "privateTest-0000.parquet"))
    images = [np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("L")) for b in split.image]
    labels = np.array([FER2013_LABELS[i] for i in split.label])
    rng = np.random.default_rng(SEED)

    # 1. examples: 8 random faces per emotion
    rows = []
    for emotion in ROWS:
        picks = rng.choice(np.flatnonzero(labels == emotion), 8, replace=False)
        rows.append((emotion, [(images[i], [emotion]) for i in sorted(picks)]))
    sheet("FER2013 PrivateTest: 8 faces per emotion (the dataset's label under each)", rows, 8,
          os.path.join(ROOT, "docs", "fer2013_examples.png"))

    # 2. misreads: the app's path (face enlarged into a grey frame, the app's detector, the CNN on the same box)
    model, detector = face_models.load("cnn"), face.detector()
    found, inputs = [], []
    for i, img in enumerate(images):
        frame = np.full((480, 640), 128, np.uint8)
        frame[120:360, 200:440] = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
        boxes = face.find_faces(frame, detector)
        if boxes:
            found.append(i)
            inputs.append(model.prepare(frame, frame, max(boxes, key=lambda b: b[2] * b[3])))
    pred = np.array([EMOTIONS[int(align_face(p).argmax())] for p in model.predict(inputs)])
    found = np.array(found)
    rows = []
    for emotion in ROWS:
        wrong = np.flatnonzero((labels[found] == emotion) & (pred != emotion))
        picks = sorted(rng.choice(wrong, min(6, len(wrong)), replace=False))
        rows.append((emotion, [(images[found[k]], [f"true: {emotion}", f"CNN: {pred[k]}"]) for k in picks]))
    sheet("FER2013 PrivateTest faces the CNN got wrong (the app's detector, crop and mirror average)", rows, 6,
          os.path.join(ROOT, "docs", "fer2013_misreads.png"))


if __name__ == "__main__":
    main()
