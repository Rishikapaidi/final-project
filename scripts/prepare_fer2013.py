"""Download FER2013 from Hugging Face and check it matches the official dataset.

Source: Aaryan333/fer2013_train_publicTest_privateTest, a copy that keeps the
three official splits. Writes data/face_emotion/raw/{train,publicTest,privateTest}-0000.parquet
and fails if the image or class counts differ from official FER2013.

Usage (from the project root):
    python scripts/prepare_fer2013.py
"""
import os

import pandas as pd
import requests

DATASET = "Aaryan333/fer2013_train_publicTest_privateTest"
OUT_DIR = os.path.join("data", "face_emotion", "raw")
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
OFFICIAL_SPLITS = {"train": 28709, "publicTest": 3589, "privateTest": 3589}
OFFICIAL_CLASSES = {"Angry": 4953, "Disgust": 547, "Fear": 5121, "Happy": 8989,
                    "Sad": 6077, "Surprise": 4002, "Neutral": 6198}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = requests.get("https://datasets-server.huggingface.co/parquet",
                         params={"dataset": DATASET}, timeout=60).json()["parquet_files"]
    parts = []
    for f in files:
        path = os.path.join(OUT_DIR, f"{f['split']}-{os.path.basename(f['url'])}")
        if not os.path.exists(path):
            print(f"downloading {f['split']} ...")
            with open(path, "wb") as out:
                out.write(requests.get(f["url"], timeout=600).content)
        parts.append(pd.read_parquet(path, columns=["label"]).assign(split=f["split"]))
    df = pd.concat(parts, ignore_index=True)

    splits = df.split.value_counts().to_dict()
    classes = df.label.map(dict(enumerate(LABELS))).value_counts().to_dict()
    assert len(df) == 35887, f"expected 35,887 images, got {len(df):,}"
    assert splits == OFFICIAL_SPLITS, f"split sizes differ from official FER2013: {splits}"
    assert classes == OFFICIAL_CLASSES, f"class counts differ from official FER2013: {classes}"
    print(f"OK: {len(df):,} images, splits {splits}, class counts match official FER2013")


if __name__ == "__main__":
    main()
