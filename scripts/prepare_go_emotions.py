"""Download GoEmotions from Hugging Face and build the text-emotion CSVs.

Writes two files to data/text_emotion/ (or --out-dir):

  go_emotions.csv          all 54,263 comments: text, emotion
                           (multi-label comments have comma-joined emotions)
  go_emotions_single.csv   45,446 single-label comments:
                           text, emotion, split, emotion_mapped

Usage (from the project root):
    python scripts/prepare_go_emotions.py
"""
import argparse
import os

import pandas as pd

PARQUET_URL = (
    "https://huggingface.co/datasets/google-research-datasets/go_emotions/"
    "resolve/refs%2Fconvert%2Fparquet/simplified/{split}/0000.parquet"
)
SPLITS = ["train", "validation", "test"]

# Label ids used by the "simplified" config, in order.
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]

# GoEmotions (28) -> FER2013 (7). Based on the GoEmotions authors' Ekman
# grouping, except approval/disapproval go to neutral instead of happy/angry.
FER2013_MAPPING = {
    "angry": ["anger", "annoyance"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "happy": ["admiration", "amusement", "caring", "desire", "excitement",
              "gratitude", "joy", "love", "optimism", "pride", "relief"],
    "sad": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "surprise": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral", "approval", "disapproval"],
}
TO_FER2013 = {label: fer for fer, labels in FER2013_MAPPING.items() for label in labels}
assert sorted(TO_FER2013) == sorted(GO_EMOTIONS_LABELS)


def download():
    parts = []
    for split in SPLITS:
        part = pd.read_parquet(PARQUET_URL.format(split=split))
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=os.path.join("data", "text_emotion"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw = download()

    full = pd.DataFrame({
        "text": raw["text"],
        "emotion": raw["labels"].apply(lambda ids: ",".join(GO_EMOTIONS_LABELS[i] for i in ids)),
    })
    full.to_csv(os.path.join(args.out_dir, "go_emotions.csv"), index=False, encoding="utf-8")

    single = raw[raw["labels"].apply(len) == 1].reset_index(drop=True)
    emotion = single["labels"].apply(lambda ids: GO_EMOTIONS_LABELS[ids[0]])
    out = pd.DataFrame({
        "text": single["text"],
        "emotion": emotion,
        "split": single["split"],
        "emotion_mapped": emotion.map(TO_FER2013),
    })
    out.to_csv(os.path.join(args.out_dir, "go_emotions_single.csv"), index=False, encoding="utf-8")

    print(f"go_emotions.csv: {len(full):,} rows")
    print(f"go_emotions_single.csv: {len(out):,} rows")
    print(out.groupby("split", sort=False)["emotion_mapped"].value_counts().unstack(0).to_string())


if __name__ == "__main__":
    main()
