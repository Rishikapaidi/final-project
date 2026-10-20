"""Check how well the face model reads your own photos, and compare three ways of cropping the face.

Put photos in a folder with one subfolder per emotion, for example

    my_photos/
        happy/   IMG_0012.jpg ...
        sad/     ...
        neutral/ ...

Subfolder names can be any of the seven emotions (angry, disgust, fear, happy,
sad, surprise, neutral) or a common variant (anger, disgusted, fearful, scared,
happiness, sadness, surprised).

For each photo it prints the Haar face box, the size of the face crop that is
shrunk to 48x48, and the face model's top three emotions. It then prints a
confusion matrix and per-emotion recall for each crop:

    (a) current        the app's crop: exactly the Haar box
    (b) padded 10%     the box widened by 10% of its size on each side
    (c) equalized      the Haar box with histogram equalization

and says whether (b) or (c) is clearly better than (a). The photos are only
read; the only file written is the results text file.

Usage (from the project root):
    python scripts/diagnose_face.py path/to/my_photos
    python scripts/diagnose_face.py path/to/my_photos --out results.txt
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face  # noqa: E402
from fusion import EMOTIONS  # noqa: E402

VARIANTS = [("a", "current", {}), ("b", "padded 10%", {"pad": 0.10}), ("c", "equalized", {"equalize": True})]
FOLDER_NAMES = {**{e: e for e in EMOTIONS}, "anger": "angry", "disgusted": "disgust", "fearful": "fear",
                "scared": "fear", "happiness": "happy", "sadness": "sad", "surprised": "surprise"}
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# "Clearly better" needs both a meaningful gain and one unlikely to be luck on these photos.
MIN_GAIN = 0.05
MAX_P_VALUE = 0.05


def find_photos(folder):
    photos, skipped = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        emotion = FOLDER_NAMES.get(name.strip().lower())
        if emotion is None:
            skipped.append(name)
            continue
        for root, _, files in os.walk(path):
            photos += [(os.path.join(root, f), emotion) for f in sorted(files) if f.lower().endswith(IMAGE_TYPES)]
    return photos, skipped


def mcnemar_p(fixed, broken):
    """Two-sided exact McNemar test: chance of a split this uneven if neither crop were better."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def confusion_table(true, pred):
    present = [e for e in EMOTIONS if e in true]
    width = max(len(e) for e in EMOTIONS) + 2
    lines = ["true \\ predicted".ljust(width + 6) + "".join(e[:7].rjust(9) for e in EMOTIONS)]
    for t in present:
        row = [sum(1 for a, b in zip(true, pred) if a == t and b == p) for p in EMOTIONS]
        lines.append(t.ljust(width + 6) + "".join(str(n).rjust(9) for n in row))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="folder with one subfolder per emotion")
    parser.add_argument("--out", default=None, help="results file (default: face_diagnosis.txt inside the folder)")
    args = parser.parse_args()
    out_path = args.out or os.path.join(args.folder, "face_diagnosis.txt")

    lines = []
    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    photos, skipped = find_photos(args.folder)
    if skipped:
        say(f"Ignored subfolders (not an emotion name): {', '.join(skipped)}")
    if not photos:
        say("No photos found. Put them in subfolders named after the emotion, e.g. happy/, sad/.")
        return
    model = face.build_model(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.h5"))
    detector = face.detector()

    results = []   # (true emotion, {variant key: predicted emotion})
    no_face, unreadable = [], []
    say(f"{len(photos)} photos in {args.folder}\n")
    say("PER PHOTO (top three emotions are the face model's output for the current crop)")
    for path, emotion in photos:
        name = os.path.relpath(path, args.folder)
        with open(path, "rb") as f:
            frame = face.load_photo(f.read())
        if frame is None:
            unreadable.append(name)
            say(f"{name}: could not be read as a picture")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = face.find_faces(gray, detector)
        if not boxes:
            no_face.append(name)
            say(f"{name}: no face found ({frame.shape[1]}x{frame.shape[0]} image)")
            continue
        box = max(boxes, key=lambda b: b[2] * b[3])   # largest face, if there are several
        probs = face.predict(model, [face.crop(gray, box, **options) for _, _, options in VARIANTS])
        ordered = [dict(zip(EMOTIONS, face.as_emotions(p))) for p in probs]
        predicted = {key: max(o, key=o.get) for (key, _, _), o in zip(VARIANTS, ordered)}
        top3 = sorted(ordered[0], key=ordered[0].get, reverse=True)[:3]
        x, y, w, h = box
        mark = "ok" if predicted["a"] == emotion else "WRONG"
        say(f"{name} [{emotion}]: box x={x} y={y} w={w} h={h} in {frame.shape[1]}x{frame.shape[0]}"
            f"{f' ({len(boxes)} faces, largest used)' if len(boxes) > 1 else ''} | crop {w}x{h} -> 48x48 | "
            + ", ".join(f"{e} {ordered[0][e]:.0%}" for e in top3)
            + f" | {mark} | padded: {predicted['b']}, equalized: {predicted['c']}")
        results.append((emotion, predicted))

    say()
    say(f"Photos with a face: {len(results)} of {len(photos)}"
        + (f" | no face found: {len(no_face)}" if no_face else "")
        + (f" | unreadable: {len(unreadable)}" if unreadable else ""))
    if not results:
        return
    true = [t for t, _ in results]
    accuracy = {}
    for key, label, _ in VARIANTS:
        pred = [p[key] for _, p in results]
        accuracy[key] = sum(a == b for a, b in zip(true, pred)) / len(true)
        say(f"\n== ({key}) {label}: accuracy {accuracy[key]:.1%} ({sum(a == b for a, b in zip(true, pred))}/{len(true)})")
        for line in confusion_table(true, pred):
            say("   " + line)
        recalls = {e: np.mean([p == e for t, p in zip(true, pred) if t == e]) for e in EMOTIONS if e in true}
        say("   recall: " + ", ".join(f"{e} {r:.0%} (n={true.count(e)})" for e, r in recalls.items())
            + f" | mean recall {np.mean(list(recalls.values())):.1%}")

    say("\n== COMPARED WITH THE CURRENT CROP")
    verdicts = []
    for key, label, _ in VARIANTS[1:]:
        fixed = sum(p["a"] != t and p[key] == t for t, p in results)
        broken = sum(p["a"] == t and p[key] != t for t, p in results)
        gain, pval = accuracy[key] - accuracy["a"], mcnemar_p(fixed, broken)
        clearly = gain >= MIN_GAIN and pval < MAX_P_VALUE
        verdicts.append((clearly, gain, key, label))
        say(f"({key}) {label}: {gain:+.1%} accuracy | fixes {fixed} photo(s), breaks {broken} | "
            f"exact McNemar p = {pval:.3f} | {'CLEARLY BETTER' if clearly else 'not clearly better'}")
    best = max((v for v in verdicts if v[0]), default=None, key=lambda v: v[1])
    say(f"\nVerdict: ({best[2]}) {best[3]} is clearly better than the current crop."
        if best else
        f"\nVerdict: neither alternative is clearly better (needs at least +{MIN_GAIN:.0%} accuracy and "
        f"p < {MAX_P_VALUE}); keep the current crop.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
