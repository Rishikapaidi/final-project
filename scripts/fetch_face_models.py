"""Download the ONNX face-expression models into models/face/onnx/ and check them.

Usage (from the project root):
    python scripts/fetch_face_models.py              FER+ (the app's default face model)
    python scripts/fetch_face_models.py --hsemotion  FER+ and HSEmotion (to compare with FACE_MODEL=hsemotion)
    python scripts/fetch_face_models.py --labels     also the FER+ labels for scripts/evaluate_face.py

  FER+  emotion-ferplus-8.onnx, from the ONNX model zoo (github.com/onnx/models,
        validated/vision/body_analysis/emotion_ferplus). MIT licence. Trained by
        Microsoft on the FER2013 training images with the FER+ labels.
  HSEmotion  enet_b0_8_best_afew.onnx, from github.com/HSE-asavchenko/face-emotion-recognition
        (code Apache-2.0). Fine-tuned on AffectNet, whose licence allows only
        non-commercial research and education: use these weights on those terms.
  FER+ labels  fer2013new.csv from github.com/microsoft/FERPlus (MIT licence), saved to
        data/face_emotion/raw/ (crowd-sourced labels for the FER2013 images, in their order).

Without these files the app uses the original CNN (model.h5). Nothing is
trained here; the files are used as published.
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "face", "onnx")
LABELS = ("fer2013new.csv", "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv",
          "9206e20d62f56475939f516847d61753e4860caeac9718b560129541b776fc2c")
FILES = {
    "emotion-ferplus-8.onnx": ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
                               "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"),
    "enet_b0_8_best_afew.onnx": ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
                                 "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    names = ["emotion-ferplus-8.onnx"] + (["enet_b0_8_best_afew.onnx"] if "--hsemotion" in sys.argv else [])
    jobs = [(name, *FILES[name], OUT) for name in names]
    if "--labels" in sys.argv:
        jobs.append((*LABELS, os.path.join(ROOT, "data", "face_emotion", "raw")))
    for name, url, digest, folder in jobs:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        if os.path.exists(path) and sha256(path) == digest:
            print(f"{name}: already here, checksum OK")
            continue
        print(f"{name}: downloading ...", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        if sha256(path + ".part") != digest:
            os.remove(path + ".part")
            sys.exit(f"{name}: the download's SHA-256 doesn't match; nothing saved")
        os.replace(path + ".part", path)
        print(f"{name}: saved to {os.path.relpath(path, ROOT)}, checksum OK")


if __name__ == "__main__":
    main()
