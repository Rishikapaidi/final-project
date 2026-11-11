"""Face detection on a contrast-boosted copy (face.find_faces), without the browser.

1. Backlit face: a FER2013 face darkened to a quarter of its brightness on a
   bright wall (a gradient from 255 at the top to 210 at the bottom), as when
   a window or a lit wall is behind the person. The old method (the default
   cascade on the plain grayscale frame) finds nothing; the new one finds
   exactly one face, on the face. Two fixed faces, one for each way the new
   method finds it: the default cascade on the boosted copy (as on the user's
   backlit screenshot), and the alt2 cascade when the default still finds
   nothing. The same is counted over the first 300 test faces, for information.
2. Still no face in the blank grey frames (webcam and upload tests) and in
   noise frames.
3. The fake-camera and upload test frames still give exactly one face, with
   the same box as before (the plain search runs first; the boost is a fallback).
4. The crop sent to the model comes from the original frame, not the boosted
   copy: find_faces leaves the frame untouched, and the same box gives the
   same crop and the same prediction as before.
"""
import io, os, sys
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import face

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
BACKLIT_ROWS = {3: "default", 0: "alt2"}   # test face -> the cascade that should find it on the boosted copy
DARK, WALL_TOP, WALL_BOTTOM = 0.25, 255, 210

det = face.detector()
old = lambda gray: face.haar_boxes(gray, det.default)     # what the app did before
new = lambda gray: face.find_faces(gray, det)
p = pd.read_parquet(f"{ROOT}/data/face_emotion/raw/privateTest-0000.parquet")
fz = np.load(f"{ROOT}/data/fusion/face_test.npz"); fprobs, flabels = fz["probs"], fz["labels"].astype(str)
image = lambda i: np.asarray(Image.open(io.BytesIO(p.image.iloc[i]["bytes"])).convert("L"))
failures = []
def check(name, ok, detail=""):
    print(f"RESULT {'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""), flush=True)
    if not ok: failures.append(name)

def backlit(i):
    wall = np.linspace(WALL_TOP, WALL_BOTTOM, 480)[:, None] * np.ones((1, 640))
    wall[120:360, 200:440] = cv2.resize(image(i), (240, 240), interpolation=cv2.INTER_CUBIC).astype(float) * DARK
    return wall.clip(0, 255).astype(np.uint8)

def on_canvas(i, size=(480, 640), top=120, left=200, side=240, grey=128):
    canvas = np.full(size, grey, np.uint8)
    canvas[top:top + side, left:left + side] = cv2.resize(image(i), (side, side), interpolation=cv2.INTER_CUBIC)
    return canvas

def test_frame(emotion, **kw):
    """The frame the browser tests use for `emotion` (the same selection as fake_camera_server.py)."""
    k = EMO.index(emotion)
    for i in np.argsort(-fprobs[:, k]):
        if flabels[i] == emotion and len(old(canvas := on_canvas(i, **kw))) == 1:
            return i, canvas

# 1. backlit
for row, cascade in BACKLIT_ROWS.items():
    gray = backlit(row)
    boxes, default_boosted = new(gray), face.haar_boxes(face.boost_contrast(gray), det.default)
    on_face = len(boxes) == 1 and all(abs(a - b) < 60 for a, b in zip(boxes[0], (200, 120, 240, 240)))
    check(f"backlit face (row {row}): old method finds nothing", old(gray) == [], f"old {old(gray)}")
    check(f"backlit face (row {row}): new method finds exactly one face, on the face", on_face, f"new {boxes} (face at (200, 120, 240, 240))")
    check(f"backlit face (row {row}): found by the {cascade} cascade on the boosted copy",
          ("default" if default_boosted else "alt2") == cascade, f"default cascade on the boosted copy {default_boosted}")
counts = {"old finds one": 0, "new finds one": 0, "old none, new one": 0, "new more than one": 0}
for i in range(300):
    g = backlit(i); o, n = old(g), new(g)
    counts["old finds one"] += len(o) == 1; counts["new finds one"] += len(n) == 1
    counts["old none, new one"] += not o and len(n) == 1; counts["new more than one"] += len(n) > 1
print(f"RESULT info: first 300 test faces made backlit the same way: {counts}")

# 2. no face
rng = np.random.default_rng(0)
for name, frame in [("blank grey webcam frame (128)", np.full((480, 640), 128, np.uint8)),
                    ("blank grey upload (140)", np.full((600, 800), 140, np.uint8)),
                    *[(f"noise frame {s}", rng.integers(0, 256, (480, 640), dtype=np.uint8)) for s in range(5)]]:
    check(f"no face in the {name}", new(frame) == [], f"new {new(frame)}")

# 3. the browser tests' frames
frames = {f"fake camera {e}": test_frame(e) for e in ("sad", "happy")}
frames["fake camera mixed"] = (977, on_canvas(977))
for e in ("happy", "sad"):
    frames[f"upload {e}"] = test_frame(e, size=(960, 1280), top=280, left=400, side=480, grey=140)
for name, (row, frame) in frames.items():
    check(f"{name} (row {row}): exactly one face, the same box as before", len(new(frame)) == 1 and new(frame) == old(frame),
          f"old {old(frame)} new {new(frame)}")

# 4. the crop comes from the original frame
model = face.build_model(f"{ROOT}/model.h5")
for name, (row, frame) in frames.items():
    before = frame.copy(); box = new(frame)[0]
    untouched = np.array_equal(frame, before)
    reference = cv2.resize(frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]], (48, 48))[:, :, None].astype("float32") / 255.0
    same_crop = np.array_equal(face.crop(frame, box), reference)
    old_box = old(frame)[0]
    same_prediction = np.array_equal(face.predict(model, [face.crop(frame, old_box)]), face.predict(model, [face.crop(before, old_box)]))
    check(f"{name}: frame untouched, crop from the original pixels, same box gives the same prediction",
          untouched and same_crop and same_prediction, f"untouched {untouched}, crop {same_crop}, prediction {same_prediction}")

print(f"RESULT {'ALL PASS' if not failures else 'FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
