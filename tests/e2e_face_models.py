"""The face-model switch (face_models.py, FACE_MODEL=cnn / ferplus / hsemotion).

Without the browser: each model loads (the ONNX ones from models/face/onnx/,
fetched by scripts/fetch_face_models.py) and returns seven probabilities in
fusion.FACE_MODEL_LABELS order that sum to one; every model reads the same box
(the CNN's crop is exactly face.crop's; FER+ and the CNN read grayscale, so a
colour change doesn't move them, HSEmotion reads the box in colour, so it
does); "contempt" is added to disgust; a missing file or an unknown name falls
back to the CNN, saying why; the default is HSEmotion. With the browser (sad
face): the app run with FACE_MODEL=ferplus reads the scan with FER+ and says
so on the face tile, and the mood card still checks out.
"""
import io, os, sys, time
import numpy as np, pandas as pd, cv2
from PIL import Image
from playwright.sync_api import sync_playwright
from pagekit import ROOT, describe_card, log, path_frame, read_card, start_server, stop

sys.path.insert(0, ROOT)
import face, face_models  # noqa: E402
from fusion import FACE_MODEL_LABELS  # noqa: E402

PORT = 8628
failures = []
def check(name, ok, detail=""):
    log(f"{'PASS' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))
    if not ok: failures.append(name)

p = pd.read_parquet(f"{ROOT}/data/face_emotion/raw/privateTest-0000.parquet")
img = np.asarray(Image.open(io.BytesIO(p.image.iloc[3139]["bytes"])).convert("L"))
canvas = np.full((480, 640), 128, np.uint8); canvas[120:360, 200:440] = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
box = face.find_faces(gray, face.detector())[0]

log("== EACH MODEL")
check("the default is HSEmotion", face_models.wanted() == "hsemotion" and face_models.FACE_MODEL_DEFAULT == "hsemotion", f"wanted {face_models.wanted()!r}")
for key in ("cnn", "ferplus", "hsemotion"):
    t0 = time.time(); m = face_models.load(key); t = time.time() - t0
    probs = m.predict([m.prepare(frame, gray, box)])
    top = FACE_MODEL_LABELS[int(probs[0].argmax())]
    check(f"{key}: loads as itself", m.key == key and m.fallback is None, f"{m.name}, {t:.1f}s")
    check(f"{key}: 7 probabilities in FACE_MODEL_LABELS order, summing to 1", probs.shape == (1, 7) and abs(probs.sum() - 1) < 1e-4,
          f"top {top} {probs[0].max():.0%}")
    check(f"{key}: no inputs gives an empty (0, 7) result", m.predict([]).shape == (0, 7))
cnn = face_models.load("cnn")
check("the CNN's crop is exactly face.crop's", np.array_equal(cnn.prepare(frame, gray, box), face.crop(gray, box)))
fp, hse = face_models.load("ferplus"), face_models.load("hsemotion")
check("FER+ reads the box in grayscale (64 x 64)", fp.prepare(frame, gray, box).shape == (1, 64, 64))
colour = frame.copy(); colour[:, :, 0] = 0   # no blue: the colour changes, the gray image passed in does not
check("HSEmotion reads the box in colour from the original frame (224 x 224)",
      hse.prepare(frame, gray, box).shape == (3, 224, 224) and not np.array_equal(hse.prepare(colour, gray, box), hse.prepare(frame, gray, box)))
check("FER+ ignores the frame's colour (it uses the gray image)", np.array_equal(fp.prepare(colour, gray, box), fp.prepare(frame, gray, box)))
mixed = face_models.to_face_labels(np.array([[0, 0, 0, 0, 0, 0.2, 0, 0.5]]), face_models.FERPLUS_LABELS)[0]
check("'contempt' is added to disgust", abs(mixed[FACE_MODEL_LABELS.index("Disgusted")] - 0.7) < 1e-9, f"disgust {mixed[FACE_MODEL_LABELS.index('Disgusted')]:.2f}")

log("== FALLBACK")
real = face_models.ONNX_DIR
face_models.ONNX_DIR = os.path.join(ROOT, "tests", "output", "no_such_folder")
m = face_models.load("ferplus")
check("a missing ONNX file falls back to the CNN, saying why", m.key == "cnn" and "missing" in (m.fallback or ""), repr(m.fallback))
face_models.ONNX_DIR = real
os.environ["FACE_MODEL"] = "nonsense"
check("an unknown FACE_MODEL means the default", face_models.wanted() == "hsemotion")
os.environ["FACE_MODEL"] = "ferplus"
check("FACE_MODEL=ferplus is read", face_models.wanted() == "ferplus")
del os.environ["FACE_MODEL"]

log("== THE SWITCH IN THE APP (FACE_MODEL=ferplus)")
server = start_server(PORT, "sad", "face_models", FACE_MODEL="ferplus")
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}"); page.wait_for_selector(".st-key-path iframe", timeout=180000); path_frame(page); time.sleep(1.5)
        label = page.locator(".st-key-facetile .tile-label").inner_text()
        card = read_card(page)
        check("the face tile says FER+ read the scan", "FER+" in label, repr(label))
        check("the mood card checks out with FER+'s reading", "all checks pass" in describe_card(card), describe_card(card)[:160])
        check("no page errors", not errors, str(errors[:3]))
        browser.close()
finally:
    stop(server)
log("ALL PASS" if not failures else "FAILED: " + ", ".join(failures))
sys.exit(1 if failures else 0)
