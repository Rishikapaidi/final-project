"""Start the real Streamlit app with a fake camera (and optionally fake previews) for browser tests.

env: TEST_FRAME = sad | happy | blank ; TEST_PREVIEWS = fake | real ; TEST_PORT
"""
import base64, hashlib, io, os, struct, sys, math, tempfile, wave
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import previews

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
blank = np.full((480, 640, 3), 128, np.uint8)
def frame_from_row(p, i):
    img = np.asarray(Image.open(io.BytesIO(p.image.iloc[i]["bytes"])).convert("L"))
    canvas = np.full((480, 640), 128, np.uint8); canvas[120:360, 200:440] = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
def face_frame(emotion):
    p = pd.read_parquet(f"{ROOT}/data/face_emotion/raw/privateTest-0000.parquet")
    if emotion == "mixed":   # a moderately happy face: with a sad sentence the top emotion stays under 50%
        return frame_from_row(p, 977)
    fz = np.load(f"{ROOT}/data/fusion/face_test.npz"); fprobs, flabels = fz["probs"], fz["labels"].astype(str)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    k = EMO.index(emotion)
    for i in np.argsort(-fprobs[:, k]):
        if flabels[i] != emotion: continue
        img = np.asarray(Image.open(io.BytesIO(p.image.iloc[i]["bytes"])).convert("L"))
        canvas = np.full((480, 640), 128, np.uint8); canvas[120:360, 200:440] = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
        f = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        if len(haar.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5)) == 1:
            return f
which = os.environ.get("TEST_FRAME", "sad")
FRAME = blank if which == "blank" else face_frame(which)

CAM_LOG = os.environ.get("TEST_CAM_LOG")
class FakeCap:
    def __init__(self, *a):
        if CAM_LOG:
            with open(CAM_LOG, "a") as f: f.write("opened;")
    def read(self): return True, FRAME.copy()
    def release(self): pass
    def isOpened(self): return True
cv2.VideoCapture = FakeCap

if os.environ.get("TEST_PREVIEWS", "fake") == "fake":
    def tone(freq, seconds=1.5, rate=8000):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(1); w.setframerate(rate)
            w.writeframes(bytes(int(128 + 40 * math.sin(2 * math.pi * freq * n / rate)) for n in range(int(seconds * rate))))
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    TONES = [tone(f) for f in (262, 294, 330, 349, 392, 440, 494)]
    def art(seed):
        hue = seed % 360
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="hsl({hue},45%,38%)"/><circle cx="50" cy="50" r="22" fill="hsl({(hue + 40) % 360},60%,60%)"/></svg>'
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    def fake_search(title, artist):
        h = int(hashlib.md5(f"{title}|{artist}".encode()).hexdigest(), 16)
        if h % 10 < 7:
            return {"preview_url": TONES[h % len(TONES)], "artwork_url": art(h), "itunes_title": title, "itunes_artist": artist}
        return None
    previews.search_itunes = fake_search
    previews.CACHE_PATH = os.path.join(tempfile.mkdtemp(), "previews.csv")
    previews._cache = None

MODULE_LOG = os.environ.get("TEST_MODULE_LOG")
if MODULE_LOG:   # record when the heavy libraries get imported in this server process
    import threading, time as _time
    def _watch():
        last = None
        while True:
            tm = sys.modules.get("textmodel")
            now = tuple(int(m in sys.modules) for m in ("tensorflow", "torch", "transformers")) + ((tm.loads, int(tm.is_loaded())) if tm else (0, 0),)
            if now != last:
                with open(MODULE_LOG, "a") as f: f.write(f"{_time.time():.2f} tensorflow={now[0]} torch={now[1]} transformers={now[2]} loads={now[3][0]} ready={now[3][1]};")
                last = now
            _time.sleep(0.25)
    threading.Thread(target=_watch, daemon=True).start()

sys.argv = ["streamlit", "run", f"{ROOT}/" + os.environ.get("TEST_APP", "app.py"), "--server.headless", "true", "--server.port", os.environ.get("TEST_PORT", "8599"),
            "--browser.gatherUsageStats", "false"] + (["--server.fileWatcherType", os.environ["TEST_WATCHER"]] if os.environ.get("TEST_WATCHER") else [])
from streamlit.web import cli
cli.main()
