"""Headless-Edge test of "Upload a photo instead"."""
import io, os, subprocess, sys, time, urllib.request, tempfile
import numpy as np, pandas as pd, cv2
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "output"); os.makedirs(S, exist_ok=True); ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable; PORT = "8586"
def log(*a): print("RESULT", *a, flush=True)
SIZE_JS = "e => e.naturalWidth + 'x' + e.naturalHeight"

# --- test inputs (these are the test's own files, made before the app starts)
assets = os.path.join(S, "upload_assets"); os.makedirs(assets, exist_ok=True)
p = pd.read_parquet(f"{ROOT}/data/face_emotion/raw/privateTest-0000.parquet")
fz = np.load(f"{ROOT}/data/fusion/face_test.npz"); fprobs, flabels = fz["probs"], fz["labels"].astype(str)
EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def face_canvas(emotion, size=(960, 1280)):
    k = EMO.index(emotion)
    for i in np.argsort(-fprobs[:, k]):
        if flabels[i] != emotion: continue
        img = np.asarray(Image.open(io.BytesIO(p.image.iloc[i]["bytes"])).convert("L"))
        canvas = np.full(size, 140, np.uint8); canvas[280:760, 400:880] = cv2.resize(img, (480, 480), interpolation=cv2.INTER_CUBIC)
        if len(haar.detectMultiScale(canvas, scaleFactor=1.3, minNeighbors=5)) == 1: return canvas
Image.fromarray(face_canvas("happy")).convert("RGB").save(f"{assets}/happy.png")
sad = Image.fromarray(face_canvas("sad")).convert("RGB")
exif = Image.Exif(); exif[0x0112] = 6            # "rotate 90 degrees clockwise to display"
sad.rotate(90, expand=True).save(f"{assets}/sad_sideways.jpg", exif=exif, quality=92)   # stored lying on its side
Image.fromarray(np.full((600, 800, 3), 140, np.uint8)).save(f"{assets}/no_face.png")
open(f"{assets}/not_a_picture.png", "wb").write(b"this is not an image")

def image_files():
    found = {}
    for base in [ROOT, tempfile.gettempdir()]:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("venv", ".git") and os.path.join(dirpath, d) != S and not d.startswith("playwright")]
            if dirpath.count(os.sep) - base.count(os.sep) > 4: dirnames[:] = []
            for f in filenames:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")):
                    try: found[os.path.join(dirpath, f)] = os.path.getmtime(os.path.join(dirpath, f))
                    except OSError: pass
    return found

cam_log = os.path.join(S, "cam_log.txt"); open(cam_log, "w").close()
opened = lambda: open(cam_log).read().count("opened")
before = image_files()
server = subprocess.Popen([PY, f"{HERE}/fake_camera_server.py"], env={**os.environ, "TEST_PORT": PORT, "TEST_FRAME": "blank", "TEST_PREVIEWS": "real", "TEST_CAM_LOG": cam_log},
                          stdout=open(f"{S}/server_upload.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(90):
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read(): break
        except Exception: time.sleep(1)
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True); page = b.new_page(viewport={"width": 1440, "height": 900})
        errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}"); page.wait_for_selector("[data-testid=stAlert]", timeout=180000)
        log(f"webcam sees no face: {page.locator('[data-testid=stAlert]').first.inner_text()!r} | camera opened {opened()} time(s)")
        def frame_():
            for _ in range(300):
                for f in page.frames:
                    try:
                        if f != page.main_frame and f.evaluate("typeof window.player") == "object": return f
                    except Exception: pass
                time.sleep(0.2)
        def upload(name, wait=14):
            page.get_by_role("button", name="⚙").click(); time.sleep(0.8)
            page.locator("[data-testid=stPopoverBody] input[type=file]").set_input_files(f"{assets}/{name}")
            time.sleep(wait)
        def describe():
            fr = frame_()
            return (f"{fr.locator('#moodhead').inner_text()!r} | mode {fr.locator('#moodmode').inner_text()!r} | bars {fr.locator('#legend span').all_inner_texts()} | "
                    f"scan picture {fr.locator('#scanimg').evaluate(SIZE_JS)} | note {fr.locator('#scanbody .small').all_inner_texts()} | rows {fr.locator('.row').count()}")
        upload("happy.png")
        log(f"uploader label in settings: {page.locator('[data-testid=stPopoverBody] [data-testid=stFileUploader] label').first.inner_text()!r}")
        page.screenshot(path=f"{S}/page_7_settings_upload.png")
        page.keyboard.press("Escape"); page.locator(".app-tagline").click(); time.sleep(0.5)
        log(f"after uploading a happy face: {describe()} | camera opened {opened()} time(s)")
        page.screenshot(path=f"{S}/page_8_uploaded_photo.png")
        fr = frame_(); fr.locator(".row").nth(0).click(); time.sleep(2.5)
        s1 = fr.evaluate("window.player.state().time")
        page.get_by_role("button", name="⚙").click(); time.sleep(1.5); page.keyboard.press("Escape"); time.sleep(0.5)
        s2 = fr.evaluate("window.player.state()")
        log(f"music keeps playing with the uploader in settings: time {s1:.1f} -> {s2['time']:.1f}, paused {s2['paused']}")
        upload("sad_sideways.jpg"); page.keyboard.press("Escape"); page.locator(".app-tagline").click(); time.sleep(0.5)
        log(f"after a sideways phone-style photo (EXIF rotation): {describe()} | camera opened {opened()} time(s)")
        page.screenshot(path=f"{S}/page_9_uploaded_sideways.png")
        upload("not_a_picture.png", wait=8)
        log(f"after a file that isn't a picture: alerts {page.locator('[data-testid=stAlert]').all_inner_texts()} | page still shows {frame_().locator('#moodhead').inner_text()!r}")
        page.keyboard.press("Escape"); time.sleep(0.3)
        upload("no_face.png"); page.keyboard.press("Escape"); time.sleep(0.5)
        log(f"after a photo with no face (and no text): alerts {[a[:70] for a in page.locator('[data-testid=stAlert]').all_inner_texts()]} | camera opened {opened()} time(s)")
        page.get_by_role("button", name="⚙").click(); time.sleep(0.6)
        box = page.get_by_label("Tell me how you feel (optional)"); box.fill("I miss her so much, everything feels empty and pointless now."); box.press("Enter"); time.sleep(6)
        page.keyboard.press("Escape"); page.locator(".app-tagline").click(); time.sleep(0.5)
        log(f"then with a sad sentence: {describe()} | camera opened {opened()} time(s)")
        page.get_by_role("button", name="Re-scan").click(); time.sleep(6)
        log(f"after Re-scan: camera opened {opened()} time(s) | {describe()}")
        log(f"page errors: {errors}")
        b.close()
finally:
    server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
after = image_files()
new = [q for q, m in after.items() if q not in before or m > before[q] + 1]
log(f"image files created or changed on disk by the app during the run: {new}")
