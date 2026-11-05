"""Headless-Edge test of "Upload a photo instead" in the face tile (the page's only uploader)."""
import io, os, subprocess, sys, time, urllib.request, tempfile
import numpy as np, pandas as pd, cv2
from PIL import Image
from playwright.sync_api import sync_playwright
from pagekit import bar_frame, full_shot, legend, path_frame, player_state, read_card, wait_for_text_bars

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
        log(f"webcam sees no face: {page.locator('[data-testid=stAlert]').first.inner_text()!r} | tile {page.locator('.st-key-facetile').inner_text()[:60]!r} | camera opened {opened()} time(s)")
        def close_popover():
            page.keyboard.press("Escape"); time.sleep(0.3); page.locator(".greeting h1").click(); time.sleep(0.4)
        def upload(name, wait=14):
            page.get_by_role("button", name="Upload a photo instead").click(); time.sleep(0.8)
            page.locator("[data-testid=stPopoverBody] input[type=file]").set_input_files(f"{assets}/{name}")
            time.sleep(wait)
        def describe():
            card = read_card(page); path = path_frame(page)
            return (f"{card['headline']!r} gauge {card['pct']} {card['band']} | path {path.locator('#title').inner_text()!r} | bars {legend(page)} | "
                    f"scan picture {page.locator('.scanimg').evaluate(SIZE_JS)} | note {page.locator('.tile-note').all_inner_texts()} | rows {path.locator('.row').count()}")
        upload("happy.png")
        log(f"uploader label in the tile: {page.locator('[data-testid=stPopoverBody] [data-testid=stFileUploader] label').first.inner_text()!r} | "
            f"note {page.locator('[data-testid=stPopoverBody] [data-testid=stCaptionContainer]').first.inner_text()!r}")
        page.screenshot(path=f"{S}/page_7_tile_upload.png")
        close_popover()
        log(f"after uploading a happy face: {describe()} | camera opened {opened()} time(s)")
        gap = page.evaluate("document.querySelector('.st-key-rescan').getBoundingClientRect().top - document.querySelector('.tile-note').getBoundingClientRect().bottom")
        log(f"space between the photo note and the buttons: {gap:.0f} px")
        full_shot(page, f"{S}/page_7_uploaded_photo.png")
        bar = bar_frame(page); path = path_frame(page); path.locator(".row").nth(0).click(); time.sleep(2.5)
        s1 = player_state(bar)["time"]
        page.get_by_role("button", name="Upload a photo instead").click(); time.sleep(1.5); close_popover()
        s2 = player_state(bar)
        log(f"music keeps playing while the upload popover opens: time {s1:.1f} -> {s2['time']:.1f}, paused {s2['paused']}")
        selected = lambda: page.locator(".choice-card.on h3").all_inner_texts()
        page.locator(".st-key-choice_uplift").scroll_into_view_if_needed(); page.locator(".st-key-choice_uplift").click(); time.sleep(6)
        log(f"chose 'Lift my mood' for the happy photo: selected {selected()} | 'Let MOOD TUNE choose' shown {page.get_by_role('button', name='Let MOOD TUNE choose').count()}")
        upload("sad_sideways.jpg"); close_popover()
        log(f"after a sideways phone-style photo (EXIF rotation): {describe()} | camera opened {opened()} time(s)")
        log(f"a new photo goes back to Auto: selected {selected()} | recommended {page.locator('.choice-card:has(.tag) h3').all_inner_texts()} | "
            f"'Let MOOD TUNE choose' shown {page.get_by_role('button', name='Let MOOD TUNE choose').count()}")
        page.screenshot(path=f"{S}/page_9_uploaded_sideways.png")
        upload("not_a_picture.png", wait=8)
        log(f"after a file that isn't a picture: alerts {page.locator('[data-testid=stAlert]').all_inner_texts()} | page still shows {read_card(page)['headline']!r}")
        close_popover()
        upload("no_face.png"); close_popover()
        log(f"after a photo with no face (and no text): alerts {[a[:80] for a in page.locator('[data-testid=stAlert]').all_inner_texts()]} | camera opened {opened()} time(s)")
        box = page.get_by_label("How was your day?"); box.fill("I miss her so much, everything feels empty and pointless now."); box.press("Enter")
        wait_for_text_bars(page); time.sleep(2)
        log(f"then with a sad sentence: {describe()} | camera opened {opened()} time(s)")
        page.get_by_role("button", name="Tap to scan again").click(); time.sleep(8)
        log(f"after 'Tap to scan again': camera opened {opened()} time(s) | {describe()}")
        log(f"page errors: {errors}")
        b.close()
finally:
    server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
after = image_files()
new = [q for q, m in after.items() if q not in before or m > before[q] + 1]
log(f"image files created or changed on disk by the app during the run: {new}")
