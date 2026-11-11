"""Cold-start timing and 'no function names on the page' check, in headless Edge."""
import os, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
from pagekit import legend, read_card

HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "output"); os.makedirs(S, exist_ok=True); ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
RUN = sys.argv[1] if len(sys.argv) > 1 else "1"; PORT = str(8570 + int(RUN))
TEXT = "I miss her so much, everything feels empty and pointless now."
FLAG = r"Running|load_songs|load_face_model|load_text_model|summarise_scan|text_probabilities|webcam_frames|build_queue|build_view|read_faces|cache_resource|cache_data"
def log(*a): print("RESULT", *a, flush=True)

# watch every change to the page: anything matching FLAG, and when the friendly messages show up
WATCH = """
window.__flagged = []; window.__seen = {};
const FLAG = new RegExp(%s);
const check = () => {
  const t = document.body ? document.body.innerText : "";
  const m = t.match(FLAG); if (m) window.__flagged.push([Date.now(), m[0], t.slice(Math.max(0, m.index - 40), m.index + 60)]);
  for (const key of ["Listening to your mood", "Reading your text", "MOOD TUNE"]) if (t.includes(key) && !window.__seen[key]) window.__seen[key] = Date.now();
};
new MutationObserver(check).observe(document, { subtree: true, childList: true, characterData: true });
setInterval(check, 50);
""" % repr(FLAG)

mod_log = f"{S}/modules_{RUN}.txt"; open(mod_log, "w").close()
t_server = time.time()
server = subprocess.Popen([PY, f"{HERE}/fake_camera_server.py"], env={**os.environ, "TEST_APP": os.environ.get("TEST_APP", "app.py"), "TEST_PORT": PORT, "TEST_FRAME": "sad", "TEST_PREVIEWS": "real", "TEST_MODULE_LOG": mod_log},
                          stdout=open(f"{S}/server_cold_{RUN}.log", "w"), stderr=subprocess.STDOUT)
def modules():
    rows = [r.split() for r in open(mod_log).read().split(";") if r.strip()]
    return [(float(r[0]), dict(x.split("=") for x in r[1:])) for r in rows]
def first_import(name):
    return next((t for t, m in modules() if m[name] == "1"), None)
try:
    for _ in range(120):
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read(): break
        except Exception: time.sleep(0.25)
    t_ready = time.time()
    log(f"run {RUN}: server process ready to accept the browser {t_ready - t_server:.1f}s after it was started (fresh process, nothing cached)")
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True); ctx = b.new_context(viewport={"width": 1440, "height": 900})
        ctx.add_init_script(WATCH); page = ctx.new_page()
        t0 = time.time() * 1000
        page.goto(f"http://localhost:{PORT}", wait_until="commit")
        page.wait_for_selector(".st-key-topleft iframe", timeout=120000); t_head = time.time() * 1000
        page.wait_for_selector(".listening", timeout=120000); t_listen = time.time() * 1000
        page.screenshot(path=f"{S}/cold_{RUN}_listening.png")
        page.wait_for_selector(".st-key-playerbar iframe", timeout=180000)
        fr = None
        while fr is None:
            for f in page.frames:
                try:
                    if f != page.main_frame and f.evaluate("typeof window.playlist") == "object": fr = f
                except Exception: pass
            time.sleep(0.1)
        t_full = time.time() * 1000
        seen = page.evaluate("window.__seen")
        tf = first_import("tensorflow")
        log(f"header (MOOD TUNE) visible:          {(t_head - t0) / 1000:5.1f}s after opening the page")
        log(f"'Listening to your mood...' visible: {(t_listen - t0) / 1000:5.1f}s")
        log(f"full page with the player:           {(t_full - t0) / 1000:5.1f}s  ({read_card(page)['headline']!r}, bars {legend(page)}, {fr.locator('.row').count()} songs)")
        log(f"tensorflow imported {tf - t0 / 1000:+.1f}s after opening the page, i.e. {'AFTER' if tf * 1000 > t_listen else 'BEFORE'} the listening screen appeared" if tf else "tensorflow never imported?!")
        log(f"before any text: torch imported {first_import('torch') is not None}, transformers imported {first_import('transformers') is not None}")
        page.screenshot(path=f"{S}/cold_{RUN}_loaded.png")

        box = page.get_by_label("How was your day?"); box.fill(TEXT)
        t_type = time.time(); box.press("Enter")
        shot = False
        for _ in range(600):
            if not shot and page.locator(".reading").count():
                page.screenshot(path=f"{S}/cold_{RUN}_reading.png"); shot = True
            try:
                if "Text" in legend(page): break
            except Exception: pass
            time.sleep(0.1)
        t_text = time.time()
        seen = page.evaluate("window.__seen")
        torch_t, tr_t = first_import("torch"), first_import("transformers")
        log(f"after typing: text result shown {t_text - t_type:.1f}s after pressing Enter | 'Reading your text...' appeared: {'Reading your text' in seen} ({(seen['Reading your text'] / 1000 - t_type):+.1f}s after Enter)" if "Reading your text" in seen else
            f"after typing: text result shown {t_text - t_type:.1f}s after pressing Enter | 'Reading your text...' appeared: False")
        log(f"   torch imported {torch_t - t_type:+.1f}s and transformers {tr_t - t_type:+.1f}s relative to pressing Enter" if torch_t and tr_t else f"   torch {torch_t} transformers {tr_t}")
        card = read_card(page)
        log(f"   result: {card['headline']!r} | gauge {card['pct']} low | bars {legend(page)} | caption {card['caption']!r}")
        # same text again must not show the message (read once per text)
        n_reading = page.evaluate("window.__seen['Reading your text'] || 0")
        flagged = page.evaluate("window.__flagged")
        log(f"'Running' or a function name seen on the page at any point: {len(flagged)} time(s) {flagged[:3]}")
        b.close()
finally:
    server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
