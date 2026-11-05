"""Cold start with the background text-model warm-up: page timing, then text timing (after 30 s, or straight away)."""
import os, subprocess, sys, time, urllib.request
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "output"); os.makedirs(S, exist_ok=True); ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
CASE = sys.argv[1]; PORT = {"wait30": "8561", "immediate": "8562"}[CASE]
TEXT = "I miss her so much, everything feels empty and pointless now."
FLAG = r"Running|load_songs|load_face_model|load_text_model|summarise_scan|text_probabilities|webcam_frames|build_queue|build_view|read_faces|textmodel|warm_up|cache_resource|cache_data"
def log(*a): print("RESULT", *a, flush=True)
WATCH = """
window.__flagged = []; window.__seen = {};
const FLAG = new RegExp(%s);
const check = () => { const t = document.body ? document.body.innerText : "";
  const m = t.match(FLAG); if (m) window.__flagged.push([m[0], t.slice(Math.max(0, m.index - 30), m.index + 50)]);
  for (const k of ["Reading your text"]) if (t.includes(k)) window.__seen[k] = (window.__seen[k] || 0) + 1; };
new MutationObserver(check).observe(document, { subtree: true, childList: true, characterData: true }); setInterval(check, 50);
""" % repr(FLAG)
mod_log = f"{S}/modules_warm_{CASE}.txt"; open(mod_log, "w").close()
server = subprocess.Popen([PY, f"{HERE}/fake_camera_server.py"], env={**os.environ, "TEST_PORT": PORT, "TEST_FRAME": "sad", "TEST_PREVIEWS": "real", "TEST_MODULE_LOG": mod_log},
                          stdout=open(f"{S}/server_warm_{CASE}.log", "w"), stderr=subprocess.STDOUT)
def mods():
    return [(float(r.split()[0]), dict(x.split("=") for x in r.split()[1:])) for r in open(mod_log).read().split(";") if r.strip()]
def when(cond):
    return next((t for t, m in mods() if cond(m)), None)
try:
    for _ in range(120):
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read(): break
        except Exception: time.sleep(0.25)
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge", headless=True); ctx = b.new_context(viewport={"width": 1440, "height": 900}); ctx.add_init_script(WATCH); page = ctx.new_page()
        t0 = time.time(); page.goto(f"http://localhost:{PORT}", wait_until="commit")
        page.wait_for_selector(".listening", timeout=120000); t_listen = time.time()
        page.wait_for_selector("iframe", timeout=180000)
        def player(need_text=False):
            for f in page.frames:
                try:
                    if f != page.main_frame and f.evaluate("typeof window.player") == "object" and (not need_text or "Text" in f.locator("#legend").inner_text()): return f
                except Exception: pass
        while player() is None: time.sleep(0.1)
        t_full = time.time()
        log(f"[{CASE}] listening screen {t_listen - t0:.1f}s | full page {t_full - t0:.1f}s after opening")
        if CASE == "wait30":
            time.sleep(30)
            t_ready = when(lambda m: m["ready"] == "1")
            log(f"[{CASE}] warm-up: torch imported {when(lambda m: m['torch'] == '1') - t_full:+.1f}s and model ready {t_ready - t_full:+.1f}s after the page appeared" if t_ready else f"[{CASE}] model not ready after 30 s?!")
        page.get_by_role("button", name="⚙").click(); time.sleep(0.5)
        box = page.get_by_label("Tell me how you feel (optional)"); box.fill(TEXT)
        t_type = time.time(); box.press("Enter")
        while (fr := player(True)) is None:
            time.sleep(0.05)
            if time.time() - t_type > 120: raise RuntimeError("no text result")
        t_text = time.time()
        page.keyboard.press("Escape"); time.sleep(0.3)
        last = mods()[-1][1]
        log(f"[{CASE}] text result {t_text - t_type:.1f}s after pressing Enter | 'Reading your text...' shown: {page.evaluate('window.__seen')} | model loads in this server process: {last['loads']}")
        log(f"[{CASE}] result: {fr.locator('#moodhead').inner_text()!r} | bars {fr.locator('#legend span').all_inner_texts()} | text bar for sad {fr.locator('.erow[data-emotion=sad] .bar.text .pct').inner_text()} | combined sad {fr.locator('.erow[data-emotion=sad] .bar.combined .pct').inner_text()}")
        page.get_by_role("button", name="⚙").click(); time.sleep(0.5)
        box.fill("Finally finished my exams, can't wait for the weekend!"); t2 = time.time(); box.press("Enter")
        while (fr := player(True)) is None or int(fr.locator('.erow[data-emotion=happy] .bar.text .pct').inner_text().rstrip('%')) < 50:
            time.sleep(0.05)
            if time.time() - t2 > 60: break
        log(f"[{CASE}] a second, different sentence: result {time.time() - t2:.1f}s after Enter -> text bar happy {fr.locator('.erow[data-emotion=happy] .bar.text .pct').inner_text()}, page {fr.locator('#moodhead').inner_text()!r} (the sad face outweighs it) | model loads: {mods()[-1][1]['loads']}")
        log(f"[{CASE}] 'Running' or a function name on the page: {len(page.evaluate('window.__flagged'))} time(s) {page.evaluate('window.__flagged')[:2]}")
        b.close()
finally:
    server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
