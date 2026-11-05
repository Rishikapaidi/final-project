"""Headless-Edge checks and screenshots of the new page layout, with real cached previews."""
import os, subprocess, sys, time, urllib.request, tempfile, glob
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "output"); os.makedirs(S, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SAD_TEXT = "I miss her so much, everything feels empty and pointless now."
SCENARIOS = [  # name, frame, text, extra playback checks
    ("1_sad_face_uplift", "sad", "", True),
    ("2_happy_face_resonance", "happy", "", False),
    ("3_mixed_happy_face_sad_text", "mixed", SAD_TEXT, False),
    ("4_no_face_text_only", "blank", SAD_TEXT, False),
]
ONLY = sys.argv[1:]  # optional scenario name filter
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
def log(*a): print("RESULT", *a, flush=True)

def image_files():
    found = {}
    for base in [ROOT, tempfile.gettempdir()]:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("venv", ".git", "node_modules") and os.path.join(dirpath, d) != S and not d.startswith("playwright")]
            if dirpath.count(os.sep) - base.count(os.sep) > 4: dirnames[:] = []
            for f in filenames:
                if f.lower().endswith(IMAGE_EXT):
                    try: found[os.path.join(dirpath, f)] = os.path.getmtime(os.path.join(dirpath, f))
                    except OSError: pass
    return found

def player_frame(page):
    t_start = time.time()
    for n in range(450):   # up to 90 s: with no face, the first sentence also loads the text model
        seen = []
        for f in page.frames:
            try:
                kind = "main" if f == page.main_frame else f.evaluate("typeof window.player")
                seen.append(kind)
                if kind == "object":
                    if n > 25: print(f"RESULT    (player found after {time.time() - t_start:.1f}s)", flush=True)
                    return f
            except Exception as e: seen.append("ERR " + type(e).__name__ + " " + str(e)[:80])
        if n % 50 == 0: print(f"RESULT    waiting {time.time() - t_start:4.1f}s: frames {seen} | iframes {page.locator('iframe').count()} | state {page.locator('[data-testid=stApp]').get_attribute('data-test-script-state')}", flush=True)
        time.sleep(0.2)
    raise RuntimeError("player frame not found")

before = image_files(); started = time.time()
with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge", headless=True)
    for n, (name, frame, text, playback) in enumerate(SCENARIOS):
        if ONLY and not any(o in name for o in ONLY): continue
        port = str(8590 + n)
        server = subprocess.Popen([PY, f"{HERE}/fake_camera_server.py"], env={**os.environ, "TEST_PORT": port, "TEST_FRAME": frame, "TEST_PREVIEWS": "real"},
                                  stdout=open(f"{S}/server_{name}.log", "w"), stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                try:
                    if b"ok" in urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read(): break
                except Exception: time.sleep(1)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://localhost:{port}")
            log(f"=== {name}")
            if frame == "blank":
                page.wait_for_selector("[data-testid=stAlert]", timeout=180000)
                log("before any text:", page.locator("[data-testid=stAlert]").first.inner_text()[:90])
            else:
                page.wait_for_selector("iframe", timeout=180000)
            log(f"header: {page.locator('.app-name').inner_text()!r} | tagline: {page.locator('.app-tagline').inner_text()!r} | logo svg: {page.locator('.app-head svg.logo').count()} | tab title: {page.title()!r}")
            if text:
                page.get_by_role("button", name="⚙").click(); time.sleep(0.6)
                box = page.get_by_label("Tell me how you feel (optional)")
                box.fill(text); box.press("Enter"); time.sleep(5); page.keyboard.press("Escape"); time.sleep(0.5)
            fr = player_frame(page)
            if text:
                # The first sentence can also load the text model: wait up to 60 s for a player showing Text bars.
                t0 = time.time()
                while time.time() - t0 < 60:
                    fr = player_frame(page)
                    try:
                        if any(g["key"] == "text" for g in fr.evaluate("DATA.view.bars.groups")): break
                    except Exception: pass
                    time.sleep(0.5)
                log(f"   text bars appeared after {time.time() - t0:.1f}s" if any(g["key"] == "text" for g in fr.evaluate("DATA.view.bars.groups")) else "   NO text bars after 60 s")
            time.sleep(1.5)
            view = fr.evaluate("DATA.view")
            card, bars = view["card"], view["bars"]
            combined = dict(zip(bars["emotions"], next(g["values"] for g in bars["groups"] if g["key"] == "combined")))
            top, second = bars["emotions"][0], bars["emotions"][1]
            expected_head = f"Mostly {top}" if combined[top] >= 0.5 else f"Mixed: {top} and {second}"
            log(f"mood card: {fr.locator('#moodemoji').inner_text()} {fr.locator('#moodhead').inner_text()!r} | mode: {fr.locator('#moodmode').inner_text()!r} | colour {card['colour']}")
            log(f"   message: {fr.locator('#moodmsg').inner_text()!r}")
            if fr.locator('#moodnote').inner_text(): log(f"   note: {fr.locator('#moodnote').inner_text()!r}")
            log(f"   headline matches the rule for combined top {top} {combined[top]:.3f}: {fr.locator('#moodhead').inner_text() == expected_head} (expected {expected_head!r})")
            groups = [g["key"] for g in bars["groups"]]
            shown = {g: [fr.locator(f'.erow[data-emotion="{e}"] .bar.{g} .pct').inner_text() for e in bars["emotions"]] for g in groups}
            want = {g["key"]: [f"{round(v * 100)}%" for v in g["values"]] for g in bars["groups"]}
            widths_ok = all(abs(float(fr.locator(f'.erow[data-emotion="{e}"] .bar.combined .fill').evaluate("e => e.style.width").rstrip('%')) - combined[e] * 100) < 0.06 for e in bars["emotions"])
            log(f"bars: groups {groups} | legend {fr.locator('#legend span').all_inner_texts()} | caption {fr.locator('#barcap').inner_text()!r}")
            log(f"   combined: " + ", ".join(f"{e} {p}" for e, p in zip(bars['emotions'], shown['combined'])) + f" | sums to {sum(combined.values()):.3f}")
            for g in groups:
                if g != "combined": log(f"   {g:8}: " + ", ".join(f"{e} {p}" for e, p in zip(bars['emotions'], shown[g])))
            log(f"   bar labels match the probabilities: {shown == want} | combined bar widths match: {widths_ok} | emotions sorted by combined: {list(combined.values()) == sorted(combined.values(), reverse=True)}")
            has_img = fr.locator("#scanimg").count()
            if has_img:
                size = fr.locator("#scanimg").evaluate("e => [e.naturalWidth, e.naturalHeight, e.src.slice(0, 23), Math.round(e.src.length / 1024)]")
                log(f"scan picture: {size[0]}x{size[1]}, src starts {size[2]!r}, {size[3]} KB in the page | note: {fr.locator('#scanbody .small').all_inner_texts()}")
            else:
                log(f"scan picture: none | {fr.locator('#scanbody').inner_text()!r}")
            rows = fr.locator(".row").count()
            log(f"top picks: {fr.locator('.picks-head h2').inner_text()!r} / {fr.locator('#qsub').inner_text()!r} | rows {rows} | play pills {fr.locator('.row .pill').count()} | last.fm links {fr.locator('a.lastfm').count()} | sections {fr.locator('.section').all_inner_texts()} | journey chart {fr.locator('details.journey').count()}")
            page.screenshot(path=f"{S}/page_{name}.png")

            if playback:
                state = lambda: fr.evaluate("(() => { const s = window.player.state(); return {current: s.current, paused: s.paused, time: +s.time.toFixed(1)}; })()")
                fr.locator(".row").nth(2).locator(".meta").click(); time.sleep(2.5); a = state()
                log(f"click on row 3: {a} | pill now says {fr.locator('.row').nth(2).locator('.pill').inner_text()!r} | bar: {fr.locator('#nowtitle').inner_text()!r}")
                fr.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); b = state()
                log(f"click on row 1's Play pill: {b} | highlighted {fr.locator('.row.playing .title').inner_text()!r}")
                fr.locator(".row").nth(0).locator(".pill").click(); time.sleep(0.5); c = state()
                fr.locator(".row").nth(0).locator(".pill").click(); time.sleep(1.0)
                log(f"same pill again pauses: {c['paused']} | and again resumes: {not state()['paused']}")
                page.get_by_role("button", name="⚙").click(); time.sleep(1.5); d1 = state(); time.sleep(2); d2 = state()
                page.screenshot(path=f"{S}/page_5_settings_open.png")
                log(f"settings open: still playing {not d2['paused']}, same track {d2['current'] == d1['current']}, time {d1['time']} -> {d2['time']} | options {page.locator('[data-testid=stPopoverBody] label').all_inner_texts()[:6]}")
                page.keyboard.press("Escape"); time.sleep(0.4)
                fr.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1.5"); time.sleep(4.5); e = state()
                log(f"near the end of track 1 -> {e} (next song started on its own: {e['current'] == 1 and not e['paused']})")
                fr.locator("details.journey summary").click(); time.sleep(0.5)
                page.screenshot(path=f"{S}/page_6_playing_with_journey.png")
            log(f"page errors: {errors}")
            page.close()
        finally:
            server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    browser.close()

after = image_files()
new = [p for p, m in after.items() if p not in before or m > before[p] + 1]
log(f"=== image files created or changed on disk during the run (outside the test's own screenshots): {new}")
