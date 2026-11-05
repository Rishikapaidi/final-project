"""Headless-Edge checks of the "Songs in" language selector, with real cached previews."""
import os, subprocess, sys, time, urllib.request, json
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "output"); os.makedirs(S, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCENARIOS = [  # name, frame, language, screenshot
    ("hindi_sad", "sad", "Hindi", True),
    ("hindi_happy", "happy", "Hindi", False),
    ("tamil_sad", "sad", "Tamil", True),
    ("telugu_sad", "sad", "Telugu", True),
    ("malayalam_sad", "sad", "Malayalam", False),
    ("korean_sad", "sad", "Korean", True),
]
ONLY = sys.argv[1:]
def log(*a): print("RESULT", *a, flush=True)

import pandas as pd
songs = pd.read_csv(f"{ROOT}/data/songs/songs_clean.csv", keep_default_na=False)
lang_of = dict(zip(songs.lastfm_url, songs.language))

def player_frame(page, not_tracks=None):
    """The player iframe; with not_tracks, wait for one showing a different track list."""
    for _ in range(450):
        for f in page.frames:
            try:
                if f != page.main_frame and f.evaluate("typeof window.player") == "object":
                    urls = f.evaluate("DATA.tracks.map(t => t.lastfm)")
                    if not_tracks is None or urls != not_tracks: return f
            except Exception: pass
        time.sleep(0.2)
    raise RuntimeError("player frame not found")

def choose_language(page, language):
    page.get_by_role("button", name="⚙").click(); time.sleep(0.8)
    page.locator("[data-testid=stPopoverBody] [data-testid=stSelectbox]").click(); time.sleep(0.5)
    page.get_by_role("option", name=language, exact=True).click(); time.sleep(0.5)
    page.keyboard.press("Escape"); time.sleep(0.4)

with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge", headless=True)
    for n, (name, frame, language, shot) in enumerate(SCENARIOS):
        if ONLY and not any(o in name for o in ONLY): continue
        port = str(8610 + n)
        cam_log = f"{S}/cam_{name}.log"
        if os.path.exists(cam_log): os.remove(cam_log)
        server = subprocess.Popen([PY, f"{HERE}/fake_camera_server.py"],
                                  env={**os.environ, "TEST_PORT": port, "TEST_FRAME": frame, "TEST_PREVIEWS": "real", "TEST_CAM_LOG": cam_log},
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
            page.wait_for_selector("iframe", timeout=180000)
            fr = player_frame(page); time.sleep(1.5)
            state = lambda f: f.evaluate("(() => { const s = window.player.state(); return {current: s.current, paused: s.paused, time: +s.time.toFixed(1)}; })()")
            english = fr.evaluate("DATA.tracks.map(t => t.lastfm)")
            fr.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); playing = state(fr)
            log(f"English first: {len(english)} rows, languages {pd.Series([lang_of[u] for u in english]).value_counts().to_dict()} | playing {playing}")

            choose_language(page, language)
            fr = player_frame(page, not_tracks=english); time.sleep(1.5)
            tracks = fr.evaluate("DATA.tracks")
            time.sleep(1)
            players = [f for f in page.frames if f != page.main_frame and f.evaluate("typeof window.player") == "object"]
            sounding = sum(not f.evaluate("window.player.audio.paused") for f in players)
            log(f"after the switch: players on the page {len(players)} | players still playing {sounding} | script {page.locator('[data-testid=stApp]').get_attribute('data-test-script-state')}")
            cam = open(cam_log).read().count("open") if os.path.exists(cam_log) else "?"
            langs = pd.Series([lang_of[t["lastfm"]] for t in tracks]).value_counts().to_dict()
            log(f"after choosing {language}: list rebuilt True | rows {fr.locator('.row').count()} | languages {langs} | only {language}: {set(langs) == {language}} | camera opened {cam} time(s)")
            log(f"mood card: {fr.locator('#moodemoji').inner_text()} {fr.locator('#moodhead').inner_text()!r} | mode: {fr.locator('#moodmode').inner_text()!r}")
            if fr.locator('#moodnote').inner_text(): log(f"   note: {fr.locator('#moodnote').inner_text()!r}")
            bars = fr.evaluate("DATA.view.bars")
            log(f"bars: groups {[g['key'] for g in bars['groups']]} | top {bars['emotions'][:3]} | legend {fr.locator('#legend span').all_inner_texts()}")
            log(f"top picks: {fr.locator('#qsub').inner_text()!r} | play pills {fr.locator('.row .pill').count()} | sections {fr.locator('.section').all_inner_texts()}")
            labels = pd.Series(fr.locator("a.lastfm").all_inner_texts()).value_counts().to_dict()
            hrefs = fr.locator("a.lastfm").evaluate_all("as => as.map(a => a.href)")
            log(f"links: {labels} | all open.spotify.com: {all(h.startswith('https://open.spotify.com/') for h in hrefs)}")
            fr.evaluate("document.querySelectorAll('.row img.art').forEach(i => i.loading = 'eager')"); time.sleep(3)
            art = fr.locator(".row img.art").evaluate_all("is => is.map(i => [i.complete && i.naturalWidth > 0, new URL(i.src).hostname])")
            hosts = pd.Series([h for _, h in art]).value_counts().to_dict()
            log(f"artwork: {len(art)} images, {sum(ok for ok, _ in art)} loaded | no-art placeholders {fr.locator('.row .noart').count()} | hosts {hosts}")

            fr.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); a = state(fr)
            log(f"play pill on row 1: {a} | pill says {fr.locator('.row').nth(0).locator('.pill').inner_text()!r} | bar: {fr.locator('#nowtitle').inner_text()!r}")
            fr.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1.5"); time.sleep(4.5); b = state(fr)
            log(f"near the end of track 1 -> {b} (next song started on its own: {b['current'] == 1 and not b['paused']})")
            if shot:
                fr.evaluate("window.player.audio.pause()"); time.sleep(0.3)
                page.screenshot(path=f"{S}/lang_{name}.png")
            log(f"page errors: {errors}")
            page.close()
        finally:
            server.terminate(); subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    browser.close()
