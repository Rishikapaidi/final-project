"""Headless-Edge checks of the "Songs in" language selector, with real cached previews."""
import os, sys, time
import pandas as pd
from playwright.sync_api import sync_playwright
from pagekit import tap_scan, OUT, ROOT, bar_frame, describe_card, full_shot, log, path_frame, player_state, players, read_card, start_server, stop

SCENARIOS = [  # name, frame, language, screenshot
    ("hindi_sad", "sad", "Hindi", "8_songs_in_hindi"),
    ("hindi_happy", "happy", "Hindi", None),
    ("tamil_sad", "sad", "Tamil", "9_songs_in_tamil"),
    ("telugu_sad", "sad", "Telugu", "11_songs_in_telugu_resonance_only"),
    ("malayalam_sad", "sad", "Malayalam", None),
    ("korean_sad", "sad", "Korean", "10_songs_in_korean"),
]
ONLY = sys.argv[1:]
songs = pd.read_csv(os.path.join(ROOT, "data", "songs", "songs_clean.csv"), keep_default_na=False)
lang_of = dict(zip(songs.lastfm_url, songs.language))


def choose_language(page, language):
    page.evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)"); time.sleep(0.8)  # the list closes if the page scrolls
    page.get_by_role("combobox", name="Songs in").click(); time.sleep(0.5)
    page.get_by_role("option", name=language, exact=True).click(); time.sleep(0.5)


with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge", headless=True)
    for n, (name, frame, language, shot) in enumerate(SCENARIOS):
        if ONLY and not any(o in name for o in ONLY): continue
        cam_log = os.path.join(OUT, f"cam_{name}.log")
        if os.path.exists(cam_log): os.remove(cam_log)
        server = start_server(8610 + n, frame, name, TEST_CAM_LOG=cam_log)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://localhost:{8610 + n}"); tap_scan(page)
            log(f"=== {name}")
            page.wait_for_selector(".st-key-playerbar iframe", timeout=180000)
            bar, path = bar_frame(page), path_frame(page); time.sleep(1.5)
            english = path.evaluate("DATA.tracks.map(t => t.lastfm)")
            path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); playing = player_state(bar)
            log(f"English first: {len(english)} rows, languages {pd.Series([lang_of[u] for u in english]).value_counts().to_dict()} | playing {playing}")

            choose_language(page, language)
            path = path_frame(page, not_tracks=english); time.sleep(1.5); bar = bar_frame(page)
            tracks = path.evaluate("DATA.tracks")
            time.sleep(1)
            count, sounding = players(page)
            log(f"after the switch: players on the page {count} | players still playing {sounding} | script {page.locator('[data-testid=stApp]').get_attribute('data-test-script-state')}")
            cam = open(cam_log).read().count("open") if os.path.exists(cam_log) else "?"
            langs = pd.Series([lang_of[t["lastfm"]] for t in tracks]).value_counts().to_dict()
            log(f"after choosing {language}: list rebuilt True | rows {path.locator('.row').count()} | languages {langs} | only {language}: {set(langs) == {language}} | camera opened {cam} time(s)")
            log(f"mood card: {describe_card(read_card(page))}")
            log(f"choice cards: selected {page.locator('.choice-card.on h3').all_inner_texts()} | note on 'Lift my mood' {page.locator('.choice-card .fallback').all_inner_texts()}")
            log(f"path card: {path.locator('#title').inner_text()!r} / {path.locator('#qsub').inner_text()!r} | sections {path.locator('.section').all_inner_texts()}")
            if path.locator("#note").is_visible(): log(f"   note: {path.locator('#note').inner_text()!r}")
            labels = pd.Series(path.evaluate("DATA.tracks.map(t => t.link_label)")).value_counts().to_dict()
            hrefs = path.evaluate("DATA.tracks.map(t => t.lastfm)")
            path.locator(".row").nth(0).locator(".more").click(); time.sleep(0.3)
            menu_link = path.locator("#menu a.lastfm")
            log(f"links: {labels} | all open.spotify.com: {all(h.startswith('https://open.spotify.com/') for h in hrefs)} "
                f"| row 1's menu: {menu_link.inner_text()!r} -> {menu_link.get_attribute('href') == hrefs[0]}")
            page.keyboard.press("Escape"); time.sleep(0.2)
            path.evaluate("document.querySelectorAll('.row img.art').forEach(i => i.loading = 'eager')"); time.sleep(3)
            art = path.locator(".row img.art").evaluate_all("is => is.map(i => [i.complete && i.naturalWidth > 0, new URL(i.src).hostname])")
            hosts = pd.Series([h for _, h in art]).value_counts().to_dict()
            log(f"artwork: {len(art)} images, {sum(ok for ok, _ in art)} loaded | no-art placeholders {path.locator('.row .noart').count()} | hosts {hosts}")

            path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); a = player_state(bar)
            log(f"play pill on row 1: {a} | pill says {path.locator('.row').nth(0).locator('.pill').inner_text()!r} | bar: {bar.locator('#nowtitle').inner_text()!r}")
            bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1.5"); time.sleep(4.5); b = player_state(bar)
            log(f"near the end of track 1 -> {b} (next song started on its own: {b['current'] == 1 and not b['paused']})")
            if shot:
                bar.evaluate("window.player.audio.pause()"); time.sleep(0.3)
                full_shot(page, os.path.join(OUT, f"page_{shot}.png"))
            log(f"page errors: {errors}")
            page.close()
        finally:
            stop(server)
    browser.close()
