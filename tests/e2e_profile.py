"""Headless-Edge checks of the avatar's menu and Settings (sad face, real cached previews).

The menu: the name and "Local profile" (the pencil changes the name),
Settings, Liked songs and History (open the Library), Keyboard shortcuts,
Privacy, About, Send feedback (the project's GitHub Issues, in a new tab; no
email address anywhere) and Reset profile. Settings: Theme (Light / Dark)
switched while a song plays (no app rerun, the same queue, every component
following), Default language and Default mode (kept in the prefs cookie;
after a reload "Songs in" and the chosen card start with them, and a new scan
goes back to the default mode), and the options from before. Reset profile,
after its confirm, clears everything kept in the browser and the cookie.
Runs in either theme (TEST_THEME=dark: it starts dark and switches to light
and back). Saves screenshots of the menu, Settings and Privacy.
"""
import os, sys, time
from playwright.sync_api import sync_playwright
from pagekit import (tap_scan, OUT, THEME, ROOT, bar_frame, count_reruns, image_files, log, named_frame, new_images, path_frame, reruns,
                     start_server, stop, theme_report)

sys.path.insert(0, ROOT)
from shell import FEEDBACK_URL  # noqa: E402

PORT = 8626
OTHER = "light" if THEME == "dark" else "dark"


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, f"page_{name}.png"))


def components(page):
    return [named_frame(page, n) for n in ("topbar", "actions", "navlists", "quickpicks", "playlist", "player")]


def cookie(page):
    return next((c["value"] for c in page.context.cookies() if c["name"] == "moodtune_prefs"), None)


before = image_files()
cam_log = os.path.join(OUT, "cam_profile.log")
if os.path.exists(cam_log): os.remove(cam_log)
server = start_server(PORT, "sad", "profile", TEST_CAM_LOG=cam_log)
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}"); tap_scan(page); page.wait_for_selector(".st-key-path iframe", timeout=180000)
        bar, path = bar_frame(page), path_frame(page); time.sleep(1.5)
        prof = named_frame(page, "actions"); state = lambda: bar.evaluate("window.player.state()")
        log(f"theme at the start ({THEME}): {theme_report(page, *components(page))}")

        log("== THE MENU")
        prof.locator("#avatar").click(); time.sleep(0.4)
        items = [t.strip() for t in prof.locator("#pmenu .pi").all_inner_texts()]
        log(f"items {items} | header {prof.locator('.who').inner_text().split(chr(10))!r} | covers the window {page.locator('.st-key-topactions iframe').bounding_box()['height']}px")
        fb = prof.locator("#m-feedback")
        log(f"Send feedback: href {fb.get_attribute('href')!r} = FEEDBACK_URL {fb.get_attribute('href') == FEEDBACK_URL} | target {fb.get_attribute('target')} | rel {fb.get_attribute('rel')!r} "
            f"| 'mailto' on the page or in a component {any('mailto' in f.content() for f in page.frames)}")
        shot(page, "24_profile_menu")
        prof.locator("#m-name").click(); time.sleep(0.3); page.locator("#mtd-input").fill("Rishika"); page.locator("#moodtune-dialog .mtd-ok").click(); time.sleep(0.4)
        prof.locator("#avatar").click(); time.sleep(0.3)
        log(f"the pencil: name 'Rishika' -> header {prof.locator('.who').inner_text().split(chr(10))!r} | avatar letter {prof.locator('#avatar').inner_text()!r}")
        page.keyboard.press("Escape"); time.sleep(0.3)

        log("== SETTINGS: THEME")
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2)
        count_reruns(page); runs0, t0, queue_id = reruns(page), state()["time"], bar.evaluate("DATA.id")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-settings").click(); time.sleep(0.5)
        groups = prof.locator("#v-settings .group").all_inner_texts()
        options = [t.split("\n")[0] for t in prof.locator("#v-settings .setting b").all_inner_texts()]
        log(f"Settings: groups {groups} | options {options} | language choices {prof.locator('#s-language option').all_inner_texts()} "
            f"| modes {prof.locator('input[name=mode]').evaluate_all('xs => xs.map(x => x.value)')} | links {[t.split(chr(10))[0] for t in prof.locator('#v-settings .link-row').all_inner_texts()]}")
        shot(page, "25_settings_theme")
        prof.locator(".seg label", has_text=OTHER.capitalize()).click(); time.sleep(1)
        s = state()
        log(f"Theme -> {OTHER} while playing: {theme_report(page, *components(page))} | app reruns {reruns(page) - runs0} | same queue {bar.evaluate('DATA.id') == queue_id} "
            f"| playing {not s['paused']} ({t0:.1f}s -> {s['time']:.1f}s) | cookie {cookie(page)!r}")
        prof.locator(".seg label", has_text=THEME.capitalize()).click(); time.sleep(1)
        log(f"Theme -> back to {THEME}: {theme_report(page, *components(page))} | still playing {not state()['paused']}")

        log("== SETTINGS: DEFAULT LANGUAGE AND MODE")
        prof.locator("#s-language").select_option("Tamil"); time.sleep(0.4)
        prof.locator(".seg label", has_text="Meet me here").click(); time.sleep(0.4)
        log(f"Default language Tamil, Default mode 'Meet me here': 'Songs in' now still {page.locator('.st-key-language input').input_value()!r} (for the next visit) "
            f"| toast {page.locator('#moodtune-toast').inner_text()!r} | cookie {cookie(page)!r} | app reruns {reruns(page) - runs0}")
        page.keyboard.press("Escape"); time.sleep(0.3)
        page.reload(); tap_scan(page); page.wait_for_selector(".st-key-path iframe", timeout=180000); bar, path = bar_frame(page), path_frame(page); time.sleep(2)
        langs = sorted(set(bar.evaluate("DATA.tracks.map(t => t.language)")))
        log(f"after a reload: 'Songs in' {page.locator('.st-key-language input').input_value()!r} | chosen card {page.locator('.choice-card.on h3').all_inner_texts()} "
            f"(recommended {[c.locator('h3').inner_text() for c in page.locator('.choice-card').all() if c.locator('.tag').count()]}) | queue {bar.evaluate('DATA.mode')} in {langs} "
            f"| theme {theme_report(page)}")
        page.locator(".st-key-choose_uplift button").click(); page.wait_for_timeout(2500)
        picked = page.locator(".choice-card.on h3").all_inner_texts()
        page.get_by_role("button", name="Tap to scan again").click(); page.wait_for_timeout(9000); page.wait_for_selector(".st-key-path iframe", timeout=60000)
        log(f"pick 'Lift my mood' ({picked}), then 'Tap to scan again': back to the default {page.locator('.choice-card.on h3').all_inner_texts()}")
        prof = named_frame(page, "actions")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-settings").click(); time.sleep(0.4)
        prof.locator("#s-language").select_option("English"); prof.locator(".seg label", has_text="Auto").click(); time.sleep(0.4)
        log(f"defaults back to English / Auto: cookie {cookie(page)!r}"); page.keyboard.press("Escape"); time.sleep(0.3)

        log("== LIKED SONGS, HISTORY, SHORTCUTS, PRIVACY, ABOUT")
        bar, path = bar_frame(page), path_frame(page)
        for i in (1, 2): path.locator(".row").nth(i).locator(".heart").click(); time.sleep(0.2)
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(1.5)
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-liked").click()
        page.wait_for_selector(".st-key-library iframe", timeout=30000); lib = named_frame(page, "library"); time.sleep(1.5)
        log(f"Liked songs -> Library open on {lib.evaluate('window.library.current()')!r} ({lib.locator('.row').count()} songs) | still playing {not bar.evaluate('window.player.audio.paused')}")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-history").click(); time.sleep(1.5)
        log(f"History -> Library open on {lib.evaluate('window.library.current()')!r} ({lib.locator('.row').count()} songs, newest first "
            f"{lib.locator('.row .title').first.inner_text() == bar.locator('#nowtitle').inner_text()})")
        shot(page, "14_library_history")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-keys").click(); time.sleep(0.4)
        log(f"Keyboard shortcuts: {prof.locator('#dtitle').inner_text()!r} | keys {prof.locator('.keys kbd').all_inner_texts()}")
        page.keyboard.press("Escape"); time.sleep(0.3)
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-privacy").click(); time.sleep(0.4)
        privacy = prof.locator("#v-privacy").inner_text()
        log(f"Privacy: {prof.locator('#dtitle').inner_text()!r} | sections {prof.locator('#v-privacy h3').all_inner_texts()} | says the camera image is processed in memory and never saved "
            f"{'camera image is processed in memory and never saved' in privacy} | names the cookie's three values {'theme, default language and default mode' in privacy}")
        shot(page, "26_privacy"); page.keyboard.press("Escape"); time.sleep(0.3)
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-about").click(); time.sleep(0.4)
        log(f"About: {prof.locator('#dtitle').inner_text()!r} | sections {prof.locator('#v-about h3').all_inner_texts()}"); page.keyboard.press("Escape"); time.sleep(0.3)

        log("== RESET PROFILE")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-reset").click(); time.sleep(0.4)
        listed = prof.locator("#v-reset li").all_inner_texts()
        prof.locator("#resetno").click(); time.sleep(0.3)
        kept = prof.evaluate("Object.keys(localStorage).filter(k => k.startsWith('moodtune.')).sort()")
        log(f"the confirm lists {listed} | Cancel keeps everything: {kept}")
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-reset").click(); time.sleep(0.3); prof.locator("#resetyes").click(); time.sleep(1)
        left = prof.evaluate("Object.keys(localStorage).filter(k => k.startsWith('moodtune.'))")
        nav = named_frame(page, "navlists")
        log(f"Reset: moodtune keys left {left} | cookie {cookie(page)!r} | avatar shows the icon {prof.locator('#avatar svg').count() == 1} | sidebar "
            f"{dict(zip(nav.locator('.item b').all_inner_texts(), nav.locator('.item small').all_inner_texts()))} | theme {theme_report(page)} "
            f"| toast {page.locator('#moodtune-toast').inner_text()!r} | music still playing {not bar.evaluate('window.player.audio.paused')}")
        log(f"page errors {errors}")
        browser.close()
finally:
    stop(server)
cam = open(cam_log).read().count("opened") if os.path.exists(cam_log) else "?"
log(f"camera opened {cam} time(s) (each tap: two page loads with a tap each, and the rescan) | image files created or changed on disk by the app: {new_images(before)}")
