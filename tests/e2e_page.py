"""Headless-Edge checks and screenshots of the page, with real cached previews.

Scenarios: sad face (Uplift), happy face (Resonance), a happy face with a sad
sentence (a "Mixed: ..." headline), and no face with a sad sentence. The first
one also checks playback, the expander, the upload popover, the choice cards and
the narrow layout.
"""
import os, sys, time
from playwright.sync_api import sync_playwright
from pagekit import (tap_scan, named_frame, theme_report, OUT, bar_frame, describe_card, full_shot, image_files, log, new_images, path_frame,
                     player_state, players, read_card, set_bars_open, start_server, stop, wait_for_text_bars)

SAD_TEXT = "I miss her so much, everything feels empty and pointless now."
SCENARIOS = [  # name, frame, text, extra playback checks
    ("1_sad_face_uplift", "sad", "", True),
    ("2_happy_face_resonance", "happy", "", False),
    ("3_mixed_happy_face_sad_text", "mixed", SAD_TEXT, False),
    ("4_no_face_text_only", "blank", SAD_TEXT, False),
]
ONLY = sys.argv[1:]  # optional scenario name filter
NO_FACE_NOTE = ["No face was found in this picture, so your mood comes from your words.", "Tip: face the light, with the window in front of you, not behind."]
# Text that must never appear on the page: Streamlit's "Running <function>()" and the app's function names.
FLAG = r"Running|load_songs|load_face_model|load_text_model|summarise_scan|text_probabilities|build_queue|build_view|cache_resource|cache_data"
WATCH = """
window.__flagged = [];
const FLAG = new RegExp(%s);
const check = () => { const t = document.body ? document.body.innerText : ""; const m = t.match(FLAG);
  if (m) window.__flagged.push(t.slice(Math.max(0, m.index - 30), m.index + 50)); };
new MutationObserver(check).observe(document, { subtree: true, childList: true, characterData: true }); setInterval(check, 100);
""" % repr(FLAG)

before = image_files()
with sync_playwright() as pw:
    browser = pw.chromium.launch(channel="msedge", headless=True)
    for n, (name, frame, text, playback) in enumerate(SCENARIOS):
        if ONLY and not any(o in name for o in ONLY): continue
        cam_log = os.path.join(OUT, f"cam_{name}.log")
        if os.path.exists(cam_log): os.remove(cam_log)
        server = start_server(8590 + n, frame, name, TEST_CAM_LOG=cam_log)
        opened = lambda: open(cam_log).read().count("opened") if os.path.exists(cam_log) else 0
        try:
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}); ctx.add_init_script(WATCH)
            page = ctx.new_page()
            errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://localhost:{8590 + n}")
            log(f"=== {name}")
            # First load: nothing is scanned and the webcam stays off until "Tap to scan".
            page.wait_for_selector("#scanidle", timeout=180000)
            page.wait_for_selector(".st-key-quickpicks iframe", timeout=180000); time.sleep(3)
            log(f"first load: camera opened {opened()} time(s) after 3 s (0 = off until the tap) | idle tile "
                f"{page.locator('#scanidle').inner_text()!r} | button {page.locator('.st-key-rescan button >> visible=true').first.inner_text()!r} | "
                f"reading placeholder {page.locator('.result-wait').count() == 1} | warnings {page.locator('[data-testid=stAlert]').count()} | "
                f"choice cards {page.locator('.st-key-choice').count()} | page errors {errors}")
            if n == 0:
                full_shot(page, os.path.join(OUT, "page_00_first_load.png"))
            tap_scan(page)
            if frame == "blank":
                page.wait_for_selector("[data-testid=stAlert]", timeout=180000)
                log("before any text:", page.locator("[data-testid=stAlert]").first.inner_text()[:100])
            else:
                page.wait_for_selector(".st-key-path iframe", timeout=180000)
            top = named_frame(page, "topbar")
            log(f"top bar: {top.locator('.brand').inner_text()!r} | logo {top.locator('svg.logo').count()} | menu button {top.locator('#hamburger').count()} "
                f"| search {top.locator('#q').get_attribute('placeholder')!r} | 'Songs in' {page.locator('.st-key-language [data-testid=stWidgetLabel]').inner_text()!r} "
                f"| avatar {named_frame(page, 'actions').locator('#avatar').get_attribute('aria-label')!r} | pinned {page.locator('.st-key-topbar').evaluate('e => getComputedStyle(e).position')}")
            log(f"sidebar: {[b.strip() for b in page.locator('.st-key-nav .stButton').all_inner_texts()]} (current: {page.locator('.st-key-nav button[kind=primary]').inner_text().strip()!r}) "
                f"| below them {named_frame(page, 'navlists').locator('#new, .item b').all_inner_texts()}")
            log(f"date {page.locator('.greeting .date').inner_text()!r} | greeting {page.locator('.greeting h1').inner_text()!r} | tab {page.title()!r}")
            log(f"hero: {page.locator('.st-key-hero h2').first.inner_text()!r} / {page.locator('.hero-sub').inner_text()!r} | "
                f"buttons {list(dict.fromkeys(b.replace('expand_more', '').strip() for b in page.locator('.st-key-hero button').all_inner_texts()))}")
            fonts = page.evaluate("[document.fonts.check('16px Fraunces'), document.fonts.check('16px Inter'), getComputedStyle(document.querySelector('.greeting h1')).fontFamily.split(',')[0]]")
            log(f"fonts loaded: Fraunces {fonts[0]}, Inter {fonts[1]} | heading font {fonts[2]}")
            if text:
                box = page.get_by_label("How was your day?")
                box.click(); page.keyboard.type("x" * 200); time.sleep(0.3)
                too_long = len(box.input_value())
                box.fill(text); time.sleep(0.3)
                counter = page.locator(".st-key-user_text [data-testid=InputInstructions]").inner_text().replace(chr(10), " ")
                log(f"text box: typing 200 characters keeps {too_long} | counter {counter!r} | placeholder {box.get_attribute('placeholder')!r}")
                box.press("Enter")
                waited = wait_for_text_bars(page)
                log(f"   Text bars appeared {waited:.1f}s after Enter" if waited is not None else "   NO Text bars after 60 s")
            bar = bar_frame(page); path = path_frame(page); time.sleep(1.5)
            log(f"theme: {theme_report(page, bar, path, named_frame(page, 'topbar'), named_frame(page, 'actions'), named_frame(page, 'navlists'), named_frame(page, 'quickpicks'))}")
            card = read_card(page)
            log(f"mood card: {describe_card(card)}")
            log(f"   sentence {card['sentence']!r}")
            combined = card["groups"]["combined"]
            log(f"   combined: " + ", ".join(f"{e} {combined[e]['pct']}" for e in card["emotions"]) + f" | caption {card['caption']!r}")
            for g in card["groups"]:
                if g != "combined": log(f"   {g:8}: " + ", ".join(f"{e} {card['groups'][g][e]['pct']}" for e in card["emotions"]))
            img = page.locator(".scanimg")
            if img.count():
                size = img.evaluate("e => [e.naturalWidth, e.naturalHeight, e.src.slice(0, 23), Math.round(e.src.length / 1024)]")
                notes = page.locator('.tile-note').all_inner_texts()
                log(f"scan picture: {size[0]}x{size[1]}, src starts {size[2]!r}, {size[3]} KB in the page | note {notes}")
                expected = NO_FACE_NOTE if frame == "blank" else []
                log(f"note under the picture {'as expected' if notes == expected else 'WRONG, expected ' + repr(expected)}"
                    f" ({'no-face message and tip' if expected else 'no note, no tip'}) | tip lines {page.locator('.tile-tip').count()}")
            else:
                log(f"scan picture: none | {page.locator('.st-key-facetile').inner_text()[:80]!r}")
            choice = page.locator(".choice-card")
            log(f"choice cards: selected {page.locator('.choice-card.on h3').all_inner_texts()} | recommended {[c.locator('h3').inner_text() for c in choice.all() if c.locator('.tag').count()]}")
            log(f"path card: {path.locator('#kicker').inner_text()!r} {path.locator('#title').inner_text()!r} / {path.locator('#qsub').inner_text()!r} | steps {path.locator('#steps li:not(.arrow)').all_inner_texts()} "
                f"| rows {path.locator('.row').count()} | pills {path.locator('.row .pill').count()} | hearts {path.locator('.row .heart').count()} | menus {path.locator('.row .more').count()} "
                f"| links {path.evaluate('[...new Set(DATA.tracks.map(t => t.link_label))]')} | search box in the card {path.locator('#search').count()} (it is in the top bar) | sections {path.locator('.section').all_inner_texts()} | journey chart {path.locator('details.journey').count()}")
            log(f"   message {path.locator('#msg').inner_text()!r}" + (f" | note {path.locator('#note').inner_text()!r}" if path.locator('#note').is_visible() else ""))
            log(f"player bar: {bar.locator('#nowtitle').inner_text()!r} | buttons {bar.locator('button').evaluate_all('bs => bs.map(b => b.ariaLabel)')} | volume {bar.locator('#volume').count()} "
                f"| pinned to the bottom: {page.locator('.st-key-playerbar').evaluate('e => getComputedStyle(e).position + \" bottom=\" + (innerHeight - e.getBoundingClientRect().bottom)')}")
            set_bars_open(page, bool(text))  # the screenshots with a sentence show the Face / Text / Combined bars
            full_shot(page, os.path.join(OUT, f"page_{name}.png"))

            if playback:
                state = lambda: player_state(bar)
                path.locator(".row").nth(2).locator(".meta").click(); time.sleep(2.5); a = state()
                log(f"click on row 3: {a} | pill now says {path.locator('.row').nth(2).locator('.pill').inner_text()!r} | bar: {bar.locator('#nowtitle').inner_text()!r}")
                path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2.5); b = state()
                log(f"click on row 1's Play pill: {b} | highlighted {path.locator('.row.playing .title').inner_text()!r}")
                path.locator(".row").nth(0).locator(".pill").click(); time.sleep(0.5); c = state()
                path.locator(".row").nth(0).locator(".pill").click(); time.sleep(1.0)
                log(f"same pill again pauses: {c['paused']} | and again resumes: {not state()['paused']}")
                bar.locator("#toggle").click(); time.sleep(0.5); c2 = state(); bar.locator("#toggle").click(); time.sleep(1)
                log(f"bar play/pause: pauses {c2['paused']}, resumes {not state()['paused']}")
                details = page.locator(".st-key-moodcard details")
                was_open = details.get_attribute("open") is not None
                page.get_by_text("See all 7 emotions").click(); time.sleep(1.5); d1 = state(); time.sleep(2); d2 = state()
                log(f"expander {'closed' if was_open else 'opened'}: still playing {not d2['paused']}, same track {d2['current'] == d1['current']}, time {d1['time']} -> {d2['time']} | open now {details.get_attribute('open') is not None}")
                page.get_by_text("See all 7 emotions").click(); time.sleep(0.5)
                log(f"old ⚙ upload popover in the top bar: {page.get_by_role('button', name='⚙').count()} | uploaders on the page: {page.locator('[data-testid=stFileUploader]').count()} (closed popover)")
                page.get_by_role("button", name="Upload a photo instead").click(); time.sleep(1.5); d1 = state(); time.sleep(2); d2 = state()
                page.screenshot(path=os.path.join(OUT, "page_5_upload_while_playing.png"))
                log(f"upload popover open: uploaders {page.locator('[data-testid=stFileUploader]').count()} | still playing {not d2['paused']}, same track {d2['current'] == d1['current']}, time {d1['time']} -> {d2['time']} | contents {page.locator('[data-testid=stPopoverBody]').inner_text()[:90]!r}")
                page.keyboard.press("Escape"); time.sleep(0.4)
                bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1.5"); time.sleep(4.5); e = state()
                log(f"near the end of track 1 -> {e} (next song started on its own: {e['current'] == 1 and not e['paused']}) | path highlights row {path.evaluate('window.playlist.state().current') + 1}")
                path.locator("details.journey summary").click(); time.sleep(0.8)
                page.locator(".st-key-path").scroll_into_view_if_needed(); time.sleep(0.5)
                page.evaluate("document.querySelector('[data-testid=stMain]').scrollBy(0, 330)"); time.sleep(0.8)
                page.screenshot(path=os.path.join(OUT, "page_6_playing_with_mood_journey.png"))
                log(f"journey chart open: {path.locator('#chart').is_visible()} | path card grew to {path.evaluate('window.frameElement.style.height')}")
                # A choice card rebuilds the queue without opening the camera again.
                cam_before, tracks_before = opened(), path.evaluate("DATA.tracks.map(t => t.lastfm)")
                # click the middle of the visible card with the mouse, the way a person would
                card = page.locator(".choice-card.resonance"); card.scroll_into_view_if_needed(); time.sleep(0.5)
                box = card.bounding_box(); page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                path = path_frame(page, not_tracks=tracks_before); time.sleep(2); bar = bar_frame(page)
                count, sounding = players(page)
                log(f"'Meet me here' card clicked (middle of the card): list rebuilt True | {path.locator('#title').inner_text()!r} | selected {page.locator('.choice-card.on h3').all_inner_texts()} "
                    f"| players {count}, still playing {sounding} | camera opened {cam_before} -> {opened()} time(s) | 'Let MOOD TUNE choose' {page.get_by_role('button', name='Let MOOD TUNE choose').count()}")
                tracks_before = path.evaluate("DATA.tracks.map(t => t.lastfm)")
                page.get_by_role("button", name="Let MOOD TUNE choose").click()
                path = path_frame(page, not_tracks=tracks_before); time.sleep(1.5)
                log(f"'Let MOOD TUNE choose': {path.locator('#title').inner_text()!r} | selected {page.locator('.choice-card.on h3').all_inner_texts()} | camera opened {opened()} time(s)")
                full_shot(page, os.path.join(OUT, "page_12_narrow_screen.png"), width=390)
                page.set_viewport_size({"width": 390, "height": 844}); time.sleep(1.5)
                widths = page.evaluate("[document.documentElement.scrollWidth, document.querySelector('[data-testid=stMain]').scrollWidth, innerWidth]")
                cols = page.evaluate("[...document.querySelectorAll('.st-key-hero [data-testid=stColumn]')].map(c => Math.round(c.getBoundingClientRect().width))")
                log(f"narrow screen (390 px): page width {widths[0]}, content width {widths[1]} (no sideways scroll: {max(widths[:2]) <= widths[2]}) | hero tiles stacked, widths {cols}")
                page.set_viewport_size({"width": 1440, "height": 900}); time.sleep(0.5)
            log(f"'Running' or a function name on the page at any point: {page.evaluate('window.__flagged.length')} {page.evaluate('window.__flagged.slice(0, 2)')}")
            log(f"page errors: {errors}")
            ctx.close()
        finally:
            stop(server)
    browser.close()

log(f"=== image files created or changed on disk during the run (outside the test's own screenshots): {new_images(before)}")
