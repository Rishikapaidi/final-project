"""Headless-Edge checks of the now-playing view (sad face, so an Uplift queue; real cached previews).

It opens from the song's title, its art and the expand arrow in the player
bar, and closes with its close button and Esc; the music keeps playing (same
song, time going on) and the app never reruns. Up next is the play order
(the 5-song path in order, marked, never shuffled; a click plays that song).
Mood shows the reading's headline, its "X% low" band and the mood card's
sentence (the same as Home's). Related lists songs of the queue's language
and the reading's top emotion that are not in the queue; a click plays one.
On a narrow screen the tabs sit under the art. Runs in either theme
(TEST_THEME=dark). Saves screenshots of the three tabs and the narrow view.
"""
import os, time
from playwright.sync_api import sync_playwright
from pagekit import (OUT, bar_frame, count_reruns, image_files, log, named_frame, new_images, path_frame, read_card, reruns,
                     start_server, stop, theme_report)

PORT = 8627


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, f"page_{name}.png"))


before = image_files()
server = start_server(PORT, "sad", "nowplaying")
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}"); page.wait_for_selector(".st-key-path iframe", timeout=180000)
        bar, path = bar_frame(page), path_frame(page); time.sleep(1.5)
        state = lambda: bar.evaluate("window.player.state()")
        count_reruns(page); runs0, queue_id = reruns(page), bar.evaluate("DATA.id")
        card = read_card(page)
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2)
        log(f"theme: {theme_report(page, bar)}")

        log("== OPEN AND CLOSE")
        frame_h = lambda: bar.evaluate("Math.round(window.frameElement.getBoundingClientRect().height)")
        results = []
        for how, sel in (("title", "#nowtitle"), ("art", "#nowart"), ("expand arrow", "#expand")):
            s0 = state(); bar.locator(sel).click(); time.sleep(0.8); s1 = state()
            opened = (s1["npOpen"], frame_h(), bar.locator("#np").is_visible(), bar.evaluate("document.activeElement.id"))
            closer = "Esc" if how == "art" else "close button"
            if closer == "Esc": page.keyboard.press("Escape")
            else: bar.locator("#npclose").click()
            time.sleep(0.8); s2 = state()
            results.append(f"{how}: opens {opened[0] and opened[2]} (iframe {opened[1]}px, focus on {opened[3]!r}), {closer} closes {not s2['npOpen']} (iframe {frame_h()}px, "
                           f"focus back on {bar.evaluate('document.activeElement.id')!r}) | same song {s0['order'][s0['pos']]['link'] == s2['order'][s2['pos']]['link']} "
                           f"| playing {not s2['paused']} ({s0['time']:.1f}s -> {s2['time']:.1f}s)")
        for r in results: log(r)
        log(f"app reruns while opening and closing: {reruns(page) - runs0} | same queue {bar.evaluate('DATA.id') == queue_id}")

        log("== UP NEXT")
        bar.locator("#nowtitle").click(); time.sleep(0.8)
        s = state(); pane = bar.locator("#pane-up")
        log(f"left side: title {bar.locator('#nptitle').inner_text()!r} = playing {bar.locator('#nptitle').inner_text() == s['order'][s['pos']]['title']} | artist "
            f"{bar.locator('#npartist').inner_text()!r} | language {bar.locator('#nplang').inner_text()!r} | art {bar.locator('#npart img').count()} image | tabs {bar.locator('.tab').all_inner_texts()} "
            f"(selected {bar.locator('.tab[aria-selected=true]').inner_text()!r})")
        rows = pane.locator(".qrow")
        titles = [t.split("\n")[0] for t in pane.locator(".qrow .title").all_inner_texts()]
        log(f"Up next: {rows.count()} songs = now + the rest of the order {rows.count() == len(s['order']) - s['pos']} | in play order {titles == [o['title'] for o in s['order'][s['pos']:]]} "
            f"| path steps shown {pane.locator('.step').all_inner_texts()} | note {pane.locator('.pnote').inner_text()!r}")
        shot(page, "27_now_playing")
        target = s["order"][s["pos"] + 3]
        pane.locator(".qrow").nth(3).click(); time.sleep(1.5); s2 = state()
        log(f"click song 4 of Up next: now {s2['order'][s2['pos']]['title']!r} = {target['title']!r} {s2['order'][s2['pos']]['link'] == target['link']} | playing {not s2['paused']} "
            f"| path order still {[o['base'] for o in s2['order'] if not o['added'] and o['base'] < s2['pathLen']]} | view still open {s2['npOpen']}")
        bar.evaluate("window.player.setShuffle(true)"); time.sleep(0.3)
        log(f"shuffle while on the path: {state()['shuffle']} | Up next's path steps still {pane.locator('.step').all_inner_texts()}")

        log("== MOOD")
        bar.locator("#tab-mood").click(); time.sleep(0.5)
        mood = bar.locator("#pane-mood")
        msg = path.locator("#msg").inner_text()
        log(f"Mood: {mood.locator('h3').inner_text()!r} (Home: {card['headline']!r}) | {mood.locator('.low').inner_text().replace(chr(10), ' ')!r} (Home: {card['pct']} low {card['band']}) "
            f"| the sentence is Home's {mood.locator('.why').inner_text() == msg} | {mood.locator('.why').inner_text()[:70]!r}...")
        shot(page, "28_now_playing_mood")

        log("== RELATED")
        bar.locator("#tab-rel").click(); time.sleep(0.5)
        rel = bar.locator("#pane-rel"); data = bar.evaluate("DATA.related.map(t => ({link: t.lastfm, language: t.language, bucket: t.bucket}))")
        shown = rel.locator(".qrow.rel").evaluate_all("rs => rs.map(r => r.dataset.link)")
        in_queue = set(bar.evaluate("order.map(e => e.t.lastfm)"))
        log(f"Related: {len(shown)} songs | note {rel.locator('.pnote').inner_text()!r} | languages {sorted({d['language'] for d in data})} | emotions {sorted({d['bucket'] for d in data})} "
            f"(queue {bar.evaluate('DATA.mode')}, reading's top emotion) | none in the queue {not (set(shown) & in_queue)}")
        shot(page, "29_now_playing_related")
        first = rel.locator(".qrow.rel").nth(0); title = first.locator(".title").inner_text()
        first.click(); time.sleep(1.5); s3 = state()
        log(f"click the first: now {s3['order'][s3['pos']]['title']!r} = {title!r} | playing {not s3['paused']} | it leaves Related {title not in rel.locator('.title').all_inner_texts()} "
            f"| path order still {[o['base'] for o in s3['order'] if not o['added'] and o['base'] < s3['pathLen']]}")
        page.keyboard.press("Escape"); time.sleep(0.5)
        log(f"Esc: closed {not state()['npOpen']} | app reruns through all of it {reruns(page) - runs0} | same queue {bar.evaluate('DATA.id') == queue_id} | playing {not state()['paused']}")

        log("== NARROW SCREEN (390 px)")
        page.set_viewport_size({"width": 390, "height": 844}); time.sleep(1.2)
        bar.locator("#nowtitle").click(); time.sleep(0.8)
        art, tabs = bar.locator("#npart").bounding_box(), bar.locator(".tabs").bounding_box()
        log(f"open: tabs under the art {tabs['y'] >= art['y'] + art['height']} | art {round(art['width'])}px wide | expand arrow hidden {bar.locator('#expand').is_hidden()} "
            f"| no sideways scroll {bar.evaluate('document.documentElement.scrollWidth <= innerWidth')} | playing {not state()['paused']}")
        shot(page, "30_now_playing_narrow")
        bar.locator("#npclose").click(); time.sleep(0.5)
        log(f"closed {not state()['npOpen']} | page errors {errors}")
        browser.close()
finally:
    stop(server)
log(f"image files created or changed on disk by the app: {new_images(before)}")
