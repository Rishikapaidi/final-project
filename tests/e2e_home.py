"""Headless-Edge checks of Home's layout (real cached previews).

With a sad face: the chips row at the top (six languages, seven moods), the
"Detect my mood" card holding the scan, the words and the reading, and Quick
picks (12 songs of the top emotion in "Songs in", three columns, Play all, a
tile plays its song, the search box). A language chip only picks the
language; a mood chip replaces the player's queue with that mood in that
language (Explore's pick, in its order) and plays the first, once: later
reruns and a trip to the Library and back don't play it again; a song from
Home's list brings the mood queue back. Quick picks stay the same through
reruns and page changes and differ from Explore's first 12. With no face: Quick picks shows a mix before there is a
reading, and the reading's placeholder sits in the card until words arrive.
Saves screenshots of the top of Home and of Quick picks.
"""
import os, time
from playwright.sync_api import sync_playwright
from pagekit import OUT, bar_frame, image_files, log, named_frame, new_images, path_frame, start_server, stop

PORT = 8624


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, f"page_{name}.png"))


def picks(page, not_id=None, timeout=30):
    """Quick picks' component; with not_id, wait for a different list."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            f = named_frame(page, "quickpicks", timeout=5)
            if f.evaluate("DATA.id") != not_id:
                return f
        except Exception:
            pass
        page.wait_for_timeout(300)
    raise RuntimeError("Quick picks not found")


def top(page, sel):
    return round(page.locator(sel).first.bounding_box()["y"])


before = image_files()
server = start_server(PORT, "sad", "home")
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}")
        page.wait_for_selector(".st-key-quickpicks iframe", timeout=180000)
        qp = picks(page); during = qp.locator("#title").inner_text()
        page.wait_for_selector(".st-key-path iframe", timeout=180000); bar = bar_frame(page); path = path_frame(page); time.sleep(1.5)
        state = lambda: bar.evaluate("window.player.state()")
        queue_id = bar.evaluate("DATA.id")
        log("== LAYOUT")
        log(f"top to bottom: chips {top(page, '.st-key-chips')} < greeting {top(page, '.greeting')} < card {top(page, '.st-key-hero')} < Quick picks "
            f"{top(page, '.st-key-quickpicks')} < choice {top(page, '.st-key-choice')} < songs {top(page, '.st-key-path')}")
        hero = page.locator(".st-key-hero")
        log(f"'Detect my mood' card: title {hero.locator('h2').first.inner_text()!r} | holds the scan {hero.locator('.st-key-facetile').count()}, the words "
            f"{hero.locator('.st-key-texttile').count()}, the gauge {hero.locator('#gauge').count()} and the result {hero.locator('#moodhead').inner_text()!r}")
        langs = [t.strip() for t in page.locator("[class*='st-key-chip_lang_'] button").all_inner_texts()]
        moods = [t.strip() for t in page.locator("[class*='st-key-chip_mood_'] button").all_inner_texts()]
        log(f"chips: languages {langs} (selected {page.locator('[class*=st-key-chip_lang_] button[kind=primary]').inner_text()!r}) | moods {moods}")

        log("== QUICK PICKS")
        qp = picks(page); tiles = qp.locator(".tile")
        xs = sorted({round(tiles.nth(k).bounding_box()["x"]) for k in range(tiles.count())})
        data = qp.evaluate("window.quickpicks.tracks()")
        log(f"while the first scan ran: {during!r} | after it: {qp.locator('#title').inner_text()!r} / {qp.locator('#sub').inner_text()!r} | {tiles.count()} tiles in "
            f"{len(xs)} columns | moods {sorted({t['bucket'] for t in data})} | languages {sorted({t['language'] for t in data})} | each shows art, title, artist "
            f"{all(tiles.nth(k).locator('.art').count() and tiles.nth(k).locator('.title').inner_text() and tiles.nth(k).locator('.artist').count() for k in range(tiles.count()))}")
        page.locator(".st-key-quickpicks").scroll_into_view_if_needed(); time.sleep(0.5); shot(page, "23_quick_picks")
        qp.locator("#playall").click(); time.sleep(2); s = state()
        log(f"Play all: now {s['order'][s['pos']]['title']!r} = tile 1 {s['order'][s['pos']]['title'] == data[0]['title']} | added {sum(o['added'] for o in s['order'])} "
            f"| playing {not s['paused']} | mood queue still there {bar.evaluate('DATA.id') == queue_id}")
        tiles.nth(4).click(); time.sleep(2); s = state()
        log(f"tile 5: now {s['order'][s['pos']]['title']!r} = {data[4]['title']!r} | tile marked playing {qp.locator('.tile.playing .title').all_inner_texts()}")
        word = data[7]["title"].split()[0].lower()[:4]
        named_frame(page, "topbar").locator("#q").fill(word); time.sleep(0.5)
        log(f"search {word!r}: {qp.locator('.tile:not([hidden])').count()} of {tiles.count()} tiles shown"); named_frame(page, "topbar").locator("#q").fill(""); time.sleep(0.3)

        log("== CHIPS")
        qp_id = picks(page).evaluate("DATA.id")
        page.locator(".st-key-chip_lang_tamil button").click(); time.sleep(2); s1 = state()
        log(f"Tamil chip: selected {page.locator('[class*=st-key-chip_lang_] button[kind=primary]').inner_text()!r} | nothing new played (same song {s1['current'] == s['current']}, "
            f"same order {s1['order'] == s['order']}) | 'Songs in' still {page.locator('.st-key-language input').input_value()!r} | same queue {bar.evaluate('DATA.id') == queue_id} "
            f"| Quick picks unchanged by the rerun {picks(page).evaluate('DATA.id') == qp_id}")
        page.locator(".st-key-chip_mood_happy button").click(); time.sleep(3); s2 = state()
        chip_links = [o["link"] for o in s2["order"]]
        art = bar.locator("#nowart").evaluate("e => e.tagName + ' ' + (e.src || '').slice(0, 30)")
        log(f"Happy chip replaces the queue: {len(s2['order'])} songs, none 'added' {not any(o['added'] for o in s2['order'])} | plays the first {s2['pos'] == 0 and not s2['paused']} "
            f"| queue panel label {s2['context']!r} | toast {page.locator('#moodtune-toast').inner_text()!r} | chip marked {[t.strip() for t in page.locator('[class*=st-key-chip_mood_] button[kind=primary]').all_inner_texts()]}")
        log(f"   the bar shows the song: {bar.locator('#nowtitle').inner_text()!r} = first song {bar.locator('#nowtitle').inner_text() == s2['order'][0]['title']} | art {art!r}")
        page.locator(".st-key-chip_lang_english button").click(); time.sleep(2); s3 = state()
        log(f"a rerun after it (English chip) plays nothing again: same song {s3['pos'] == s2['pos'] and s3['order'][s3['pos']]['link'] == s2['order'][s2['pos']]['link']} "
            f"| same list {[o['link'] for o in s3['order']] == chip_links}")
        page.locator(".st-key-nav_library button").click(); page.wait_for_selector(".st-key-library iframe", timeout=30000); time.sleep(1)
        page.locator(".st-key-nav_home button").click(); page.wait_for_selector(".st-key-path iframe", timeout=30000); time.sleep(2); s4 = state()
        log(f"   nor does a trip to the Library and back: same song {s4['order'][s4['pos']]['link'] == s3['order'][s3['pos']]['link']} | same list {[o['link'] for o in s4['order']] == chip_links} "
            f"| Quick picks the same {picks(page).evaluate('DATA.id') == qp_id}")
        page.locator(".st-key-nav_explore button").click(); page.wait_for_selector(".st-key-explorelist iframe", timeout=30000)
        page.locator(".st-key-xcard_lang_tamil button").click(); time.sleep(2.5); page.locator(".st-key-xcard_mood_happy button").click(); time.sleep(3)
        ex = named_frame(page, "songlist")
        ex_links = ex.evaluate("DATA.tracks.map(t => t.lastfm)"); ex_info = ex.evaluate("[[...new Set(DATA.tracks.map(t => t.bucket))], [...new Set(DATA.tracks.map(t => t.language))]]")
        log(f"   the chip's list is Explore's Happy / Tamil pick: {ex.locator('#title').inner_text()!r} {ex_info} | same songs, same order {chip_links == ex_links}")
        page.locator(".st-key-xcard_lang_english button").click(); time.sleep(2.5); page.locator(".st-key-xcard_mood_sad button").click(); time.sleep(3)
        ex_first = named_frame(page, "songlist").evaluate("DATA.tracks.slice(0, 12).map(t => t.lastfm)")
        page.locator(".st-key-nav_home button").click(); page.wait_for_selector(".st-key-path iframe", timeout=30000); time.sleep(1.5)
        qp_links = [t["link"] for t in picks(page).evaluate("window.quickpicks.tracks()")]
        log(f"Quick picks (sad, English) against Explore's first 12 for sad / English: {len(set(qp_links) & set(ex_first))} in common | same as before the trips "
            f"{picks(page).evaluate('DATA.id') == qp_id}")
        path = path_frame(page); path.locator(".row").nth(3).locator(".pill").click(); time.sleep(2); s5 = state()
        log(f"a song of Home's list brings the mood queue back: label {s5['context']!r} | {len(s5['order'])} songs = the list's {len(s5['order']) == path.evaluate('DATA.tracks.length')} "
            f"| playing song 4 {s5['current'] == 3 and not s5['paused']} | same queue {bar.evaluate('DATA.id') == queue_id}")
        page.locator(".st-key-chips").scroll_into_view_if_needed(); page.evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)"); time.sleep(0.5)
        shot(page, "22_home_top")

        log("== NARROW SCREEN (390 px)")
        page.set_viewport_size({"width": 390, "height": 844}); time.sleep(1.5)
        qp = picks(page); tiles = qp.locator(".tile")
        cols = len({round(tiles.nth(k).bounding_box()["x"]) for k in range(tiles.count())})
        chips = page.locator(".st-key-chips").evaluate("e => [e.scrollWidth > e.clientWidth, getComputedStyle(e).flexWrap]")
        log(f"Quick picks in {cols} column(s) | chips row scrolls sideways {chips[0]} ({chips[1]}) | page has no sideways scroll {page.evaluate('document.documentElement.scrollWidth') <= 390}")
        log(f"page errors {errors}")
        browser.close()
finally:
    stop(server)

server = start_server(PORT + 1, "blank", "home_blank")
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT + 1}"); page.wait_for_selector("[data-testid=stAlert]", timeout=180000); time.sleep(1.5)
        log("== NO FACE, BEFORE ANY READING")
        qp = picks(page); data = qp.evaluate("window.quickpicks.tracks()")
        log(f"Quick picks: {qp.locator('#title').inner_text()!r} / {qp.locator('#sub').inner_text()!r} | {len(data)} songs | languages {sorted({t['language'] for t in data})} "
            f"| moods {sorted({t['bucket'] for t in data})} | reading placeholder in the card {page.locator('.st-key-hero .result-wait').inner_text()!r}")
        old = qp.evaluate("DATA.id")
        box = page.get_by_label("How was your day?"); box.fill("I miss her so much, everything feels empty."); box.press("Enter")
        page.wait_for_selector(".st-key-path iframe", timeout=120000); qp = picks(page, old, timeout=60); data = qp.evaluate("window.quickpicks.tracks()")
        log(f"after sad words: Quick picks {qp.locator('#title').inner_text()!r} | moods {sorted({t['bucket'] for t in data})} | the reading in the card "
            f"{page.locator('.st-key-hero #moodhead').inner_text()!r} | placeholder gone {page.locator('.result-wait').count() == 0} | page errors {errors}")
        browser.close()
finally:
    stop(server)
log(f"image files created or changed on disk by the app: {new_images(before)}")
