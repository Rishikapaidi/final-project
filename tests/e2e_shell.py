"""Headless-Edge checks of the app's shell (sad face, so an Uplift queue, real cached previews).

- Moving between Home, Explore and Library: the music keeps playing (same
  queue, same song, time going on), the words typed on Home are still there,
  and "Songs in" changed on another page still rebuilds the queue.
- Explore: seven mood tiles and six language tiles (Telugu and Malayalam
  say "Meet me here only"); a mood or language picks a list of songs of only
  that mood and language; a song played there starts at once; the search box.
- Library and the sidebar: New playlist (from the sidebar), Rename, Delete,
  Recently played (the songs played, newest first, six fields only).
- The hamburger folds the sidebar to icons (kept after a reload) and, on a
  narrow screen, slides it in over the page.
- The avatar shows the first letter of the name given in its menu.
Saves screenshots of Explore, the folded sidebar, the narrow sidebar and the
New playlist dialog.
"""
import os, time
from playwright.sync_api import sync_playwright
from pagekit import (OUT, bar_frame, count_reruns, image_files, log, named_frame, new_images, path_frame, reruns,
                     start_server, stop)

PORT = 8622
FLAG = r"Running|load_songs|load_face_model|load_text_model|summarise_scan|text_probabilities|build_queue|build_view|cache_resource|cache_data"
WATCH = """
window.__flagged = [];
const FLAG = new RegExp(%s);
const check = () => { const t = document.body ? document.body.innerText : ""; const m = t.match(FLAG);
  if (m) window.__flagged.push(t.slice(Math.max(0, m.index - 30), m.index + 50)); };
new MutationObserver(check).observe(document, { subtree: true, childList: true, characterData: true }); setInterval(check, 100);
""" % repr(FLAG)
SELECTOR = {"home": ".st-key-path iframe", "explore": ".st-key-explorelist iframe", "library": ".st-key-library iframe"}


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, f"page_{name}.png"))


def go(page, name):
    page.locator(f".st-key-nav_{name} button").click()
    page.wait_for_selector(SELECTOR[name], timeout=30000); page.wait_for_timeout(1500)


def sidebar(page):
    nav = named_frame(page, "navlists")
    return dict(zip(nav.locator(".item b").all_inner_texts(), nav.locator(".item small").all_inner_texts()))


def songlist(page, not_id=None, timeout=30):
    """Explore's list; with not_id, wait for a different one."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            f = named_frame(page, "songlist", timeout=5)
            if f.evaluate("DATA.id") != not_id:
                return f
        except Exception:
            pass
        page.wait_for_timeout(300)
    raise RuntimeError("Explore list not found")


before = image_files()
server = start_server(PORT, "sad", "shell")
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}); ctx.add_init_script(WATCH)
        page = ctx.new_page(); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://localhost:{PORT}"); page.wait_for_selector(SELECTOR["home"], timeout=180000)
        bar, path = bar_frame(page), path_frame(page); time.sleep(1.5)
        state = lambda: bar.evaluate("window.player.state()")
        count_reruns(page)
        top, prof = named_frame(page, "topbar"), named_frame(page, "actions")
        log(f"top bar: brand {top.locator('.brand').inner_text()!r} | hamburger {top.locator('#hamburger').count()} | search {top.locator('#q').count()} "
            f"| 'Songs in' {page.locator('.st-key-language').count()} | avatar {prof.locator('#avatar').count()} | pinned at top {page.locator('.st-key-topbar').bounding_box()['y'] == 0}")
        log(f"sidebar: {[b.strip() for b in page.locator('.st-key-nav .stButton').all_inner_texts()]} | New playlist {named_frame(page, 'navlists').locator('#new').count()} | {sidebar(page)}")

        # --- the music keeps playing across pages
        log("== MOVING BETWEEN PAGES")
        box = page.get_by_label("How was your day?"); box.fill("I miss her so much, everything feels empty."); box.press("Enter")
        page.wait_for_timeout(2000); page.wait_for_selector(SELECTOR["home"], timeout=90000)
        for _ in range(90):
            if page.locator(".legend").count() and "Text" in page.locator(".legend").first.inner_text(): break
            page.wait_for_timeout(1000)
        bar = bar_frame(page); path = path_frame(page); time.sleep(1)
        queue_id = bar.evaluate("DATA.id")
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2)
        runs0 = reruns(page)
        trip = []
        for name in ("explore", "library", "home"):
            t0 = state()["time"]; go(page, name); s = state()
            trip.append(f"{name}: playing {not s['paused']}, song {s['current'] + 1}, time {t0:.1f}->{s['time']:.1f}, same queue {bar.evaluate('DATA.id') == queue_id}")
        log("trip Home -> Explore -> Library -> Home: " + " | ".join(trip))
        log(f"   one player on the page: {page.locator('.st-key-playerbar iframe').count()} | words still in the box {page.get_by_label('How was your day?').input_value()!r} "
            f"| app reruns {reruns(page) - runs0} (one per page) | current page marked {page.locator('.st-key-nav button[kind=primary]').inner_text().strip()!r}")
        path = path_frame(page)

        # --- Explore
        log("== EXPLORE")
        go(page, "explore")
        moods = page.locator(".xcard.mood h3").all_inner_texts(); langs = page.locator(".xcard.lang h3").all_inner_texts()
        only = [page.locator(".xcard.lang").nth(k).locator("h3").inner_text() for k in range(page.locator(".xcard.lang").count())
                if "Meet me here only" in page.locator(".xcard.lang").nth(k).inner_text()]
        log(f"mood tiles {moods} | language tiles {langs} | 'Meet me here only' on {only} | counts {page.locator('.xcard.mood p').all_inner_texts()}")
        lst = songlist(page); log(f"before a mood: {lst.locator('#title').inner_text()!r} ({lst.locator('.row').count()} songs) | languages {sorted(set(lst.evaluate('DATA.tracks.map(t => t.language)')))}")
        old = lst.evaluate("DATA.id"); page.locator(".st-key-xcard_mood_happy button").click(); lst = songlist(page, old)
        log(f"Happy: {lst.locator('#title').inner_text()!r} | {lst.locator('.row').count()} songs | moods {sorted(set(lst.evaluate('DATA.tracks.map(t => t.bucket)')))} "
            f"| languages {sorted(set(lst.evaluate('DATA.tracks.map(t => t.language)')))} | tile selected {page.locator('.xcard.mood.on h3').all_inner_texts()} | all have a preview {lst.evaluate('DATA.tracks.every(t => t.preview)')}")
        shot(page, "18_explore")
        old = lst.evaluate("DATA.id"); page.locator(".st-key-xcard_lang_tamil button").click(); lst = songlist(page, old)
        log(f"then Tamil: {lst.locator('#title').inner_text()!r} | moods {sorted(set(lst.evaluate('DATA.tracks.map(t => t.bucket)')))} | languages {sorted(set(lst.evaluate('DATA.tracks.map(t => t.language)')))} "
            f"| 'Songs in' unchanged {page.locator('.st-key-language input').input_value()!r} | same queue {bar.evaluate('DATA.id') == queue_id}")
        old = lst.evaluate("DATA.id"); page.locator(".st-key-xcard_lang_telugu button").click(); lst = songlist(page, old)
        log(f"Telugu: {lst.locator('#title').inner_text()!r} | languages {sorted(set(lst.evaluate('DATA.tracks.map(t => t.language)')))}")
        lst.locator(".row").nth(2).locator(".pill").click(); time.sleep(2); s = state()
        log(f"play Explore song 3: now {s['order'][s['pos']]['title']!r} == {lst.locator('.row').nth(2).locator('.title').inner_text()!r} (added {s['order'][s['pos']]['added']}) "
            f"| row marked {lst.locator('.row.playing').count()} | Home's queue still there {bar.evaluate('DATA.id') == queue_id}")
        word = lst.locator(".row").nth(4).locator(".title").inner_text().split()[0].lower()[:4]
        top.locator("#q").fill(word); time.sleep(0.5)
        log(f"search {word!r} on Explore: {lst.locator('.row:not([hidden])').count()} of {lst.locator('.row').count()} shown ({lst.locator('#matches').inner_text()!r})")
        top.locator("#q").fill(""); time.sleep(0.3)
        old = lst.evaluate("DATA.id"); page.locator(".st-key-xcard_mood_happy button").click(); lst = songlist(page, old)
        log(f"Happy again drops the mood: {lst.locator('#title').inner_text()!r} | tiles selected {page.locator('.xcard.on h3').all_inner_texts()}")

        # --- Library: playlists and Recently played
        log("== LIBRARY AND THE SIDEBAR")
        nav = named_frame(page, "navlists")
        nav.locator("#new").click(); page.wait_for_timeout(400)
        log(f"New playlist (sidebar): dialog {page.locator('#moodtune-dialog h2').inner_text()!r} | focus in the box {page.evaluate('document.activeElement.id')!r}")
        page.locator("#mtd-input").fill("Rainy day"); shot(page, "21_new_playlist")
        page.locator("#moodtune-dialog .mtd-ok").click()
        page.wait_for_selector(SELECTOR["library"], timeout=30000); lib = named_frame(page, "library"); time.sleep(1.5)
        log(f"   -> Library open on {lib.locator('#vtitle').inner_text()!r} | empty note {lib.locator('#empty').inner_text()!r} | sidebar {list(sidebar(page))}")
        lib.locator("#vrename").click(); page.wait_for_timeout(300); page.locator("#mtd-input").fill("Rainy days"); page.locator("#moodtune-dialog .mtd-ok").click(); time.sleep(0.5)
        log(f"   Rename -> {lib.locator('#vtitle').inner_text()!r} | sidebar {list(sidebar(page))}")
        lib.locator("#vdelete").click(); time.sleep(0.2); text = lib.locator("#confirmtext").inner_text(); lib.locator("#confirmyes").click(); time.sleep(0.5)
        log(f"   Delete ({text!r}) -> back on the cards {lib.locator('#home').is_visible()} | cards {lib.locator('.coll b').all_inner_texts()} | sidebar {list(sidebar(page))}")
        lib.locator(".coll", has_text="Recently played").click(); time.sleep(0.5)
        recent = lib.locator(".row .title").all_inner_texts()
        fields = prof.evaluate("Object.keys(JSON.parse(localStorage.getItem('moodtune.recent'))[0]).sort()")
        log(f"Recently played: {len(recent)} songs, newest first {recent[:1] == [bar.locator('#nowtitle').inner_text()]} | {lib.locator('#vcount').inner_text()!r} | stored fields {fields}")
        page.get_by_role("combobox", name="Songs in").click(); time.sleep(0.5); page.get_by_role("option", name="Hindi", exact=True).click()
        page.wait_for_timeout(1500); bar = bar_frame(page)
        for _ in range(60):
            if bar.evaluate("DATA.id") != queue_id: break
            page.wait_for_timeout(500); bar = bar_frame(page)
        log(f"'Songs in' Hindi while on the Library: new queue {bar.evaluate('DATA.id') != queue_id} | its songs {sorted(set(bar.evaluate('DATA.tracks.map(t => t.language)')))} "
            f"| still on the Library {page.locator(SELECTOR['library']).count() == 1}")

        # --- hamburger and narrow screens
        log("== HAMBURGER")
        top.locator("#hamburger").click(); time.sleep(0.6)
        nb = page.locator(".st-key-nav").bounding_box()
        log(f"fold: sidebar {nb['width']}px, labels under icons {page.locator('.st-key-nav_home p').evaluate('e => getComputedStyle(e).fontSize')} | playlists hidden "
            f"{not page.locator('.st-key-navlists').is_visible()} | aria-expanded {top.locator('#hamburger').get_attribute('aria-expanded')}")
        shot(page, "19_sidebar_folded")
        page.reload(); page.wait_for_selector(SELECTOR["home"], timeout=180000); top = named_frame(page, "topbar"); time.sleep(1.5)
        log(f"   kept after a reload: {page.evaluate('document.documentElement.classList.contains(`mt-nav-mini`)')} | page after a reload: Home {page.locator(SELECTOR['home']).count() == 1}")
        top.locator("#hamburger").click(); time.sleep(0.6)
        log(f"   unfold: sidebar {page.locator('.st-key-nav').bounding_box()['width']}px")
        page.set_viewport_size({"width": 390, "height": 844}); time.sleep(1.5)
        hidden = page.locator(".st-key-nav").evaluate("e => e.getBoundingClientRect().right <= 0 || getComputedStyle(e).visibility === 'hidden'")
        top.locator("#hamburger").click(); time.sleep(0.6)
        shown = page.locator(".st-key-nav").bounding_box()
        shot(page, "20_narrow_sidebar")
        page.mouse.click(370, 400); time.sleep(0.6)
        closed = not page.evaluate("document.documentElement.classList.contains('mt-nav-open')")
        top.locator("#hamburger").click(); time.sleep(0.5); page.locator(".st-key-nav_explore button").click()
        page.wait_for_selector(SELECTOR["explore"], timeout=30000); time.sleep(0.8)
        log(f"narrow (390 px): sidebar hidden {hidden} | hamburger slides it in at x={shown['x']} ({shown['width']}px) | tap outside closes {closed} "
            f"| picking Explore closes it {not page.evaluate('document.documentElement.classList.contains(`mt-nav-open`)')} and opens Explore "
            f"| no sideways scroll {page.evaluate('document.documentElement.scrollWidth') <= 390}")
        page.set_viewport_size({"width": 1440, "height": 900}); time.sleep(1)

        # --- profile name
        log("== PROFILE")
        prof = named_frame(page, "actions")
        icon0 = prof.locator("#avatar svg").count()
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-name").click(); time.sleep(0.3)
        page.locator("#mtd-input").fill("Rishika"); page.locator("#moodtune-dialog .mtd-ok").click(); time.sleep(0.4)
        letter = prof.locator("#avatar").inner_text()
        prof.locator("#avatar").click(); time.sleep(0.3); prof.locator("#m-name").click(); time.sleep(0.3)
        page.locator("#mtd-input").fill(""); page.locator("#moodtune-dialog .mtd-ok").click(); time.sleep(0.4)
        log(f"avatar: person icon at first {icon0 == 1} | name 'Rishika' -> {letter!r} | name removed -> icon {prof.locator('#avatar svg').count() == 1} "
            f"| stored {prof.evaluate('localStorage.getItem(`moodtune.profile`)')}")
        log(f"'Running' or a function name on the page at any point: {page.evaluate('window.__flagged.length')} | page errors {errors}")
        browser.close()
finally:
    stop(server)
log(f"image files created or changed on disk by the app: {new_images(before)}")
