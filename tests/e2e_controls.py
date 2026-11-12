"""Headless-Edge checks of the music-app controls, with a sad face (Uplift) and real cached previews.

Player bar: shuffle (never the 5-song path, locked until it has finished),
repeat off / all / one, seeking by click and drag, mute, the keyboard
shortcuts and the queue panel. Song list: the "..." menu (Play next, Add to
queue, Add to Liked songs, Add to playlist, Open on ..., Copy link), hearts,
and the top bar's search. Liked songs and playlists: the sidebar's counts,
the Library (play, Play all, remove), kept after a reload, cleared from
Settings, and still working when localStorage throws. The avatar's menu:
Settings and About. Every control is used while a song plays, and the music
must keep playing; settings must never rerun the app. Also saves the
screenshots of the queue panel, the Library, the song menu, Settings and About.
"""
import os, re, sys, time
from playwright.sync_api import sync_playwright
from pagekit import (OUT, actions_frame, bar_frame, count_reruns, image_files, log, named_frame, new_images, path_frame,
                     reruns, start_server, stop)

PORT = 8620
# Text that must never appear on the page: Streamlit's "Running <function>()" and the app's function names.
FLAG = r"Running|load_songs|load_face_model|load_text_model|summarise_scan|text_probabilities|build_queue|build_view|cache_resource|cache_data"
WATCH = """
window.__flagged = [];
const FLAG = new RegExp(%s);
const check = () => { const t = document.body ? document.body.innerText : ""; const m = t.match(FLAG);
  if (m) window.__flagged.push(t.slice(Math.max(0, m.index - 30), m.index + 50)); };
new MutationObserver(check).observe(document, { subtree: true, childList: true, characterData: true }); setInterval(check, 100);
""" % repr(FLAG)
# A browser where localStorage throws on every call (blocked site data).
BROKEN_STORAGE = """(() => { const no = () => { throw new DOMException("blocked", "SecurityError"); };
  Storage.prototype.getItem = no; Storage.prototype.setItem = no; Storage.prototype.removeItem = no; Storage.prototype.clear = no; })();"""


def shot(page, name):
    page.screenshot(path=os.path.join(OUT, f"page_{name}.png"))


def open_page(ctx):
    page = ctx.new_page()
    page.errors = []; page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.goto(f"http://localhost:{PORT}")
    page.wait_for_selector(".st-key-path iframe", timeout=180000)
    bar, path, top = bar_frame(page), path_frame(page), actions_frame(page)
    time.sleep(1.5)
    return page, bar, path, top


def to_middle(frame, locator):
    """Scroll a row to the middle of the window (the pinned player bar covers the bottom)."""
    locator.evaluate("e => e.scrollIntoView({block: 'center'})"); time.sleep(0.4)


def go(page, name):
    """Press a sidebar button (Home, Explore, Library) and wait for its page."""
    page.locator(f".st-key-nav_{name} button").click()
    page.wait_for_selector({"home": ".st-key-path iframe", "explore": ".st-key-explorelist iframe", "library": ".st-key-library iframe"}[name], timeout=30000)
    page.wait_for_timeout(1500)


def sidebar(page):
    """The sidebar's collections and their counts."""
    nav = named_frame(page, "navlists")
    return dict(zip(nav.locator(".item b").all_inner_texts(), nav.locator(".item small").all_inner_texts()))


before = image_files()
cam_log = os.path.join(OUT, "cam_controls.log")
if os.path.exists(cam_log): os.remove(cam_log)
server = start_server(PORT, "sad", "controls", TEST_CAM_LOG=cam_log)
try:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=f"http://localhost:{PORT}")
        ctx.add_init_script(WATCH)
        page, bar, path, top = open_page(ctx)
        state = lambda: bar.evaluate("window.player.state()")
        playing = lambda: not state()["paused"]
        order = lambda: [(o["base"], o["added"]) for o in state()["order"]]
        path_order = lambda: [o["base"] for o in state()["order"] if not o["added"] and o["base"] < state()["pathLen"]]
        list_order = lambda: [o["base"] for o in state()["order"] if not o["added"]]
        liked = lambda: top.evaluate("savedSongs().length")
        queue_id = bar.evaluate("DATA.id")
        count_reruns(page)
        s = state()
        log(f"=== Uplift queue: {len(s['order'])} songs, path of {s['pathLen']} | buttons in the bar: {bar.locator('button').evaluate_all('bs => bs.map(b => b.id)')}")

        # --- PLAYER BAR
        log("== PLAYER BAR")
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(2)
        sh = bar.locator("#shuffle")
        log(f"shuffle during the path: aria-disabled {sh.get_attribute('aria-disabled')} | tooltip {sh.get_attribute('title')!r}")
        sh.click(force=True); time.sleep(0.3)
        log(f"   clicked anyway: shuffle {state()['shuffle']} | order unchanged {list_order() == list(range(len(s['order'])))} | playing {playing()}")
        reps = []
        for _ in range(3):
            bar.locator("#repeat").click(); time.sleep(0.2); reps.append((state()["repeat"], bar.locator("#repeat").get_attribute("aria-label")))
        log(f"repeat button cycles: all -> {reps} | playing {playing()}")
        box = bar.locator("#track").bounding_box(); y = box["y"] + box["height"] / 2
        page.mouse.click(box["x"] + box["width"] * 0.5, y); time.sleep(0.4); t1 = state()["time"]
        page.mouse.move(box["x"] + box["width"] * 0.2, y); page.mouse.down(); page.mouse.move(box["x"] + box["width"] * 0.8, y, steps=6); page.mouse.up(); time.sleep(0.4)
        t2 = state()["time"]; dur = bar.evaluate("window.player.audio.duration")
        log(f"seek: click at 50% -> {t1:.1f}s of {dur:.1f} | drag to 80% -> {t2:.1f}s | labels {bar.locator('#tcur').inner_text()} / {bar.locator('#tdur').inner_text()} "
            f"| ok {abs(t1 - dur * 0.5) < 1.5 and abs(t2 - dur * 0.8) < 1.5} | playing {playing()}")
        rail = bar.evaluate("""() => { const r = id => document.getElementById(id).getBoundingClientRect(), cs = getComputedStyle(document.getElementById('track'), '::before');
          return { rail: cs.backgroundColor, fill: getComputedStyle(document.getElementById('fill')).backgroundColor,
                   between: r('tcur').right < r('track').left && r('track').right < r('tdur').left, gaps: [Math.round(r('track').left - r('tcur').right), Math.round(r('tdur').left - r('track').right)],
                   width: Math.round(r('track').width), fillShare: Math.round(r('fill').width / r('track').width * 100) }; }""")
        log(f"   the seek bar: rail {rail['rail']} (drawn {rail['rail'] not in ('rgba(0, 0, 0, 0)', 'transparent')}) | played part {rail['fill']} ({rail['fillShare']}%) "
            f"| {rail['width']}px between the two times {rail['between']} (gaps {rail['gaps']}px)")
        bar.locator("#mute").click(); m1 = (state()["muted"], bar.locator("#mute").get_attribute("aria-pressed")); bar.locator("#mute").click(); m2 = state()["muted"]
        log(f"mute button: muted {m1[0]} (aria-pressed {m1[1]}) -> unmuted {not m2} | playing {playing()}")

        log("keyboard (on the page):")
        page.locator(".greeting h1").click()
        page.keyboard.press("Space"); time.sleep(0.4); a = state()["paused"]; page.keyboard.press("Space"); time.sleep(0.8); b = state()["paused"]
        log(f"   Space pauses {a} and resumes {not b}")
        page.keyboard.press("ArrowRight"); time.sleep(1); r = state()["current"]; page.keyboard.press("ArrowLeft"); time.sleep(1); l = state()["current"]
        log(f"   Right -> song {r + 1}, Left -> song {l + 1} (expected 2 then 1: {(r, l) == (1, 0)})")
        page.keyboard.press("m"); m = state()["muted"]; page.keyboard.press("m")
        log(f"   M mutes {m} and unmutes {not state()['muted']}")
        saved0 = liked(); page.keyboard.press("l"); time.sleep(0.4); saved1 = liked()
        log(f"   L likes the song playing: {saved0} -> {saved1} | bar heart {bar.locator('#like').get_attribute('aria-pressed')} ({bar.locator('#like').get_attribute('title')!r})")
        page.keyboard.press("l"); time.sleep(0.3)
        box = page.get_by_label("How was your day?"); box.click(); before_keys = state()
        page.keyboard.type("m l "); page.keyboard.press("ArrowLeft"); time.sleep(0.4); after_keys = state()
        log(f"   typing 'm l ' + Left in the text box: box {box.input_value()!r} | muted {after_keys['muted']} | paused {after_keys['paused']} | same song {after_keys['current'] == before_keys['current']} "
            f"| liked {liked()}")
        box.fill(""); page.locator(".greeting h1").click()
        path.locator(".row").nth(2).click(); time.sleep(1); path.locator(".row").nth(2).press("Space"); time.sleep(0.4)
        log(f"   Space on the playing row (in the song list) pauses it: {state()['paused']} (song {state()['current'] + 1})")
        path.locator(".row").nth(2).press("Space"); time.sleep(0.8)

        # --- SONG MENU, QUEUE, SEARCH
        log("== SONG MENU, PLAY NEXT, ADD TO QUEUE")
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(1.5)
        lst, pth = list_order(), path_order()
        row = path.locator(".row").nth(8); to_middle(path, row); row.locator(".more").click(); time.sleep(0.3)
        items = path.locator("#menu .mi").all_inner_texts()
        log(f"menu on row 9: {items} | focus on {path.evaluate('document.activeElement.textContent')!r}")
        shot(page, "15_song_menu")
        path.locator("#menu .mi", has_text="Play next").click(); time.sleep(0.4); toast1 = page.locator("#moodtune-toast").inner_text()
        for i in (12, 3):
            row = path.locator(".row").nth(i); to_middle(path, row); row.locator(".more").click(); path.locator("#menu .mi", has_text="Add to queue").click(); time.sleep(0.3)
        log(f"Play next (row 9), then Add to queue (rows 13, 4): order starts {order()[:5]} | toast {toast1!r}")
        log(f"   list order unchanged {list_order() == lst} | path order unchanged {path_order() == pth} {path_order()} | rows in the song list still in list order "
            f"{path.evaluate('[...document.querySelectorAll(`.row`)].map(r => +r.dataset.index)') == list(range(len(lst)))} | playing {playing()}")
        bar.locator("#queuebtn").click(); time.sleep(0.6)
        log(f"queue panel: {bar.locator('.qhead').all_inner_texts()} | now {bar.locator('.qrow.now .title').inner_text()!r} | next {bar.locator('.qrow:not(.now) .title').all_inner_texts()[:4]} "
            f"| 'Added by you' {bar.locator('.byyou').count()} | remove buttons {bar.locator('.qrow .remove').count()} (only on added songs) | covers the window {bar.evaluate('window.frameElement.getBoundingClientRect().height')}")
        shot(page, "13_queue_panel")
        bar.locator(".qrow .remove").last.click(); time.sleep(0.3)
        log(f"   remove the last added song: added left {sum(a for _, a in order())} | playing {playing()}")
        page.keyboard.press("Escape"); time.sleep(0.4)
        log(f"   Escape closes it: {not state()['queueOpen']} | bar back to {bar.evaluate('window.frameElement.getBoundingClientRect().height')}px")
        bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s1 = state()
        bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s2 = state()
        log(f"next songs start by themselves: {s1['current'] + 1} (added: {s1['order'][s1['pos']]['added']}), then {s2['current'] + 1} (added: {s2['order'][s2['pos']]['added']}) "
            f"| added songs leave once played: {sum(o['added'] for o in s2['order'])} left | path order still {path_order()}")
        row = path.locator(".row").nth(2); to_middle(path, row); row.locator(".more").click(); path.locator("#menu .mi", has_text="Copy link").click(); time.sleep(0.6)
        copied = page.evaluate("navigator.clipboard.readText()")
        log(f"Copy link: clipboard = row 3's link {copied == path.evaluate('DATA.tracks[2].lastfm')} | toast {page.locator('#moodtune-toast').inner_text()!r}")
        row.locator(".more").click(); a = path.locator("#menu a.lastfm")
        log(f"Open on ...: {a.inner_text()!r} href = row 3's link {a.get_attribute('href') == path.evaluate('DATA.tracks[2].lastfm')} target {a.get_attribute('target')}")
        page.keyboard.press("Escape"); time.sleep(0.2)
        log(f"   Escape closes the menu {path.locator('#menu').is_hidden()} and returns focus to {path.evaluate('document.activeElement.className')!r}")
        last = path.locator(".row").last; to_middle(path, last); last.locator(".more").click(); time.sleep(0.3)
        mb, bb = path.locator("#menu").bounding_box(), last.locator(".more").bounding_box()
        log(f"   on the last row the menu opens above the button: {mb['y'] + mb['height'] <= bb['y']}"); page.keyboard.press("Escape")
        search = named_frame(page, "topbar").locator("#q")
        word = re.sub(r"[^a-z]", "", path.evaluate("DATA.tracks[9].title").lower())[:4]
        search.click(); search.type(word); time.sleep(0.4)
        shown = path.locator(".row:not([hidden]) .title").all_inner_texts()
        log(f"search (top bar) {word!r}: {len(shown)} of {path.locator('.row').count()} shown ({path.locator('#matches').inner_text()!r}) | all match "
            f"{all(word in (t + ' ' + a).lower() for t, a in zip(shown, path.locator('.row:not([hidden]) .artist').all_inner_texts()))} | Space typed into the box, music still playing {playing()}")
        search.fill("zzqqxx"); time.sleep(0.3); none = (path.locator(".row:not([hidden])").count(), path.locator("#nomatch").is_visible())
        search.fill(""); time.sleep(0.3)
        log(f"   no match -> {none[0]} rows and the note {none[1]} | cleared -> {path.locator('.row:not([hidden])').count()} rows")

        # --- SHUFFLE AFTER THE PATH, REPEAT AT THE END
        log("== SHUFFLE AND REPEAT")
        bar.evaluate(f"window.player.play({state()['pathLen']})"); time.sleep(1.5)   # the first song after the path
        before_shuffle = list_order(); played = state()["pos"]
        log(f"after the path (song {state()['current'] + 1}): shuffle enabled {bar.locator('#shuffle').get_attribute('aria-disabled') == 'false'} | tooltip {bar.locator('#shuffle').get_attribute('title')!r}")
        bar.locator("#shuffle").click(timeout=3000); time.sleep(0.3); after_shuffle = list_order()
        log(f"   shuffle on: path order unchanged {path_order() == list(range(state()['pathLen']))} | played songs unchanged {after_shuffle[:played + 1] == before_shuffle[:played + 1]} "
            f"| songs to come reordered {after_shuffle != before_shuffle} (same songs {sorted(after_shuffle) == sorted(before_shuffle)}) | playing {playing()}")
        row = path.locator(".row").nth(6); to_middle(path, row); row.locator(".more").click(); path.locator("#menu .mi", has_text="Play next").click(); time.sleep(0.3)
        log(f"   Play next while shuffled: next is song 7 {order()[state()['pos'] + 1] == (6, True)} | path order {path_order()}")
        bar.locator("#shuffle").click(); time.sleep(0.3)
        log(f"   shuffle off: songs to come back in list order {[b for b in list_order()[played + 1:]] == sorted(list_order()[played + 1:])}")
        while state()["repeat"] != "one": bar.locator("#repeat").click()
        cur = state()["current"]; bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s = state()
        log(f"repeat one: the same song starts again {s['current'] == cur and not s['paused'] and s['time'] < 5}")
        last_pos = len(state()["order"]) - 1
        while state()["repeat"] != "all": bar.locator("#repeat").click()
        bar.evaluate(f"window.player.playAt({last_pos})"); time.sleep(1.5)
        bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s = state()
        log(f"repeat all at the end of the list: starts again at song {s['current'] + 1} (loop start {bar.evaluate('DATA.loopFrom') + 1}, after the path) playing {not s['paused']}")
        while state()["repeat"] != "off": bar.locator("#repeat").click()
        bar.evaluate(f"window.player.playAt({len(state()['order']) - 1})"); time.sleep(1.5)
        bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s = state()
        log(f"repeat off at the end of the list: stops {s['paused']} on the last song {s['pos'] == len(s['order']) - 1}")
        while state()["repeat"] != "all": bar.locator("#repeat").click()

        # --- LIKED SONGS, PLAYLISTS AND THE LIBRARY
        log("== LIKED SONGS, PLAYLISTS AND THE LIBRARY")
        bar.evaluate("window.player.playAt(0)"); time.sleep(1.5)
        log(f"sidebar before: {sidebar(page)}")
        for i in (1, 6):
            row = path.locator(".row").nth(i); to_middle(path, row); row.locator(".heart").click(); time.sleep(0.3)
        row = path.locator(".row").nth(9); to_middle(path, row); row.locator(".more").click(); path.locator("#menu .mi", has_text="Add to playlist").click(); time.sleep(0.3)
        sub = path.locator("#menu .mi").all_inner_texts()
        path.locator("#menu .mi", has_text="My playlist").click(); time.sleep(0.3)
        log(f"2 hearts + Add to playlist > My playlist (submenu {sub}): sidebar {sidebar(page)} | filled hearts {path.locator('.heart[aria-pressed=true]').count()} "
            f"| toast {page.locator('#moodtune-toast').inner_text()!r}")
        stored = top.evaluate("({ liked: JSON.parse(localStorage.getItem('moodtune.saved')), lists: JSON.parse(localStorage.getItem('moodtune.playlists')), "
                              "recent: JSON.parse(localStorage.getItem('moodtune.recent')), keys: Object.keys(localStorage).sort() })")
        log(f"   stored in localStorage: liked {len(stored['liked'])} songs, fields {sorted(stored['liked'][0])} | playlists {[(p['name'], len(p['songs'])) for p in stored['lists']]}, "
            f"song fields {sorted(stored['lists'][0]['songs'][0])} | recently played {len(stored['recent'])}, fields {sorted(stored['recent'][0])} | keys {stored['keys']}")
        runs0, t0 = reruns(page), state()["time"]
        named_frame(page, "navlists").locator(".item", has_text="Liked songs").click()
        page.wait_for_selector(".st-key-library iframe", timeout=30000); lib = named_frame(page, "library"); time.sleep(1.5)
        log(f"sidebar > Liked songs: Library open on {lib.evaluate('window.library.current()')!r} | rows {lib.locator('.row .title').all_inner_texts()} "
            f"| same queue {bar.evaluate('DATA.id') == queue_id} | playing {playing()} ({t0:.1f}s -> {state()['time']:.1f}s)")
        shot(page, "14_library")
        lib.locator(".row").nth(1).locator(".pill").click(); time.sleep(1.5); s = state()
        log(f"play liked song 2: now {bar.locator('#nowtitle').inner_text()!r} (added {s['order'][s['pos']]['added']}) | row marked playing {lib.locator('.row.playing .title').all_inner_texts()} | path order {path_order()}")
        lib.locator("#vplay").click(); time.sleep(1.5); s = state()
        log(f"Play all: now {bar.locator('#nowtitle').inner_text()!r} | added songs in the queue {sum(o['added'] for o in s['order'])} | list order unchanged {list_order() == sorted(list_order())}")
        lib.locator("#back").click(); time.sleep(0.4); lib.locator(".coll", has_text="My playlist").click(); time.sleep(0.5)
        n0 = lib.locator(".row").count()
        lib.locator(".row").nth(0).locator(".more").click(); time.sleep(0.2); lib.locator("#menu .mi", has_text="Remove from My playlist").click(); time.sleep(0.4)
        log(f"My playlist: {n0} song(s) -> Remove from My playlist -> {lib.locator('.row').count()} | empty note {lib.locator('#empty').inner_text()!r} | sidebar {sidebar(page)}")
        go(page, "home"); path = path_frame(page)
        log(f"back on Home: same queue {bar.evaluate('DATA.id') == queue_id} | playing {playing()} | hearts filled {path.locator('.heart[aria-pressed=true]').count()} | app reruns for the trip {reruns(page) - runs0}")

        # --- SETTINGS AND ABOUT (the avatar's menu)
        log("== SETTINGS")
        bar.evaluate("window.player.playAt(1)"); time.sleep(1.5)
        runs0, t0 = reruns(page), state()["time"]
        top.locator("#avatar").click(); time.sleep(0.4)
        log(f"avatar menu: {[t.strip() for t in top.locator('#pmenu .pi').all_inner_texts()]} | covers the window {page.locator('.st-key-topactions iframe').bounding_box()['height']}px")
        top.locator("#m-settings").click(); time.sleep(0.5)
        log(f"settings: {top.locator('#dtitle').inner_text()!r} | defaults {top.locator('input[role=switch]').evaluate_all('xs => xs.map(x => x.id + `=` + x.checked)')} repeat {top.locator('input[name=repeat]:checked').get_attribute('value')}")
        shot(page, "16_settings")
        top.locator("label.setting", has_text="Autoplay").click(); time.sleep(0.3)
        top.locator(".seg label", has_text="This song").click(); time.sleep(0.3); rep = state()["repeat"]
        top.locator("label.setting", has_text="Show all 7").click(); time.sleep(0.6)
        exp = page.locator(".st-key-moodcard details").get_attribute("open") is not None
        top.locator("label.setting", has_text="Larger text").click(); time.sleep(0.6)
        large = page.evaluate("[document.documentElement.classList.contains('moodtune-large'), getComputedStyle(document.querySelector('.st-key-hero')).zoom]")
        s = state()
        log(f"   autoplay off, repeat 'This song' (player now {rep}), emotions shown ({exp}), larger text {large}")
        log(f"   music through 4 settings: playing {not s['paused']}, same song {s['current'] == 1}, time {t0:.1f} -> {s['time']:.1f} | same queue {bar.evaluate('DATA.id') == queue_id} | app reruns {reruns(page) - runs0}")
        top.locator("#openabout").click(); time.sleep(0.4)
        about = top.locator("#v-about").inner_text()
        log(f"About: {top.locator('#dtitle').inner_text()!r} | sections {top.locator('#v-about h3').all_inner_texts()} | mentions: "
            + ", ".join(f"{k} {k in about}" for k in ("Muse", "gauthamvijayaraj", "Apache 2.0", "iTunes", "30-second", "63.2%", "FER2013", "Fear and sad", "as labelled",
                                                      "Meet me here", "never saved", "only for this visit", "without times", "this browser")))
        shot(page, "17_about")
        top.locator("#back").click(); time.sleep(0.3); log(f"   back -> {top.locator('#dtitle').inner_text()!r}")
        page.keyboard.press("Escape"); time.sleep(0.4)
        wide = page.evaluate("[document.querySelector('[data-testid=stMain]').scrollWidth, innerWidth]")
        log(f"larger text: no sideways scroll {wide[0] <= wide[1]}")
        bar.locator("#repeat").click(); time.sleep(0.2)
        log(f"repeat button after the setting: {state()['repeat']}")
        cur = state()["current"]; bar.evaluate("window.player.audio.currentTime = window.player.audio.duration - 1"); time.sleep(3); s = state()
        log(f"autoplay off: stops at the end of the song {s['paused']} (same song {s['current'] == cur})")
        page.reload(); page.wait_for_selector(".st-key-path iframe", timeout=180000)
        bar, path, top = bar_frame(page), path_frame(page), actions_frame(page); time.sleep(2.5)
        log(f"after a reload: liked songs kept {liked()} | sidebar {sidebar(page)} | repeat {state()['repeat']} | emotions open "
            f"{page.locator('.st-key-moodcard details').get_attribute('open') is not None} | larger text {page.evaluate('document.documentElement.classList.contains(`moodtune-large`)')}")
        top.locator("#avatar").click(); time.sleep(0.3); top.locator("#m-settings").click(); time.sleep(0.4)
        top.locator("#s-clear").click(); time.sleep(0.2); confirm = top.locator("#confirmtext").inner_text()
        top.locator("#confirmyes").click(); time.sleep(0.3)
        log(f"Settings > Clear liked songs: {confirm!r} -> {liked()} liked | {top.locator('#s-count').inner_text()!r} | sidebar {sidebar(page)}")
        for name in ("Autoplay", "Show all 7", "Larger text"):
            top.locator("label.setting", has_text=name).click(); time.sleep(0.2)
        top.locator(".seg label", has_text="All").click(); time.sleep(0.2)
        log(f"   settings back to the defaults: {top.evaluate('JSON.stringify(window.actions.settings())')}")
        page.keyboard.press("Escape"); time.sleep(0.3)
        log(f"'Running' or a function name on the page at any point: {page.evaluate('window.__flagged.length')} | page errors {page.errors}")

        # --- NARROW SCREEN
        log("== NARROW SCREEN (390 px)")
        page.set_viewport_size({"width": 390, "height": 844}); time.sleep(1.5)
        wide = page.evaluate("[document.documentElement.scrollWidth, document.querySelector('[data-testid=stMain]').scrollWidth, innerWidth]")
        visible = bar.locator("#bar button").evaluate_all("bs => bs.filter(b => b.offsetParent).map(b => b.id)")
        log(f"no sideways scroll {max(wide[:2]) <= wide[2]} | bar buttons shown {visible} | avatar shown {top.locator('#avatar').is_visible()}")
        top.locator("#avatar").click(); time.sleep(0.3); top.locator("#m-settings").click(); time.sleep(0.5)
        log(f"   Settings drawer fits: {top.locator('#drawer').bounding_box()['width']} px wide"); page.keyboard.press("Escape"); time.sleep(0.3)
        ctx.close()

        # --- localStorage throws
        log("== localStorage THROWS ON EVERY CALL")
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}); ctx.add_init_script(BROKEN_STORAGE)
        page, bar, path, top = open_page(ctx)
        log(f"storage blocked in the components: {path.evaluate('(() => { try { localStorage.getItem(`x`); return false; } catch (e) { return true; } })()')}")
        path.locator(".row").nth(0).locator(".pill").click(); time.sleep(1.5)
        for i in (1, 2):
            row = path.locator(".row").nth(i); to_middle(path, row); row.locator(".heart").click(); time.sleep(0.3)
        page.locator(".greeting h1").click(); page.keyboard.press("l"); time.sleep(0.3)
        log(f"2 hearts + L: sidebar {sidebar(page)} | hearts filled {path.locator('.heart[aria-pressed=true]').count()}")
        named_frame(page, "navlists").locator(".item", has_text="Liked songs").click()
        page.wait_for_selector(".st-key-library iframe", timeout=30000); lib = named_frame(page, "library"); time.sleep(1.5)
        n = lib.locator(".row").count(); lib.locator(".row").nth(1).locator(".pill").click(); time.sleep(1.5)
        log(f"   the Library (after a page change) lists {n} liked songs | play one: now {bar.locator('#nowtitle').inner_text()!r}, playing {not bar.evaluate('window.player.audio.paused')}")
        top.locator("#avatar").click(); time.sleep(0.3); top.locator("#m-settings").click(); time.sleep(0.3); top.locator("label.setting", has_text="Autoplay").click(); time.sleep(0.3)
        log(f"   a setting still applies (in memory): autoplay {top.evaluate('window.actions.settings().autoplay')}")
        top.locator("label.setting", has_text="Autoplay").click(); page.keyboard.press("Escape")
        page.reload(); page.wait_for_selector(".st-key-playerbar iframe", timeout=180000); top = actions_frame(page); time.sleep(1.5)
        log(f"   after a reload the in-memory copy is gone (as expected): {top.evaluate('savedSongs().length')} songs | page errors {page.errors}")
        ctx.close()
        browser.close()
finally:
    stop(server)
cam = open(cam_log).read().count("opened") if os.path.exists(cam_log) else "?"
log(f"camera opened {cam} time(s) (once per page load) | image files created or changed on disk by the app: {new_images(before)}")
