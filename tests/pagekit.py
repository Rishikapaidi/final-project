"""Shared helpers for the headless-Edge tests: start the app, find its parts, read the mood card."""
import os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output"); os.makedirs(OUT, exist_ok=True)
PY = sys.executable
NEGATIVE = ["sad", "angry", "fear", "disgust"]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


def log(*a):
    print("RESULT", *a, flush=True)


def start_server(port, frame, name, previews="real", **env):
    """Start the real app with a fake camera (tests/fake_camera_server.py) and wait until it answers."""
    server = subprocess.Popen([PY, os.path.join(HERE, "fake_camera_server.py")],
                              env={**os.environ, "TEST_PORT": str(port), "TEST_FRAME": frame, "TEST_PREVIEWS": previews, **env},
                              stdout=open(os.path.join(OUT, f"server_{name}.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read():
                break
        except Exception:
            time.sleep(0.5)
    return server


def stop(server):
    server.terminate()
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)


def _frame_with(page, name):
    for f in page.frames:
        if f == page.main_frame:
            continue
        try:
            if f.evaluate(f"typeof window.{name}") == "object":
                return f
        except Exception:
            pass
    return None


def bar_frame(page, timeout=90):
    """The player bar's iframe (it holds window.player and the page's only <audio>)."""
    end = time.time() + timeout
    while time.time() < end:
        f = _frame_with(page, "player")
        if f:
            return f
        time.sleep(0.2)
    raise RuntimeError("player bar not found")


def path_frame(page, not_tracks=None, timeout=90):
    """The path card's iframe (window.playlist); with not_tracks, wait for one showing a different list."""
    end = time.time() + timeout
    while time.time() < end:
        f = _frame_with(page, "playlist")
        try:
            if f and (not_tracks is None or f.evaluate("DATA.tracks.map(t => t.lastfm)") != not_tracks):
                return f
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("path card not found")


def players(page):
    """(players on the page, players making sound)."""
    found = [f for f in page.frames if f != page.main_frame and _safe(f, "typeof window.player") == "object"]
    return len(found), sum(1 for f in found if _safe(f, "!window.player.audio.paused") is True)


def _safe(f, js):
    try:
        return f.evaluate(js)
    except Exception:
        return None


def player_state(bar):
    return bar.evaluate("(() => { const s = window.player.state(); return {current: s.current, paused: s.paused, time: +s.time.toFixed(1)}; })()")


def open_bars(page):
    """Open "See all 7 emotions" if it is closed (its bars are only in the page while it is open).

    Opening it happens in the browser only; it doesn't rerun the app.
    """
    if not page.locator("#legend").count():
        page.get_by_text("See all 7 emotions").click()
        page.wait_for_selector("#legend", timeout=10000)


def set_bars_open(page, want):
    """Open or close "See all 7 emotions"."""
    details = page.locator(".st-key-moodcard details")
    if (details.get_attribute("open") is not None) != want:
        page.get_by_text("See all 7 emotions").click(); time.sleep(0.8)


def legend(page):
    """The groups in "See all 7 emotions"."""
    open_bars(page)
    return [s.strip() for s in page.locator("#legend span").all_text_contents()]


def wait_for_text_bars(page, timeout=60):
    """Wait until the emotion bars include Text (the first sentence can also load the text model)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "Text" in legend(page):
                return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    return None


def read_card(page):
    """Everything the mood card shows, plus the expander's bars, read from the page."""
    js = r"""() => {
      const q = s => document.querySelector(s);
      const groups = {};
      document.querySelectorAll('#bars .erow').forEach(row => row.querySelectorAll('.bar').forEach(b => {
        (groups[b.dataset.group] = groups[b.dataset.group] || {})[row.dataset.emotion] = { value: +b.dataset.value, pct: b.querySelector('.pct').textContent,
          width: parseFloat(b.querySelector('.fill').style.width) }; }));
      return { emoji: q('#moodemoji').textContent, headline: q('#moodhead').textContent,
        share: +q('#gauge').dataset.share, pct: q('#gaugepct').textContent, band: q('#band').textContent, sentence: q('#bandtext').textContent,
        leans: [...document.querySelectorAll('.lean')].map(l => ({ source: l.dataset.source, share: +l.dataset.share, shown: l.querySelector('b').textContent })),
        blend: !!q('.blend'), emotions: [...document.querySelectorAll('#bars .erow')].map(r => r.dataset.emotion), groups,
        caption: q('#barcap').textContent };
    }"""
    open_bars(page)
    return page.evaluate(js)


def check_card(card):
    """Check the card against its own bars: gauge, band, leans, headline rule, bar labels and widths."""
    comb = {e: v["value"] for e, v in card["groups"]["combined"].items()}
    share = sum(comb[e] for e in NEGATIVE)     # the bars are rounded to 4 decimals
    pct = int(card["share"] * 100 + 1e-9)
    band = "Light" if card["share"] < 0.25 else "In between" if card["share"] < 0.5 else "Low"
    top, second = card["emotions"][0], card["emotions"][1]
    head = f"Mostly {top}" if comb[top] >= 0.5 else f"Mixed: {top} and {second}"
    leans = {l["source"]: l for l in card["leans"]}
    lean_ok = all(abs(l["share"] - sum(card["groups"][k][e]["value"] for e in NEGATIVE)) < 5e-4 and l["shown"] == f"{int(l['share'] * 100 + 1e-9)}%"
                  for k, l in leans.items())
    labels_ok = all(v["pct"] == f"{round(v['value'] * 100)}%" and abs(v["width"] - v["value"] * 100) < 0.06
                    for g in card["groups"].values() for v in g.values())
    return {
        "gauge = sum of combined sad+angry+fear+disgust": abs(card["share"] - share) < 5e-4,
        "gauge label": card["pct"] == f"{pct}%",
        "band": card["band"] == band,
        "leans match each source's bars": lean_ok and set(leans) == set(card["groups"]) - {"combined"},
        "blend note only with both": card["blend"] == (len(leans) == 2),
        "headline rule": card["headline"] == head,
        "bar labels and widths match the values": labels_ok,
        "emotions sorted by combined": list(comb.values()) == sorted(comb.values(), reverse=True),
    }


def describe_card(card):
    ok = check_card(card)
    leans = ", ".join(f"{l['source']} {l['shown']} ({l['share']:.4f})" for l in card["leans"]) or "none"
    return (f"{card['emoji']} {card['headline']!r} | gauge {card['pct']} low ({card['share']:.4f}) {card['band']} | leans {leans} | "
            f"blend note {card['blend']} | bars {list(card['groups'])} | "
            + ("all checks pass" if all(ok.values()) else "FAILED: " + ", ".join(k for k, v in ok.items() if not v)))


def full_shot(page, path, width=None):
    """A screenshot of the whole page: the window is made as tall as the page for the shot."""
    size = page.viewport_size
    width = width or size["width"]
    page.set_viewport_size({"width": width, "height": size["height"]}); time.sleep(0.8)
    height = page.evaluate("document.querySelector('[data-testid=stMain]').scrollHeight")
    page.set_viewport_size({"width": width, "height": max(height, 600)}); time.sleep(1.5)
    page.screenshot(path=path)
    page.set_viewport_size(size); time.sleep(0.8)


def image_files():
    """Image files under the project and the temp folder (to check the app writes none)."""
    found = {}
    for base in [ROOT, tempfile.gettempdir()]:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("venv", ".git", "node_modules") and os.path.join(dirpath, d) != OUT
                           and not d.startswith("playwright")]
            if dirpath.count(os.sep) - base.count(os.sep) > 4:
                dirnames[:] = []
            for f in filenames:
                if f.lower().endswith(IMAGE_EXT):
                    try:
                        found[os.path.join(dirpath, f)] = os.path.getmtime(os.path.join(dirpath, f))
                    except OSError:
                        pass
    return found


def new_images(before):
    after = image_files()
    return [p for p, m in after.items() if p not in before or m > before[p] + 1]
