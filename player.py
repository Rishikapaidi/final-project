"""Build the play queue and render the two HTML/JS components that play it.

The queue is a list of tracks with a playable 30-second preview (see
previews.py). Streamlit reruns the script on every interaction, which would
stop any music started from a Streamlit widget, so playback lives in the
browser:

- render_path(): the path card (Uplift) or the track list (Resonance), with a
  Play pill and a link per song, and the mood-journey chart.
- render_bar(): the player bar pinned to the bottom of the window, which holds
  the page's only <audio> element: play, pause, next/previous, volume and
  "play the next song when this one ends".

Both are st.components.v1.html iframes. The list sends commands to the bar,
and the bar sends back what is playing, over a BroadcastChannel named after
the queue's id, so a new queue never talks to an old player.
"""
import json

import numpy as np
import pandas as pd

import previews
import uplift

RESONANCE_TRACKS = 30
# After the Uplift journey, this many more songs from the bucket it ended in.
CONTINUE_TRACKS = 15
# Share of the queue per detected emotion (most likely first), by how many
# emotions were detected. Same proportions the app has always used.
SAMPLE_SIZES = {
    1: [30],
    2: [30, 20],
    3: [55, 20, 15],
    4: [30, 29, 18, 9],
}
DEFAULT_SAMPLE_SIZES = [10, 7, 6, 5, 2]
# Unchecked songs are looked up this many at a time, at most this many rounds.
LOOKUP_BATCH = 12
LOOKUP_ROUNDS = 6


def _keys(rows):
    return list(zip(rows.lastfm_url, rows.title, rows.artist))


def _track(song, preview, **extra):
    # The link is Last.fm for the English (Muse) songs and Spotify for the others;
    # artwork is iTunes', or the song file's own (Spotify) artwork if iTunes has none.
    track = {"title": song.title, "artist": song.artist, "lastfm": song.lastfm_url, "bucket": song.bucket,
             "link_label": getattr(song, "link_source", "") or "Last.fm", "language": getattr(song, "language", "") or "",
             "valence": round(float(song.valence_tags), 2), "arousal": round(float(song.arousal_tags), 2),
             "preview": preview["preview_url"], "art": preview["artwork_url"] or getattr(song, "artwork_url", "")}
    track.update(extra)
    return track


def playable_songs(rows, wanted, rng):
    """Up to `wanted` rows of `rows` that have a preview, in random order.

    Songs already known to have a preview are used first. With live lookups
    on (previews.LIVE_LOOKUPS), unchecked songs are looked up if those are not
    enough; otherwise only cached songs are used.
    """
    rows = rows.iloc[rng.permutation(len(rows))]
    known = [previews.cached(url) for url in rows.lastfm_url]
    ready = [(song, row) for song, row in zip(rows.itertuples(), known) if row and row["preview_url"]]
    unchecked = rows[[row is None for row in known]]
    for round_ in range(LOOKUP_ROUNDS):
        if len(ready) >= wanted or not previews.LIVE_LOOKUPS or previews.rate_limited():
            break
        batch = unchecked.iloc[round_ * LOOKUP_BATCH:(round_ + 1) * LOOKUP_BATCH]
        if batch.empty:
            break
        found = previews.resolve(_keys(batch))
        ready += [(song, found[song.lastfm_url]) for song in batch.itertuples() if song.lastfm_url in found]
    return ready[:wanted]


def resonance_queue(song_buckets, emotions, rng=None):
    """Tracks matching the detected emotions, mixed in their usual proportions."""
    rng = rng or np.random.default_rng()
    sizes = SAMPLE_SIZES.get(len(emotions), DEFAULT_SAMPLE_SIZES)[:len(emotions)]
    tracks = []
    for emotion, size in zip(emotions, sizes):
        wanted = max(1, round(RESONANCE_TRACKS * size / sum(sizes)))
        tracks += [_track(song, preview, section="mood")
                   for song, preview in playable_songs(song_buckets[emotion], wanted, rng)]
    tracks = [tracks[i] for i in rng.permutation(len(tracks))][:RESONANCE_TRACKS]
    for number, track in enumerate(tracks, 1):
        track["number"] = number
    return tracks


def uplift_queue(songs, fused, emotions, rng=None, pool=None):
    """(tracks, journey) for an Uplift playlist, or (None, None) if Uplift does not apply.

    The journey's songs come first (section "journey"), followed by more songs
    from the bucket it ended in (section "more"). pool: optional boolean array,
    one per row of `songs`, limiting which songs may be picked (e.g. one
    language); bucket centres still come from the English songs.
    """
    rng = rng or np.random.default_rng()
    pool = np.ones(len(songs), dtype=bool) if pool is None else np.asarray(pool, dtype=bool)

    def playable(positions):
        found = previews.resolve(_keys(songs.iloc[positions]))
        return np.array([p for p in positions if songs.lastfm_url.iloc[p] in found], dtype=int)

    if previews.LIVE_LOOKUPS:
        # Check the nearest candidates of each waypoint, looking up unseen songs.
        playlist, journey = uplift.build_playlist(songs, fused, emotions, rng, playable=playable, pool=pool)
    else:
        # Cache-only: pick among songs already known to have a preview.
        known = songs.lastfm_url.isin(previews.playable_urls()).values
        playlist, journey = uplift.build_playlist(songs, fused, emotions, rng, pool=pool & known)
    if playlist is None:
        return None, None
    tracks = [_track(song, previews.cached(song.lastfm_url), section="journey", number=song.step,
                     waypoint=[round(float(song.waypoint_valence), 2), round(float(song.waypoint_arousal), 2)])
              for song in playlist.itertuples()]
    if tracks:
        # Keep the music going in the mood the journey reached (happy after
        # sad, neutral after angry, fear or disgust).
        final_bucket = journey["targets"][-1]
        rest = songs[pool & (songs.bucket == final_bucket).values & ~songs.lastfm_url.isin(playlist.lastfm_url).values]
        tracks += [_track(song, preview, section="more", number=len(playlist) + i)
                   for i, (song, preview) in enumerate(playable_songs(rest, CONTINUE_TRACKS, rng), 1)]
    return tracks, journey


def journey_summary(journey):
    """The parts of an Uplift journey the player needs for its chart."""
    return {"start": journey["start_emotion"], "targets": journey["targets"],
            "waypoints": [[round(float(v), 2), round(float(a), 2)] for v, a in journey["waypoints"]]}




# The journey's start and end, in words, for the Uplift card's title.
START_WORDS = {"sad": "heavy", "angry": "tense", "fear": "uneasy", "disgust": "unsettled"}
END_WORDS = {"happy": "hopeful", "neutral": "calm"}


def path_title(journey):
    """"From heavy to hopeful", from an Uplift journey's start emotion and last target."""
    start = START_WORDS.get(journey["start"], journey["start"])
    end = END_WORDS.get(journey["targets"][-1], journey["targets"][-1])
    return f"From {start} to {end}"


def _payload(queue, **extra):
    data = {"id": queue["id"], "tracks": queue["tracks"], "loopFrom": queue["loop_from"], **extra}
    return json.dumps(data, ensure_ascii=True).replace("</", "<\\/")


def render_path(queue, message):
    """The path card (Uplift) or the track list (Resonance), as one HTML page.

    queue: from app.build_queue; message: the mood card's sentence about the
    playlist. All text is inserted as text, never as HTML. The page plays
    nothing itself: rows, Play pills and each row's "..." menu (Play next,
    Add to queue, Open on Spotify / Last.fm, Copy link) send commands to the
    player bar (render_bar) over a BroadcastChannel named after the queue id.
    A search box filters the list by title or artist, in the browser.
    """
    title = path_title(queue["journey"]) if queue["mode"] == "Uplift" else "Songs for this moment"
    payload = _payload(queue, mode=queue["mode"], title=title, message=message, subtitle=queue["subtitle"],
                       note=queue["note"] or "", journey=queue["journey"])
    return PATH_HTML.replace("__FONTS__", FONT_CSS).replace("__DATA__", payload)


def render_bar(queue):
    """The player bar (the only <audio> on the page), as one HTML page.

    app.py pins it to the bottom of the window with CSS. Controls: shuffle
    (only the songs still to come, never Uplift's path, and only once the path
    has finished), previous / play / next, repeat (off / all / one), a
    progress bar to click or drag, the queue panel, mute and volume, and the
    keyboard shortcuts. When a song ends the next one starts; with repeat
    "all" the queue loops from queue["loop_from"] (after an Uplift journey,
    the "keep listening" songs).
    """
    return BAR_HTML.replace("__FONTS__", FONT_CSS).replace("__DATA__", _payload(queue, mode=queue["mode"]))


# The bundled fonts (static/fonts, served at app/static/). The component pages
# are srcdoc iframes, so relative URLs resolve against the app's own address.
FONT_CSS = """
  @font-face { font-family:"Fraunces"; src:url("app/static/fonts/Fraunces-latin-variable.woff2") format("woff2"); font-weight:100 900; font-display:swap; }
  @font-face { font-family:"Inter"; src:url("app/static/fonts/Inter-latin-variable.woff2") format("woff2"); font-weight:100 900; font-display:swap; }
  :root { --cream:#f6f3ee; --card:#ffffff; --ink:#2b2840; --muted:#625e78; --line:#e7e1d8; --purple:#2d2a52;
          --lavender:#b9b0e8; --lav-soft:#eeebfa; --terra:#b85c38; --terra-soft:#f8e9e1; --terra-ink:#7a3a20;
          --serif:"Fraunces", Georgia, "Times New Roman", serif;
          --sans:"Inter", "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif; }
"""

# Shared by the scripts of all three pages (the song list, the player bar and
# the top bar's buttons).
BASE_JS = r"""
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const PLAY = '<svg class="f" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 1.8v12.4L14 8z"/></svg>';
const PAUSE = '<svg class="f" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 1.8h3.6v12.4H3zM9.4 1.8H13v12.4H9.4z"/></svg>';
// Line icons (24 x 24, drawn with the current text colour).
const ICONS = {
  shuffle: '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>',
  repeat: '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
  repeatOne: '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/><path d="M11 10.5l1.5-1v5"/>',
  queue: '<line x1="9" y1="6" x2="21" y2="6"/><line x1="9" y1="12" x2="21" y2="12"/><line x1="9" y1="18" x2="16" y2="18"/><polyline points="3 5 6 7 3 9"/>',
  volume: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>',
  muted: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>',
  close: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  more: '<circle cx="5" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="19" cy="12" r="1.2"/>',
  search: '<circle cx="11" cy="11" r="7"/><line x1="20.5" y1="20.5" x2="16" y2="16"/>',
  heart: '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
  play: '<polygon points="7 4 20 12 7 20 7 4"/>',
  back: '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
};
const icon = name => '<svg class="s" viewBox="0 0 24 24" aria-hidden="true">' + ICONS[name] + '</svg>';
function artNode(track, cls) {
  if (track.art) { const img = el("img", cls); img.src = track.art; img.alt = ""; img.loading = "lazy"; return img; }
  const box = el("div", cls + " noart", "♫"); box.setAttribute("aria-hidden", "true"); return box;
}
// Keyboard shortcuts: Space = play/pause, Right/Left = next/previous, M = mute,
// L = save the song playing to My playlist (or remove it).
// Nothing happens while typing in a text box, or with a modifier key held;
// Space on a button, link or row is left to that control.
function shortcut(e) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return null;
  const t = e.target || {}, tag = (t.tagName || "").toLowerCase(), role = t.getAttribute ? t.getAttribute("role") || "" : "";
  if (t.isContentEditable || tag === "input" || tag === "textarea" || tag === "select" || role === "combobox" || role === "textbox") return null;
  const key = { " ": "space", ArrowRight: "next", ArrowLeft: "prev", m: "mute", M: "mute", l: "like", L: "like" }[e.key] || null;
  if (key === "space" && (tag === "button" || tag === "a" || tag === "summary" || role === "button" || role === "menuitem")) return null;
  if ((key === "next" || key === "prev") && role === "slider") return null;
  if (key && key !== "next" && key !== "prev" && e.repeat) return "held";
  return key;
}
// A short message at the bottom of the page (in the page itself, so it shows
// wherever the song list is scrolled to). The CSS animation fades it out.
function toast(text) {
  let doc = document;
  try { doc = window.parent.document; } catch (e) {}
  let box = doc.getElementById("moodtune-toast");
  if (!box) { box = doc.createElement("div"); box.id = "moodtune-toast"; box.setAttribute("role", "status"); box.setAttribute("aria-live", "polite"); doc.body.appendChild(box); }
  box.textContent = text; box.classList.remove("show"); void box.offsetWidth; box.classList.add("show");
}

// Saved songs (My playlist) and settings live only in this browser:
// localStorage when it works, otherwise an in-memory copy kept on the page
// (shared by the three components until the page is reloaded). Nothing here
// is ever sent to the server. "hub" tells the other components something changed.
const hub = new BroadcastChannel("moodtune");
function pageMemory() {
  try { const page = window.parent; return page.__moodtuneMemory || (page.__moodtuneMemory = {}); }
  catch (e) { return window.__moodtuneMemory || (window.__moodtuneMemory = {}); }
}
function load(key, fallback) {
  let raw = null;
  try { raw = localStorage.getItem(key); } catch (e) {}
  if (raw == null) raw = pageMemory()[key];
  if (raw == null) return fallback;
  try { return JSON.parse(raw); } catch (e) { return fallback; }
}
function save(key, value) {
  const raw = JSON.stringify(value);
  pageMemory()[key] = raw;
  try { localStorage.setItem(key, raw); } catch (e) {}
}
const SAVED = "moodtune.saved";
const linkLabel = link => (/spotify\.com/.test(link || "") ? "Spotify" : "Last.fm");
function savedSongs() { const list = load(SAVED, []); return Array.isArray(list) ? list.filter(s => s && s.link && s.preview) : []; }
const isSaved = link => savedSongs().some(s => s.link === link);
let onSavedChange = () => {};   // each page redraws its hearts / list
function setSaved(list) { save(SAVED, list); hub.postMessage({ type: "saved" }); onSavedChange(); }
// Only these six fields are kept for a saved song.
function saveSong(t) {
  if (isSaved(t.lastfm)) return;
  setSaved(savedSongs().concat([{ title: t.title, artist: t.artist, link: t.lastfm, preview: t.preview, art: t.art || "", language: t.language || "" }]));
}
function unsaveSong(link) { setSaved(savedSongs().filter(s => s.link !== link)); }
function toggleSaved(t) {
  if (isSaved(t.lastfm)) { unsaveSong(t.lastfm); toast("Removed “" + t.title + "” from My playlist"); return false; }
  saveSong(t); toast("Saved “" + t.title + "” to My playlist"); return true;
}
const songTrack = s => ({ title: s.title, artist: s.artist, lastfm: s.link, link_label: linkLabel(s.link), preview: s.preview, art: s.art, language: s.language, bucket: "" });
// Settings (the gear in the top bar). None of them changes the queue, so they
// never rerun the app: each component reads them here.
const SETTINGS = "moodtune.settings";
const DEFAULTS = { autoplay: true, repeat: "all", emotions: false, large: false };
function settings() {
  const s = load(SETTINGS, {}), out = Object.assign({}, DEFAULTS);
  if (s && typeof s === "object") for (const k in DEFAULTS) if (typeof s[k] === typeof DEFAULTS[k]) out[k] = s[k];
  if (!["off", "all", "one"].includes(out.repeat)) out.repeat = DEFAULTS.repeat;
  return out;
}
let onSettingsChange = () => {};
function setSetting(key, value) { const s = settings(); s[key] = value; save(SETTINGS, s); hub.postMessage({ type: "settings" }); onSettingsChange(); }
hub.addEventListener("message", e => {
  if (e.data && e.data.type === "saved") onSavedChange();
  if (e.data && e.data.type === "settings") onSettingsChange();
});
window.addEventListener("storage", e => { if (e.key === SAVED) onSavedChange(); if (e.key === SETTINGS) onSettingsChange(); });
"""

# The queue's two pages (song list and player bar) also get the queue and the
# channel named after it.
COMMON_JS = r"""
const DATA = __DATA__;
const tracks = DATA.tracks;
const channel = new BroadcastChannel("moodtune:" + DATA.id);
""" + BASE_JS

PATH_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
__FONTS__
  * { box-sizing:border-box; }
  html, body { margin:0; background:transparent; color:var(--ink); font-family:var(--sans); overflow:hidden; }
  #card { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:26px 26px 18px;
          box-shadow:0 1px 2px rgba(45,42,82,.05), 0 8px 26px rgba(45,42,82,.07); margin:2px 2px 14px; }
  .kicker { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 6px; }
  .head { display:flex; align-items:flex-start; gap:18px; justify-content:space-between; }
  h2 { font-family:var(--serif); font-weight:600; font-size:30px; line-height:1.15; margin:0; letter-spacing:-.01em; }
  .msg { margin:10px 0 0; font-size:15px; line-height:1.55; color:var(--ink); max-width:68ch; }
  .sub { margin:6px 0 0; font-size:13px; color:var(--muted); }
  .note { margin:14px 0 0; font-size:13.5px; line-height:1.5; color:var(--ink); background:var(--lav-soft); border-radius:12px; padding:10px 14px; }
  #bigplay { width:52px; height:52px; border-radius:50%; border:none; background:var(--purple); cursor:pointer; flex:none;
             display:flex; align-items:center; justify-content:center; box-shadow:0 6px 16px rgba(45,42,82,.25); transition:transform .12s; }
  #bigplay:hover { transform:scale(1.05); } #bigplay svg { width:20px; height:20px; fill:#fff; }
  svg.s { fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
  #bigplay:focus-visible, .pill:focus-visible, .lastfm:focus-visible, .row:focus-visible, summary:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  /* Uplift: the warm terracotta panel holding the journey */
  .uplift #panel { background:var(--terra-soft); border-radius:18px; padding:22px 22px 12px; margin:-6px -6px 6px; }
  .uplift .kicker { color:var(--terra-ink); }
  .uplift h2 { color:var(--terra-ink); }
  .uplift #bigplay { background:var(--terra); box-shadow:0 6px 16px rgba(184,92,56,.3); }
  .steps { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 0; padding:0; list-style:none; }
  .steps li { font-size:12.5px; color:var(--terra-ink); background:rgba(255,255,255,.7); border-radius:999px; padding:3px 11px; text-transform:capitalize; }
  .steps li.arrow { background:none; padding:3px 0; }
  .section { color:var(--muted); font-size:12px; letter-spacing:.1em; text-transform:uppercase; font-weight:600; margin:20px 6px 6px; }
  .uplift #panel .section { color:var(--terra-ink); margin-top:16px; }
  .row { display:grid; grid-template-columns:30px 48px minmax(0,1fr) auto auto auto auto; align-items:center; gap:10px;
         padding:8px 8px 8px 6px; border-radius:12px; cursor:pointer; }
  .row[hidden], .section[hidden] { display:none !important; }
  .icon-btn { width:34px; height:34px; border:none; background:none; border-radius:999px; color:var(--muted); cursor:pointer;
              display:flex; align-items:center; justify-content:center; padding:0; }
  .icon-btn:hover, .icon-btn[aria-expanded="true"] { color:var(--purple); background:rgba(45,42,82,.07); }
  .icon-btn svg { width:19px; height:19px; } .icon-btn:focus-visible { outline:3px solid var(--lavender); outline-offset:1px; }
  .heart[aria-pressed="true"] { color:var(--terra); } .heart[aria-pressed="true"] svg { fill:currentColor; }
  /* search */
  .tools { display:flex; align-items:center; gap:12px; margin:16px 0 0; }
  .search { flex:1; max-width:380px; display:flex; align-items:center; gap:8px; background:#fff; border:1px solid var(--line);
            border-radius:999px; padding:0 14px; color:var(--muted); }
  .search:focus-within { border-color:var(--lavender); box-shadow:0 0 0 3px rgba(185,176,232,.35); }
  .search svg { width:16px; height:16px; flex:none; }
  #search { flex:1; min-width:0; border:none; outline:none; background:none; font:inherit; font-size:14px; color:var(--ink); padding:9px 0; }
  #matches { font-size:12.5px; color:var(--muted); }
  #nomatch { color:var(--muted); font-size:14px; padding:14px 8px 6px; }
  /* the song menu */
  #menu { position:absolute; z-index:5; min-width:220px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:6px;
          box-shadow:0 14px 40px rgba(45,42,82,.2); }
  #menu .mi { display:block; width:100%; text-align:left; border:none; background:none; font:inherit; font-size:14px; color:var(--ink);
              padding:9px 12px; border-radius:9px; cursor:pointer; text-decoration:none; }
  #menu .mi:hover, #menu .mi:focus { background:var(--lav-soft); outline:none; }
  #menu hr { border:none; border-top:1px solid var(--line); margin:5px 4px; }
  .row:hover { background:#f4f1fb; } .uplift #panel .row:hover { background:rgba(255,255,255,.6); }
  .row.playing { background:var(--lav-soft); } .uplift #panel .row.playing { background:#fff; }
  .num { text-align:center; color:var(--muted); font-size:14.5px; font-variant-numeric:tabular-nums; }
  .row.playing .num, .row.playing .title { color:var(--purple); font-weight:600; }
  .uplift #panel .row.playing .title, .uplift #panel .row.playing .num { color:var(--terra-ink); }
  .art { width:48px; height:48px; border-radius:10px; background:#ebe6dc; object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:20px; }
  .meta { min-width:0; } .title { font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:13px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  .tag { font-size:11.5px; color:var(--muted); border:1px solid var(--line); background:#fff; border-radius:999px; padding:2px 9px; white-space:nowrap; text-transform:capitalize; }
  .pill { border:none; background:var(--purple); color:#fff; font-weight:600; font-size:12.5px; border-radius:999px; font-family:var(--sans);
          padding:6px 14px; cursor:pointer; white-space:nowrap; min-width:76px; }
  .pill:hover { background:#433e78; }
  .uplift #panel .pill { background:var(--terra); } .uplift #panel .pill:hover { background:#a04d2d; }
  .eq { display:inline-flex; gap:2px; align-items:flex-end; height:14px; }
  .eq i { width:3px; background:currentColor; animation:eq .9s ease-in-out infinite; }
  .eq i:nth-child(2) { animation-delay:.25s; } .eq i:nth-child(3) { animation-delay:.5s; }
  .paused .eq i { animation-play-state:paused; }
  @keyframes eq { 0%,100% { height:4px; } 50% { height:14px; } }
  details.journey { background:rgba(255,255,255,.75); border-radius:12px; margin:14px 0 6px; padding:0 14px; }
  details.journey summary { cursor:pointer; padding:11px 0; font-size:13.5px; font-weight:600; color:var(--terra-ink); }
  details.journey .cap { color:var(--muted); font-size:12px; margin:0 0 12px; }
  .empty { color:var(--muted); padding:30px 8px; }
  @media (max-width: 640px) {
    #card { padding:20px 14px 12px; border-radius:18px; } h2 { font-size:25px; }
    .uplift #panel { padding:18px 12px 10px; margin:-4px -4px 6px; }
    .row { grid-template-columns:22px 44px minmax(0,1fr) auto auto auto; gap:6px; padding:8px 2px; }
    .row .tag { display:none; } .art { width:44px; height:44px; }
    .pill { min-width:0; padding:6px 11px; } .search { max-width:none; }
  }
</style></head>
<body>
<div id="card">
  <div id="panel">
    <p class="kicker" id="kicker"></p>
    <div class="head"><div><h2 id="title"></h2><p class="msg" id="msg"></p><p class="sub" id="qsub"></p></div>
      <button id="bigplay" title="Play" aria-label="Play"></button></div>
    <p class="note" id="note" hidden></p>
    <ul class="steps" id="steps" hidden></ul>
    <div class="tools" id="tools"><label class="search"><span id="searchicon"></span>
      <input id="search" type="search" placeholder="Search this list" aria-label="Search this list by title or artist" autocomplete="off"></label>
      <span id="matches" aria-live="polite"></span></div>
    <div id="journeyrows"></div>
    <div id="journey"></div>
  </div>
  <div id="rows"></div>
  <div id="nomatch" hidden>No songs in this list match your search.</div>
</div>
<div id="menu" role="menu" hidden></div>
<script>
__COMMON__
const SECTIONS = { journey: "Your path, in play order", more: "Keep listening", mood: "" };
let current = -1, paused = true;
const send = msg => channel.postMessage(msg);

function row(t, i) {
  const r = el("div", "row"); r.dataset.index = i; r.dataset.section = t.section; r.tabIndex = 0; r.setAttribute("role", "button");
  r.setAttribute("aria-label", "Play " + t.title + " by " + t.artist);
  r.appendChild(el("div", "num", String(t.number)));
  r.appendChild(artNode(t, "art"));
  const meta = el("div", "meta"); meta.appendChild(el("div", "title", t.title)); meta.appendChild(el("div", "artist", t.artist)); r.appendChild(meta);
  r.appendChild(el("span", "tag", t.bucket));
  const heart = el("button", "icon-btn heart"); heart.innerHTML = icon("heart"); heart.dataset.index = i;
  heart.addEventListener("click", e => { e.stopPropagation(); toggleSaved(t); });
  heart.addEventListener("keydown", e => e.stopPropagation()); r.appendChild(heart);
  const pill = el("button", "pill", "▶ Play"); pill.title = "Play this song";
  pill.addEventListener("click", e => { e.stopPropagation(); send(i === current ? { type: "toggle" } : { type: "play", index: i }); }); r.appendChild(pill);
  const more = el("button", "icon-btn more"); more.innerHTML = icon("more");
  more.title = "More options"; more.ariaLabel = "More options for " + t.title; more.setAttribute("aria-haspopup", "menu"); more.setAttribute("aria-expanded", "false");
  more.addEventListener("click", e => { e.stopPropagation(); if (menuFor === i) closeMenu(true); else openMenu(i, more); });
  more.addEventListener("keydown", e => e.stopPropagation()); r.appendChild(more);
  r.addEventListener("click", () => send({ type: "play", index: i }));
  r.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); send(i === current ? { type: "toggle" } : { type: "play", index: i }); } });
  return r;
}

function build() {
  const uplift = DATA.mode === "Uplift";
  document.body.classList.toggle("uplift", uplift);
  $("kicker").textContent = uplift ? "Your path" : "Your songs";
  $("title").textContent = DATA.title; $("msg").textContent = DATA.message || ""; $("qsub").textContent = DATA.subtitle;
  if (DATA.note) { $("note").textContent = DATA.note; $("note").hidden = false; }
  if (!tracks.length) { $("rows").appendChild(el("div", "empty", "No playable songs were found for this mood.")); $("bigplay").hidden = true; $("tools").hidden = true; return; }
  if (uplift && DATA.journey) {
    const steps = [DATA.journey.start].concat(DATA.journey.targets);
    steps.forEach((s, k) => { if (k) { const a = el("li", "arrow", "→"); a.setAttribute("aria-hidden", "true"); $("steps").appendChild(a); } $("steps").appendChild(el("li", null, s)); });
    $("steps").hidden = false;
  }
  let section = null;
  tracks.forEach((t, i) => {
    const target = uplift && t.section === "journey" ? $("journeyrows") : $("rows");
    if (t.section !== section) {
      section = t.section;
      let label = SECTIONS[section];
      if (section === "more") label += " · " + t.bucket;
      if (label) { const head = el("div", "section", label); head.dataset.section = section; target.appendChild(head); }
    }
    target.appendChild(row(t, i));
  });
  buildJourney();
}

function buildJourney() {
  const j = DATA.journey; if (!j) return;
  const songs = tracks.filter(t => t.section === "journey");
  const box = el("details", "journey"); box.appendChild(el("summary", null, "Mood journey: " + [j.start].concat(j.targets).join(" → ")));
  const W = 520, H = 250, P = { l: 52, r: 18, t: 14, b: 40 };
  const pts = j.waypoints.concat(songs.map(s => [s.valence, s.arousal]));
  const pad = (lo, hi) => { const d = (hi - lo) * 0.15 || 0.5; return [lo - d, hi + d]; };
  const [x0, x1] = pad(Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0])));
  const [y0, y1] = pad(Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1])));
  const X = v => P.l + (v - x0) / (x1 - x0) * (W - P.l - P.r), Y = a => H - P.b - (a - y0) / (y1 - y0) * (H - P.t - P.b);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%"); svg.style.maxWidth = W + "px"; svg.id = "chart";
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Chart of the planned mood path and the songs picked along it");
  const add = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.appendChild(n); return n; };
  add("line", { x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b, stroke: "#d9d1c5" }); add("line", { x1: P.l, y1: P.t, x2: P.l, y2: H - P.b, stroke: "#d9d1c5" });
  add("text", { x: (P.l + W - P.r) / 2, y: H - 8, fill: "#625e78", "font-size": 12, "text-anchor": "middle" }, "valence (negative → positive)");
  add("text", { x: 14, y: (P.t + H - P.b) / 2, fill: "#625e78", "font-size": 12, "text-anchor": "middle", transform: `rotate(-90 14 ${(P.t + H - P.b) / 2})` }, "arousal (calm → energetic)");
  [x0, x1].forEach((v, k) => add("text", { x: k ? W - P.r : P.l, y: H - P.b + 15, fill: "#8a8599", "font-size": 10, "text-anchor": k ? "end" : "start" }, v.toFixed(1)));
  [y0, y1].forEach((v, k) => add("text", { x: P.l - 6, y: k ? P.t + 9 : H - P.b, fill: "#8a8599", "font-size": 10, "text-anchor": "end" }, v.toFixed(1)));
  add("polyline", { points: j.waypoints.map(p => X(p[0]) + "," + Y(p[1])).join(" "), fill: "none", stroke: "#a39cc4", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
  add("polyline", { points: songs.map(s => X(s.valence) + "," + Y(s.arousal)).join(" "), fill: "none", stroke: "#b85c38", "stroke-width": 2 });
  songs.forEach(s => { add("circle", { cx: X(s.valence), cy: Y(s.arousal), r: 11, fill: "#b85c38" }); add("text", { x: X(s.valence), y: Y(s.arousal) + 4, fill: "#fff", "font-size": 12, "font-weight": 700, "text-anchor": "middle" }, String(s.number)); });
  box.appendChild(svg);
  box.appendChild(el("p", "cap", "Dashed line: the planned path. Solid line: the songs picked, in play order. Values come from listeners' mood tags."));
  $("journey").appendChild(box);
}

// The "..." menu. Play next and Add to queue insert the song after the one
// playing (the player bar keeps the list's order and Uplift's path as they are).
let menuFor = -1, menuButton = null;
function openMenu(i, button) {
  closeMenu(false);
  const t = tracks[i], menu = $("menu"), label = t.link_label || "Last.fm";
  menu.replaceChildren();
  const item = (text, action, href) => {
    const b = el(href ? "a" : "button", "mi", text); b.setAttribute("role", "menuitem"); b.tabIndex = -1;
    if (href) { b.href = href; b.target = "_blank"; b.rel = "noopener noreferrer"; b.classList.add("lastfm"); b.addEventListener("click", () => closeMenu(false)); }
    else b.addEventListener("click", () => { closeMenu(true); action(); });
    menu.appendChild(b);
  };
  item("Play next", () => { send({ type: "insert", index: i, where: "next" }); toast("“" + t.title + "” will play next"); });
  item("Add to queue", () => { send({ type: "insert", index: i, where: "queue" }); toast("Added “" + t.title + "” to the queue"); });
  item(isSaved(t.lastfm) ? "Remove from My playlist" : "Add to My playlist", () => toggleSaved(t));
  menu.appendChild(el("hr"));
  item("Open on " + label, null, t.lastfm);
  item("Copy link", () => copyLink(t.lastfm));
  const roomBelow = document.body.scrollHeight;
  menu.hidden = false;
  const r = button.getBoundingClientRect(), h = menu.offsetHeight;
  // open below the button, or above it near the bottom of the card
  menu.style.top = (r.bottom + 6 + h > roomBelow - 4 ? r.top - h - 6 : r.bottom + 6) + "px";
  menu.style.left = Math.max(8, r.right - menu.offsetWidth) + "px";
  menuFor = i; menuButton = button; button.setAttribute("aria-expanded", "true");
  menu.querySelector(".mi").focus();
}
function closeMenu(refocus) {
  if (menuFor < 0) return;
  $("menu").hidden = true; menuButton.setAttribute("aria-expanded", "false");
  if (refocus) menuButton.focus();
  menuFor = -1; menuButton = null;
}
$("menu").addEventListener("keydown", e => {
  const items = [...$("menu").querySelectorAll(".mi")], k = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); items[(k + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length].focus(); }
  else if (e.key === "Escape") { e.preventDefault(); closeMenu(true); }
  else if (e.key === "Tab") closeMenu(false);
  e.stopPropagation();
});
document.addEventListener("click", e => { if (menuFor >= 0 && !$("menu").contains(e.target)) closeMenu(false); });
async function copyLink(link) {
  let ok = false;
  try { await navigator.clipboard.writeText(link); ok = true; }
  catch (e) {
    try { const box = el("textarea"); box.value = link; box.style.cssText = "position:fixed;opacity:0"; document.body.appendChild(box); box.select(); ok = document.execCommand("copy"); box.remove(); } catch (e2) {}
  }
  toast(ok ? "Link copied" : "Couldn't copy the link");
}

// Search: filters the list by title or artist, as you type (nothing is sent anywhere).
const norm = s => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLocaleLowerCase();
function filter() {
  const q = norm($("search").value.trim());
  let shown = 0;
  document.querySelectorAll(".row").forEach(r => {
    const t = tracks[+r.dataset.index], hit = !q || norm(t.title + " " + t.artist).includes(q);
    r.hidden = !hit; if (hit) shown++;
  });
  document.querySelectorAll(".section").forEach(s => { s.hidden = !!q && !document.querySelector(`.row[data-section="${s.dataset.section}"]:not([hidden])`); });
  $("matches").textContent = q ? shown + " of " + tracks.length : "";
  $("nomatch").hidden = !q || shown > 0;
  closeMenu(false);
}
$("search").addEventListener("input", filter);
$("searchicon").outerHTML = icon("search");

// Hearts: filled for songs in My playlist (kept in step with the other components).
function refreshHearts() {
  document.querySelectorAll(".heart").forEach(h => {
    const t = tracks[+h.dataset.index], on = isSaved(t.lastfm);
    h.setAttribute("aria-pressed", on); h.title = on ? "Remove from My playlist" : "Add to My playlist";
    h.ariaLabel = (on ? "Remove " : "Add ") + t.title + (on ? " from My playlist" : " to My playlist");
  });
}
onSavedChange = refreshHearts;

function refresh() {
  const playing = current >= 0 && !paused;
  document.querySelectorAll(".row").forEach(r => {
    const i = +r.dataset.index, on = i === current;
    r.classList.toggle("playing", on); r.classList.toggle("paused", on && paused);
    const num = r.querySelector(".num");
    if (on) num.innerHTML = '<span class="eq" aria-hidden="true"><i></i><i></i><i></i></span>'; else num.textContent = String(tracks[i].number);
    r.querySelector(".pill").textContent = on && playing ? "❚❚ Pause" : "▶ Play";
  });
  $("bigplay").innerHTML = playing ? PAUSE : PLAY;
  $("bigplay").title = $("bigplay").ariaLabel = playing ? "Pause" : "Play";
}

// Grow the iframe to fit its content, so the page scrolls instead of the card.
function fit() { if (window.frameElement) window.frameElement.style.height = Math.ceil(document.documentElement.scrollHeight) + "px"; }
new ResizeObserver(fit).observe(document.body);

channel.onmessage = e => { if (e.data.type === "state") { current = e.data.current; paused = e.data.paused; refresh(); } };
$("bigplay").addEventListener("click", () => send({ type: "toggle" }));
// Shortcut keys pressed in this card go to the player bar.
document.addEventListener("keydown", e => { const key = shortcut(e); if (!key) return; e.preventDefault(); if (key !== "held") send({ type: "key", key }); });
build(); refresh(); refreshHearts(); fit();
send({ type: "hello" });
window.playlist = { state: () => ({ current, paused }) };
</script>
</body></html>
""".replace("__COMMON__", COMMON_JS)

BAR_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
__FONTS__
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; overflow:hidden; background:transparent; color:var(--ink); font-family:var(--sans); }
  /* While the queue is open the iframe fills the window: the shade (a click on it closes the queue) and the panel sit above the bar. */
  #shade { position:absolute; left:0; right:0; top:0; bottom:86px; display:none; background:rgba(43,40,64,.16); }
  body.queue-open #shade { display:block; }
  #bar { position:absolute; left:0; right:0; bottom:0; height:86px; background:rgba(255,253,249,.97); border-top:1px solid var(--line);
         box-shadow:0 -6px 24px rgba(45,42,82,.08); display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.4fr) minmax(0,1fr);
         align-items:center; gap:18px; padding:0 22px; }
  #now { display:flex; align-items:center; gap:12px; min-width:0; }
  .art { width:52px; height:52px; border-radius:10px; background:#ebe6dc; object-fit:cover; flex:none; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:20px; }
  .meta { min-width:0; } .title { font-size:14.5px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:12.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  #center { display:flex; flex-direction:column; align-items:center; gap:5px; min-width:0; }
  .controls { display:flex; align-items:center; gap:14px; }
  button { background:none; border:none; cursor:pointer; padding:7px; display:flex; align-items:center; justify-content:center;
           border-radius:999px; color:var(--muted); position:relative; font-family:var(--sans); }
  button:hover { color:var(--purple); background:rgba(45,42,82,.06); }
  button svg { width:18px; height:18px; }
  svg.f { fill:currentColor; }
  svg.s { fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
  button.on { color:var(--purple); }
  button.on::after { content:""; position:absolute; bottom:-2px; left:50%; width:4px; height:4px; margin-left:-2px; border-radius:50%; background:var(--purple); }
  button[aria-disabled="true"] { opacity:.35; cursor:not-allowed; }
  button[aria-disabled="true"]:hover { background:none; color:var(--muted); }
  button:focus-visible, input:focus-visible, #track:focus-visible, .qrow:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  #toggle { width:42px; height:42px; background:var(--purple); color:#fff; } #toggle:hover { background:#433e78; color:#fff; }
  #toggle svg { width:16px; height:16px; }
  .progress { display:flex; align-items:center; gap:10px; width:100%; font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
  #track { flex:1; height:16px; cursor:pointer; position:relative; touch-action:none; display:flex; align-items:center; }
  #track::before { content:""; position:absolute; left:0; right:0; height:5px; background:#e5e0f2; border-radius:3px; }
  #fill { position:relative; height:5px; width:0; background:var(--purple); border-radius:3px; }
  #knob { position:absolute; top:2px; left:0; width:12px; height:12px; margin-left:-6px; border-radius:50%; background:var(--purple); opacity:0; transition:opacity .15s; }
  #track:hover #knob, #track.drag #knob, #track:focus-visible #knob { opacity:1; }
  #right { display:flex; align-items:center; justify-content:flex-end; gap:6px; }
  #volume { width:104px; accent-color:var(--purple); }
  #queuebtn.on::after { display:none; } #queuebtn.on { background:var(--lav-soft); }
  #like { flex:none; } #like[aria-pressed="true"] { color:var(--terra); } #like[aria-pressed="true"] svg { fill:currentColor; }
  #like[hidden] { display:none; }
  /* the queue panel */
  #queue { position:absolute; right:16px; bottom:98px; width:min(400px, calc(100% - 32px)); max-height:calc(100% - 118px); display:none; flex-direction:column;
           background:var(--card); border:1px solid var(--line); border-radius:18px; box-shadow:0 18px 50px rgba(45,42,82,.22); overflow:hidden; }
  body.queue-open #queue { display:flex; }
  #queue header { display:flex; align-items:center; justify-content:space-between; padding:16px 12px 6px 20px; }
  #queue h2 { font-family:var(--serif); font-weight:600; font-size:22px; margin:0; }
  #qlist { overflow:auto; padding:0 10px 14px; }
  .qhead { font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:12px 10px 6px; }
  .qrow { display:grid; grid-template-columns:40px minmax(0,1fr) auto; align-items:center; gap:12px; padding:6px 8px; border-radius:10px; cursor:pointer; }
  .qrow:hover { background:#f4f1fb; } .qrow.now { background:var(--lav-soft); cursor:default; }
  .qrow .art { width:40px; height:40px; border-radius:8px; }
  .qrow .title { font-size:14px; font-weight:500; } .qrow .artist { font-size:12px; }
  .byyou { display:inline-block; font-size:11px; color:var(--purple); background:var(--lav-soft); border-radius:999px; padding:1px 8px; margin-top:4px; }
  .qrow.now .byyou { background:#fff; }
  .qempty { color:var(--muted); font-size:13.5px; padding:4px 10px 8px; margin:0; }
  @media (max-width: 760px) {
    #bar { grid-template-columns:minmax(0,1fr) auto auto; gap:4px; padding:0 8px 0 12px; }
    .art { width:44px; height:44px; }
    .controls { gap:0; } #toggle { margin:0 2px; }
    /* the progress bar becomes a thin line along the top of the bar (still click/drag to seek) */
    .progress { position:absolute; left:0; right:0; top:-7px; } #tcur, #tdur, #volume, #mute { display:none; }
  }
</style></head>
<body>
<div id="shade"></div>
<section id="queue" role="dialog" aria-label="Queue">
  <header><h2>Queue</h2><button id="qclose" title="Close the queue" aria-label="Close the queue"></button></header>
  <div id="qlist"></div>
</section>
<div id="bar">
  <div id="now"><div class="art noart" id="nowart" aria-hidden="true">&#9835;</div>
    <div class="meta"><div class="title" id="nowtitle">Nothing playing</div><div class="artist" id="nowartist">Press play or pick a song</div></div>
    <button id="like" aria-pressed="false" hidden></button></div>
  <div id="center">
    <div class="controls">
      <button id="shuffle" aria-label="Shuffle" aria-pressed="false"></button>
      <button id="prev" title="Previous" aria-label="Previous"><svg class="f" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2h2v12H3zM14 2v12L6 8z"/></svg></button>
      <button id="toggle" title="Play" aria-label="Play"></button>
      <button id="next" title="Next" aria-label="Next"><svg class="f" viewBox="0 0 16 16" aria-hidden="true"><path d="M11 2h2v12h-2zM2 2l8 6-8 6z"/></svg></button>
      <button id="repeat"></button>
    </div>
    <div class="progress"><span id="tcur">0:00</span>
      <div id="track" role="slider" aria-label="Position in the song" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" tabindex="0"><div id="fill"></div><div id="knob"></div></div>
      <span id="tdur">0:30</span></div>
  </div>
  <div id="right">
    <button id="queuebtn" title="Queue" aria-label="Queue" aria-expanded="false" aria-controls="queue"></button>
    <button id="mute" aria-label="Mute" aria-pressed="false"></button>
    <input id="volume" type="range" min="0" max="1" step="0.05" value="0.8" aria-label="Volume"></div>
</div>
<script>
__COMMON__
const audio = new Audio();
audio.preload = "auto";
const BAR_HEIGHT = 86;
const REPEAT_TITLES = { off: "Repeat: off", all: "Repeat: all", one: "Repeat: this song" };
// Uplift's path songs (the first pathLen songs of the list) are never shuffled.
const pathLen = DATA.mode === "Uplift" ? tracks.filter(t => t.section === "journey").length : 0;
// The play order: the list's songs (base = their place in the list), plus
// songs added with "Play next" / "Add to queue" (added = true), which leave
// the order once they have been played.
let order = tracks.map((t, i) => ({ t, base: i, added: false }));
let repeatDefault = settings().repeat;
let pos = -1, failures = 0, shuffle = false, repeat = repeatDefault, pathDone = pathLen === 0, queueOpen = false;
const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
$("qclose").innerHTML = icon("close"); $("queuebtn").innerHTML = icon("queue"); $("shuffle").innerHTML = icon("shuffle");

// The volume is a per-browser convenience; the page works without storage.
try { const v = parseFloat(localStorage.getItem("moodtune.volume")); if (v >= 0 && v <= 1) $("volume").value = v; } catch (e) {}
audio.volume = +$("volume").value;
$("volume").addEventListener("input", () => {
  audio.volume = +$("volume").value; if (audio.volume > 0) audio.muted = false;
  try { localStorage.setItem("moodtune.volume", $("volume").value); } catch (e) {}
  refreshVolume();
});
function setMuted(on) { audio.muted = on; refreshVolume(); }
function refreshVolume() {
  const off = audio.muted || audio.volume === 0;
  $("mute").innerHTML = icon(off ? "muted" : "volume"); $("mute").title = audio.muted ? "Unmute (M)" : "Mute (M)";
  $("mute").setAttribute("aria-pressed", audio.muted);
}

const now = () => (pos >= 0 ? order[pos] : null);
function lastPathPos() { let p = -1; order.forEach((e, k) => { if (!e.added && e.base < pathLen) p = k; }); return p; }
function announce() { const c = now(); channel.postMessage({ type: "state", current: c ? c.base : -1, paused: audio.paused }); }

function refresh() {
  const c = now(), playing = c && !audio.paused;
  $("toggle").innerHTML = playing ? PAUSE : PLAY;
  $("toggle").title = $("toggle").ariaLabel = playing ? "Pause" : "Play";
  if (c) {
    $("nowtitle").textContent = c.t.title; $("nowartist").textContent = c.t.artist;
    const old = $("nowart"), art = artNode(c.t, "art"); art.id = "nowart"; old.replaceWith(art);
  }
  refreshLike();
  const sh = $("shuffle");
  sh.classList.toggle("on", shuffle); sh.setAttribute("aria-pressed", shuffle); sh.setAttribute("aria-disabled", !pathDone);
  sh.title = !pathDone ? "Shuffle turns on once the " + pathLen + "-song path has finished" : shuffle ? "Shuffle: on" : "Shuffle: off";
  const rp = $("repeat");
  rp.innerHTML = icon(repeat === "one" ? "repeatOne" : "repeat"); rp.classList.toggle("on", repeat !== "off");
  rp.title = rp.ariaLabel = REPEAT_TITLES[repeat];
  $("queuebtn").classList.toggle("on", queueOpen);
  if (queueOpen) renderQueue();
  announce();
}

function playAt(p) {
  if (!order.length) return;
  p = Math.max(0, Math.min(p, order.length - 1));
  // an added song leaves the order once it has been played
  if (pos >= 0 && p !== pos && order[pos].added) { order.splice(pos, 1); if (p > pos) p--; }
  pos = p;
  if (pos > lastPathPos()) pathDone = true;
  audio.src = order[pos].t.preview;
  audio.play().catch(() => {});
  refresh();
}
function playBase(i) { const p = order.findIndex(e => !e.added && e.base === i); if (p >= 0) playAt(p); }
function loopStart() { const p = order.findIndex(e => !e.added && e.base === DATA.loopFrom); return p < 0 ? 0 : p; }
// auto: the song ended by itself. Repeat "off" then stops at the end of the
// queue; the Next button always goes on (from the start of the loop).
function next(auto) {
  if (pos < 0) return playAt(0);
  if (pos + 1 < order.length) return playAt(pos + 1);
  if (auto && repeat === "off") { audio.pause(); audio.currentTime = 0; return refresh(); }
  playAt(loopStart());
}
function prev() {
  if (pos < 0) return playAt(0);
  if (audio.currentTime > 3 || pos === 0) { audio.currentTime = 0; audio.play().catch(() => {}); } else playAt(pos - 1);
}
function toggle() { if (pos < 0) return playAt(0); if (audio.paused) audio.play().catch(() => {}); else audio.pause(); }

// Shuffle reorders only the songs still to come, and never the path's songs
// or the songs added by the listener (they keep their places).
function setShuffle(on) {
  if (on && !pathDone) return false;
  shuffle = on;
  const slots = [];
  for (let k = pos + 1; k < order.length; k++) if (!order[k].added && order[k].base >= pathLen) slots.push(k);
  const items = slots.map(k => order[k]);
  if (on) for (let k = items.length - 1; k > 0; k--) { const j = Math.floor(Math.random() * (k + 1)); [items[k], items[j]] = [items[j], items[k]]; }
  else items.sort((a, b) => a.base - b.base);
  slots.forEach((k, n) => { order[k] = items[n]; });
  refresh();
  return true;
}
function cycleRepeat() { repeat = { off: "all", all: "one", one: "off" }[repeat]; refresh(); }

// Insert a song after the current one: "next" right after it, "queue" after
// the songs already added. The list's own order is never changed.
function insert(t, base, where) {
  let p = pos + 1;
  if (where === "queue") while (p < order.length && order[p].added) p++;
  order.splice(p, 0, { t, base, added: true });
  refresh();
  return p;
}
function removeAt(p) { if (p > pos && order[p] && order[p].added) { order.splice(p, 1); refresh(); } }

// The queue panel: while it is open the iframe itself is pinned to the whole
// window (Streamlit's wrapper keeps the bar's height, so growing inside it
// would push the panel off the bottom of the screen).
function openQueue(on) {
  queueOpen = on;
  document.body.classList.toggle("queue-open", on);
  $("queuebtn").setAttribute("aria-expanded", on);
  const f = window.frameElement;
  if (f) Object.assign(f.style, on ? { position: "fixed", left: "0", bottom: "0", width: "100%", height: "100vh", zIndex: "1001" }
                                   : { position: "", left: "", bottom: "", width: "", height: BAR_HEIGHT + "px", zIndex: "" });
  refresh();
  (on ? $("qclose") : $("queuebtn")).focus();
}
function qrow(e, p, isNow) {
  const r = el("div", "qrow" + (isNow ? " now" : ""));
  r.appendChild(artNode(e.t, "art"));
  const meta = el("div", "meta"); meta.appendChild(el("div", "title", e.t.title)); meta.appendChild(el("div", "artist", e.t.artist));
  if (e.added) meta.appendChild(el("span", "byyou", "Added by you"));
  r.appendChild(meta);
  if (e.added && !isNow) {
    const x = el("button", "remove"); x.innerHTML = icon("close"); x.title = "Remove from the queue"; x.ariaLabel = "Remove " + e.t.title + " from the queue";
    x.addEventListener("click", ev => { ev.stopPropagation(); removeAt(p); }); r.appendChild(x);
  }
  if (!isNow) {
    r.tabIndex = 0; r.setAttribute("role", "button"); r.ariaLabel = "Play " + e.t.title + " by " + e.t.artist;
    r.addEventListener("click", () => playAt(p));
    r.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); playAt(p); } });
  }
  return r;
}
function renderQueue() {
  const list = $("qlist"), keep = list.scrollTop;
  list.replaceChildren();
  list.appendChild(el("div", "qhead", "Now playing"));
  if (pos >= 0) list.appendChild(qrow(order[pos], pos, true)); else list.appendChild(el("p", "qempty", "Nothing yet. Press play or pick a song."));
  const rest = order.length - pos - 1;
  list.appendChild(el("div", "qhead", "Next up" + (rest ? " · " + rest : "")));
  if (!rest) list.appendChild(el("p", "qempty", repeat === "off" ? "That's the end of the list." : "Then the list starts again."));
  for (let p = pos + 1; p < order.length; p++) list.appendChild(qrow(order[p], p, false));
  list.scrollTop = keep;
}

// The heart next to the song playing (and the L key).
function refreshLike() {
  const c = now(), on = !!c && isSaved(c.t.lastfm);
  $("like").hidden = !c; $("like").innerHTML = icon("heart"); $("like").setAttribute("aria-pressed", on);
  $("like").title = $("like").ariaLabel = on ? "Remove from My playlist (L)" : "Add to My playlist (L)";
}
function like() { const c = now(); if (c) toggleSaved(c.t); }
onSavedChange = refreshLike;
// Songs played from My playlist go in after the current song, like "Play
// next"; the first one starts straight away.
function playSongs(songs) {
  if (!songs || !songs.length) return;
  const p = pos + 1;
  songs.forEach((s, k) => order.splice(p + k, 0, { t: songTrack(s), base: tracks.findIndex(t => t.lastfm === s.link), added: true }));
  playAt(p);
}

function act(key) {
  if (key === "space") toggle(); else if (key === "next") next(false); else if (key === "prev") prev();
  else if (key === "mute") setMuted(!audio.muted); else if (key === "like") like();
}
function onKey(e) {
  if (e.key === "Escape" && queueOpen) { openQueue(false); return; }
  const key = shortcut(e);
  if (!key) return;
  e.preventDefault();
  if (key !== "held") act(key);
}
document.addEventListener("keydown", onKey);
// Keys pressed on the page itself (outside the song list and this bar): one
// listener on the page, replaced by each new player bar.
try {
  const page = window.parent;
  if (page.__moodtuneKeys) page.document.removeEventListener("keydown", page.__moodtuneKeys);
  page.__moodtuneKeys = e => { try { onKey(e); } catch (err) {} };
  page.document.addEventListener("keydown", page.__moodtuneKeys);
} catch (e) {}

audio.addEventListener("ended", () => {
  failures = 0;
  if (repeat === "one") { audio.currentTime = 0; audio.play().catch(() => {}); return; }
  if (!settings().autoplay) return refresh();   // "Autoplay next song" is off: stop here
  next(true);
});
// A new default repeat (Settings) also applies straight away.
onSettingsChange = () => {
  const d = settings().repeat;
  if (d !== repeatDefault) { repeatDefault = d; repeat = d; }
  refresh();
};
audio.addEventListener("error", () => { if (pos >= 0 && ++failures < order.length) next(true); });
audio.addEventListener("playing", () => { failures = 0; refresh(); });
audio.addEventListener("pause", refresh);
function showTime(t) {
  const d = audio.duration || 30, p = Math.min(100, Math.max(0, t / d * 100));
  $("fill").style.width = p + "%"; $("knob").style.left = p + "%"; $("track").setAttribute("aria-valuenow", Math.round(p));
  $("tcur").textContent = fmt(t); $("tdur").textContent = fmt(d);
}
audio.addEventListener("timeupdate", () => showTime(audio.currentTime));
audio.addEventListener("loadedmetadata", () => showTime(audio.currentTime));
// Click or drag the progress bar to seek.
function seekTo(x) {
  if (pos < 0 || !audio.duration) return;
  const r = $("track").getBoundingClientRect(), f = Math.min(1, Math.max(0, (x - r.left) / r.width));
  audio.currentTime = f * audio.duration; showTime(audio.currentTime);
}
$("track").addEventListener("pointerdown", e => { if (pos < 0) return; $("track").classList.add("drag"); $("track").setPointerCapture(e.pointerId); seekTo(e.clientX); });
$("track").addEventListener("pointermove", e => { if ($("track").classList.contains("drag")) seekTo(e.clientX); });
["pointerup", "pointercancel"].forEach(n => $("track").addEventListener(n, () => $("track").classList.remove("drag")));
$("track").addEventListener("keydown", e => { if (pos < 0 || !audio.duration) return; if (e.key === "ArrowRight") audio.currentTime += 5; if (e.key === "ArrowLeft") audio.currentTime -= 5; });
$("toggle").addEventListener("click", toggle);
$("next").addEventListener("click", () => next(false)); $("prev").addEventListener("click", prev);
$("shuffle").addEventListener("click", () => setShuffle(!shuffle));
$("repeat").addEventListener("click", cycleRepeat);
$("mute").addEventListener("click", () => setMuted(!audio.muted));
$("queuebtn").addEventListener("click", () => openQueue(!queueOpen));
$("like").addEventListener("click", like);
// From the top bar's component: keys pressed there, and "play from My playlist".
hub.addEventListener("message", e => {
  const m = e.data || {};
  if (m.type === "key") act(m.key); else if (m.type === "playSaved") playSongs(m.songs);
});
$("qclose").addEventListener("click", () => openQueue(false));
$("shade").addEventListener("click", () => openQueue(false));
channel.onmessage = e => {
  const m = e.data;
  if (m.type === "play") playBase(m.index); else if (m.type === "toggle") toggle(); else if (m.type === "hello") announce();
  else if (m.type === "key") act(m.key);
  else if (m.type === "insert" && tracks[m.index]) insert(tracks[m.index], m.index, m.where);
};
refreshVolume(); refresh();
window.player = {
  audio, play: playBase, playAt, toggle, next: () => next(false), prev, setShuffle, cycleRepeat, setMuted, openQueue, insert, removeAt, playSongs,
  state: () => ({ current: now() ? now().base : -1, pos, paused: audio.paused, time: audio.currentTime, src: audio.src, shuffle, repeat,
                  muted: audio.muted, pathDone, pathLen, queueOpen,
                  order: order.map(e => ({ base: e.base, added: e.added, title: e.t.title, link: e.t.lastfm })) }),
};
</script>
</body></html>
""".replace("__COMMON__", COMMON_JS)


def render_actions():
    """The top bar's buttons (My playlist, Settings) and the drawer they open, as one HTML page.

    It holds no queue data, so Streamlit never reloads it on a rerun. While
    the drawer is open the iframe is pinned over the whole window. Saved songs
    and settings are read from and written to the browser only (see BASE_JS);
    playing a saved song sends it to the player bar. Settings never rerun the
    app: autoplay and repeat are read by the player bar, "Larger text" puts a
    class on the page, and "Show all 7 emotions" opens the mood card's
    expander in the browser. The About view says what the app does, where
    its data comes from, its limits and what it keeps.
    """
    return ACTIONS_HTML.replace("__FONTS__", FONT_CSS)


ACTIONS_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
__FONTS__
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; overflow:hidden; background:transparent; color:var(--ink); font-family:var(--sans); }
  /* room around the buttons for their focus rings */
  #buttons { display:flex; align-items:center; justify-content:flex-end; gap:8px; height:50px; padding:5px 6px; }
  .top { height:40px; border:1px solid var(--line); background:#fff; border-radius:999px; display:flex; align-items:center; gap:8px;
         padding:0 14px; font:inherit; font-size:14px; color:var(--ink); cursor:pointer; white-space:nowrap; }
  .top:hover { border-color:var(--lavender); background:#fbfaff; }
  .top svg { width:18px; height:18px; flex:none; }
  .badge { min-width:20px; height:20px; border-radius:999px; background:var(--terra); color:#fff; font-size:11.5px; font-weight:700;
           display:inline-flex; align-items:center; justify-content:center; padding:0 6px; }
  .badge[hidden] { display:none; }
  button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  svg.s { fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
  /* narrow screens: icons only */
  @media (max-width: 170px) { .label { display:none; } .top { padding:0 11px; gap:6px; } }
  /* the drawer */
  #overlay { position:fixed; inset:0; }
  #overlay[hidden] { display:none; }
  #backdrop { position:absolute; inset:0; background:rgba(43,40,64,.38); }
  #drawer { position:absolute; top:0; right:0; bottom:0; width:min(440px, 100%); background:var(--cream); display:flex; flex-direction:column;
            box-shadow:-14px 0 44px rgba(45,42,82,.22); animation:slide .22s ease; }
  @keyframes slide { from { transform:translateX(28px); opacity:0; } }
  @media (prefers-reduced-motion: reduce) { #drawer { animation:none; } }
  header { display:flex; align-items:center; gap:6px; padding:22px 14px 6px 24px; }
  h2 { font-family:var(--serif); font-weight:600; font-size:27px; margin:0; flex:1; letter-spacing:-.01em; }
  .icon-btn { width:38px; height:38px; border:none; background:none; border-radius:999px; color:var(--muted); cursor:pointer;
              display:flex; align-items:center; justify-content:center; padding:0; flex:none; }
  .icon-btn:hover { color:var(--purple); background:rgba(45,42,82,.07); } .icon-btn svg { width:19px; height:19px; }
  .icon-btn[disabled] { opacity:.35; cursor:not-allowed; } .icon-btn[disabled]:hover { background:none; color:var(--muted); }
  .view { overflow:auto; flex:1; padding:6px 24px 28px; }
  .view[hidden] { display:none; }
  .hint { color:var(--muted); font-size:13px; line-height:1.5; margin:0 0 14px; }
  .row-actions { display:flex; gap:8px; align-items:center; margin:0 0 12px; flex-wrap:wrap; }
  .btn { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:999px; font:inherit; font-size:13.5px; font-weight:600;
         padding:8px 16px; cursor:pointer; display:inline-flex; align-items:center; gap:7px; }
  .btn:hover { border-color:var(--lavender); }
  .btn.primary { background:var(--purple); color:#fff; border-color:var(--purple); } .btn.primary:hover { background:#433e78; }
  .btn.danger { color:#9b2f2f; } .btn.danger.primary { background:#9b2f2f; border-color:#9b2f2f; color:#fff; }
  .btn[disabled] { opacity:.45; cursor:not-allowed; }
  .btn svg { width:14px; height:14px; }
  .confirm { background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin:0 0 12px; font-size:14px; }
  .confirm[hidden] { display:none; }
  .confirm p { margin:0 0 10px; }
  .song { display:grid; grid-template-columns:44px minmax(0,1fr) auto auto; align-items:center; gap:10px; padding:7px 6px; border-radius:12px; }
  .song:hover { background:rgba(255,255,255,.7); }
  .art { width:44px; height:44px; border-radius:9px; background:#ebe6dc; object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:18px; }
  .meta { min-width:0; } .title { font-size:14.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:12.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  .lang { font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:1px 7px; margin-left:6px; background:#fff; }
  .empty { color:var(--muted); font-size:14px; line-height:1.55; background:#fff; border:1px dashed var(--line); border-radius:14px; padding:18px; text-align:center; }
  .note { font-size:13px; color:var(--ink); background:var(--lav-soft); border-radius:12px; padding:10px 14px; margin:0 0 12px; }
  .note[hidden], .empty[hidden], .row-actions[hidden] { display:none; }
  #opensettings { width:40px; padding:0; justify-content:center; }
  #back { margin-left:-10px; } #back[hidden] { display:none; }
  /* settings */
  .setting { display:flex; align-items:center; justify-content:space-between; gap:16px; background:#fff; border:1px solid var(--line);
             border-radius:14px; padding:14px 16px; margin:0 0 10px; cursor:pointer; }
  .setting b { display:block; font-size:14.5px; font-weight:600; } .setting small { display:block; color:var(--muted); font-size:12.5px; line-height:1.45; margin-top:3px; }
  input[type="checkbox"][role="switch"] { appearance:none; flex:none; width:44px; height:26px; border-radius:999px; background:#d9d4e6; position:relative;
         cursor:pointer; margin:0; transition:background .15s; }
  input[type="checkbox"][role="switch"]::after { content:""; position:absolute; top:3px; left:3px; width:20px; height:20px; border-radius:50%; background:#fff;
         box-shadow:0 1px 3px rgba(0,0,0,.2); transition:left .15s; }
  input[type="checkbox"][role="switch"]:checked { background:var(--purple); }
  input[type="checkbox"][role="switch"]:checked::after { left:21px; }
  fieldset.setting { display:block; cursor:default; } fieldset.setting legend { float:left; width:100%; padding:0; margin:0 0 10px; }
  .seg { display:flex; background:var(--lav-soft); border-radius:999px; padding:3px; clear:both; }
  .seg label { flex:1; text-align:center; font-size:13.5px; padding:7px 0; border-radius:999px; cursor:pointer; color:var(--muted); font-weight:600; }
  .seg input { position:absolute; opacity:0; pointer-events:none; }
  .seg label:has(input:checked) { background:#fff; color:var(--purple); box-shadow:0 1px 3px rgba(45,42,82,.15); }
  .seg label:has(input:focus-visible) { outline:3px solid var(--lavender); }
  .group { font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:18px 2px 8px; }
  .link-row { display:flex; width:100%; align-items:center; justify-content:space-between; background:#fff; border:1px solid var(--line); border-radius:14px;
              padding:14px 16px; font:inherit; font-size:14.5px; font-weight:600; color:var(--ink); cursor:pointer; text-align:left; margin:0 0 10px; }
  .link-row:hover { border-color:var(--lavender); }
  .link-row[disabled] { cursor:default; color:var(--muted); } .link-row[disabled]:hover { border-color:var(--line); }
  /* about */
  #v-about h3 { font-family:var(--serif); font-size:18px; font-weight:600; margin:20px 0 6px; }
  #v-about p, #v-about li { font-size:14px; line-height:1.6; margin:0 0 8px; }
  #v-about ul { padding-left:20px; margin:0 0 8px; }
  #v-about a { color:var(--purple); }
</style></head>
<body>
<div id="buttons">
  <button class="top" id="openlist" aria-haspopup="dialog" title="My playlist"><span id="listicon"></span><span class="label">My playlist</span><span class="badge" id="badge" hidden>0</span></button>
  <button class="top" id="opensettings" aria-haspopup="dialog" title="Settings" aria-label="Settings"></button>
</div>
<div id="overlay" hidden>
  <div id="backdrop"></div>
  <section id="drawer" role="dialog" aria-modal="true" aria-labelledby="dtitle">
    <header><button class="icon-btn" id="back" title="Back to Settings" aria-label="Back to Settings" hidden></button>
      <h2 id="dtitle">My playlist</h2><button class="icon-btn" id="dclose" title="Close" aria-label="Close"></button></header>
    <div class="view" id="v-list" data-title="My playlist">
      <p class="hint">Songs you save with ♡ stay in this browser, on this computer only. They are never sent anywhere.</p>
      <p class="note" id="noplayer" hidden>Songs can play once your mood has been read and the player is on the page.</p>
      <div class="row-actions" id="listactions"><button class="btn primary" id="playall"></button><button class="btn danger" id="clear">Clear</button></div>
      <div class="confirm" id="confirmclear" hidden><p id="confirmtext"></p>
        <button class="btn danger primary" id="clearyes">Clear all</button> <button class="btn" id="clearno">Cancel</button></div>
      <div id="songs"></div>
      <p class="empty" id="empty" hidden>No saved songs yet. Tap ♡ next to a song, or press L while it plays, to keep it here.</p>
    </div>
    <div class="view" id="v-settings" data-title="Settings" hidden>
      <p class="hint">Changes apply straight away, without stopping the music. They are kept in this browser only.</p>
      <div class="group">Playback</div>
      <label class="setting"><span><b>Autoplay next song</b><small>When a song ends, the next one starts.</small></span>
        <input type="checkbox" role="switch" id="s-autoplay"></label>
      <fieldset class="setting"><legend><b>Repeat</b><small>Where the player starts. The repeat button in the player changes it for now.</small></legend>
        <div class="seg" role="radiogroup" aria-label="Default repeat">
          <label><input type="radio" name="repeat" value="off">Off</label><label><input type="radio" name="repeat" value="all">All</label><label><input type="radio" name="repeat" value="one">This song</label></div></fieldset>
      <div class="group">Display</div>
      <label class="setting"><span><b>Show all 7 emotions</b><small>Open "See all 7 emotions" on the mood card by default.</small></span>
        <input type="checkbox" role="switch" id="s-emotions"></label>
      <label class="setting"><span><b>Larger text</b><small>Makes the page's text and cards bigger.</small></span>
        <input type="checkbox" role="switch" id="s-large"></label>
      <div class="group">Your data</div>
      <button class="link-row" id="s-clear"><span>Clear my saved songs <small id="s-count" style="display:block;font-weight:400;color:var(--muted);font-size:12.5px;margin-top:3px"></small></span><span aria-hidden="true">›</span></button>
      <div class="confirm" id="confirmclear2" hidden><p id="confirmtext2"></p>
        <button class="btn danger primary" id="clearyes2">Clear all</button> <button class="btn" id="clearno2">Cancel</button></div>
      <button class="link-row" id="openabout"><span>About MOOD TUNE</span><span aria-hidden="true">›</span></button>
    </div>
    <div class="view" id="v-about" data-title="About MOOD TUNE" hidden>
      <p>MOOD TUNE reads your expression from the webcam (or a photo) and, if you like, a sentence about your day, and combines the two into a mood reading.
        Then it suggests music: songs that match how you feel ("Meet me here"), or a five-song path from a low mood towards a lighter one ("Lift my mood").</p>
      <h3>Where the songs come from</h3>
      <ul>
        <li>English songs: the Muse dataset (<code>muse_v3.csv</code>), with listeners' mood tags. Links go to Last.fm.</li>
        <li>Hindi, Tamil, Telugu, Malayalam and Korean songs: "Spotify Tracks Dataset (Updated Every Week)" by gauthamvijayaraj on Kaggle, Apache 2.0 licence. Links go to Spotify.</li>
        <li>Previews and artwork: 30-second previews from the iTunes Search API, played from Apple's servers.</li>
        <li>Fonts: Fraunces and Inter, SIL Open Font License.</li>
      </ul>
      <h3>Limits</h3>
      <ul>
        <li>The face model reads 63.2% of faces correctly on the FER2013 test set (3,589 faces). Fear and sad are the emotions it gets wrong most often.</li>
        <li>Only 30-second previews play, not whole songs.</li>
        <li>Each song's language is as labelled in its dataset; MOOD TUNE does not check it.</li>
      </ul>
      <h3>Privacy</h3>
      <ul>
        <li>Your webcam picture (or photo) is analysed in memory and never saved.</li>
        <li>Your words are kept only for this visit.</li>
        <li>Saved songs and these settings stay in this browser, on this computer.</li>
      </ul>
    </div>
  </section>
</div>
<script>
__BASE__
let opener = null;
$("listicon").outerHTML = icon("heart"); $("dclose").innerHTML = icon("close");
$("opensettings").innerHTML = icon("gear"); $("back").innerHTML = icon("back");
$("playall").innerHTML = icon("play") + "Play all";
const playerOnPage = () => { try { return !!window.parent.document.querySelector(".st-key-playerbar iframe"); } catch (e) { return false; } };

// Opening: the iframe is pinned over the whole window, and the buttons are
// drawn where they were, under the dimmed backdrop.
function openDrawer(view, button) {
  const f = window.frameElement;
  if (f) {
    const r = f.getBoundingClientRect();
    $("buttons").style.cssText = `position:fixed; left:${r.left}px; top:${r.top}px; width:${r.width}px;`;
    Object.assign(f.style, { position: "fixed", left: "0", top: "0", width: "100vw", height: "100vh", zIndex: "1002" });
  }
  $("overlay").hidden = false; opener = button;
  showView(view);
}
function showView(view) {
  document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== view; });
  $("dtitle").textContent = $(view).dataset.title;
  $("back").hidden = view !== "v-about";
  $("confirmclear").hidden = $("confirmclear2").hidden = true;
  $(view).scrollTop = 0;
  render(); renderSettings();
  (view === "v-about" ? $("back") : $("dclose")).focus();
}
function closeDrawer() {
  if ($("overlay").hidden) return;
  $("overlay").hidden = true; $("confirmclear").hidden = true;
  const f = window.frameElement;
  if (f) Object.assign(f.style, { position: "", left: "", top: "", width: "", height: "", zIndex: "" });
  $("buttons").style.cssText = "";
  if (opener) opener.focus();
}

function songRow(s, k) {
  const r = el("div", "song");
  r.appendChild(artNode(s, "art"));
  const meta = el("div", "meta"), title = el("div", "title", s.title);
  if (s.language) title.appendChild(el("span", "lang", s.language));
  meta.appendChild(title); meta.appendChild(el("div", "artist", s.artist)); r.appendChild(meta);
  const play = el("button", "icon-btn play"); play.innerHTML = icon("play"); play.title = "Play"; play.ariaLabel = "Play " + s.title;
  play.disabled = !playerOnPage();
  play.addEventListener("click", () => { hub.postMessage({ type: "playSaved", songs: [s] }); closeDrawer(); });
  const remove = el("button", "icon-btn remove"); remove.innerHTML = icon("close"); remove.title = "Remove"; remove.ariaLabel = "Remove " + s.title + " from My playlist";
  remove.addEventListener("click", () => { unsaveSong(s.link); toast("Removed “" + s.title + "” from My playlist"); });
  r.appendChild(play); r.appendChild(remove);
  return r;
}
function render() {
  const songs = savedSongs(), n = songs.length;
  $("badge").textContent = n; $("badge").hidden = !n;
  $("openlist").ariaLabel = "My playlist, " + n + (n === 1 ? " song" : " songs");
  if ($("overlay").hidden) return;
  const list = $("songs"); list.replaceChildren(); songs.forEach((s, k) => list.appendChild(songRow(s, k)));
  $("empty").hidden = n > 0; $("listactions").hidden = !n;
  $("playall").disabled = !playerOnPage(); $("noplayer").hidden = !n || playerOnPage();
  if (!n) $("confirmclear").hidden = true;
}
onSavedChange = () => { render(); renderSettings(); };

// Settings. "Larger text" and "Show all 7 emotions" act on the page itself
// (a class on the page, and opening the mood card's expander), so they never
// rerun the app; autoplay and repeat are read by the player bar.
function renderSettings() {
  const s = settings(), n = savedSongs().length;
  $("s-autoplay").checked = s.autoplay; $("s-emotions").checked = s.emotions; $("s-large").checked = s.large;
  document.querySelectorAll('input[name="repeat"]').forEach(r => { r.checked = r.value === s.repeat; });
  $("s-count").textContent = n ? n + (n === 1 ? " song" : " songs") + " saved in this browser" : "Nothing saved";
  $("s-clear").disabled = !n;
  if (!n) $("confirmclear2").hidden = true;
}
function applyLarge() { try { window.parent.document.documentElement.classList.toggle("moodtune-large", settings().large); } catch (e) {} }
function applyEmotions(now) {
  if (!settings().emotions) return;
  try {
    // each mood card's expander is opened once, the first time it appears (or now, when the setting is turned on)
    window.parent.document.querySelectorAll(".st-key-moodcard details").forEach(d => {
      if (d.dataset.moodtuneSeen && !now) return;
      d.dataset.moodtuneSeen = "1";
      if (!d.open) { const s = d.querySelector("summary"); if (s) s.click(); }
    });
  } catch (e) {}
}
try { new MutationObserver(() => applyEmotions(false)).observe(window.parent.document.body, { childList: true, subtree: true }); } catch (e) {}
onSettingsChange = () => { renderSettings(); applyLarge(); };
$("s-autoplay").addEventListener("change", e => setSetting("autoplay", e.target.checked));
$("s-emotions").addEventListener("change", e => { setSetting("emotions", e.target.checked); applyEmotions(true); });
$("s-large").addEventListener("change", e => setSetting("large", e.target.checked));
document.querySelectorAll('input[name="repeat"]').forEach(r => r.addEventListener("change", () => setSetting("repeat", r.value)));
$("s-clear").addEventListener("click", () => {
  const n = savedSongs().length; if (!n) return;
  $("confirmtext2").textContent = "Remove all " + n + (n === 1 ? " saved song" : " saved songs") + " from this browser?";
  $("confirmclear2").hidden = false; $("clearno2").focus();
});
$("clearno2").addEventListener("click", () => { $("confirmclear2").hidden = true; $("s-clear").focus(); });
$("clearyes2").addEventListener("click", () => { setSaved([]); $("confirmclear2").hidden = true; toast("Your saved songs are cleared"); $("dclose").focus(); });
$("openabout").addEventListener("click", () => showView("v-about"));
$("back").addEventListener("click", () => showView("v-settings"));
$("opensettings").addEventListener("click", () => openDrawer("v-settings", $("opensettings")));

$("openlist").addEventListener("click", () => openDrawer("v-list", $("openlist")));
$("dclose").addEventListener("click", closeDrawer);
$("backdrop").addEventListener("click", closeDrawer);
$("playall").addEventListener("click", () => { const songs = savedSongs(); if (songs.length) { hub.postMessage({ type: "playSaved", songs }); closeDrawer(); } });
$("clear").addEventListener("click", () => {
  const n = savedSongs().length;
  $("confirmtext").textContent = "Remove all " + n + (n === 1 ? " saved song" : " saved songs") + " from this browser?";
  $("confirmclear").hidden = false; $("clearno").focus();
});
$("clearno").addEventListener("click", () => { $("confirmclear").hidden = true; $("clear").focus(); });
$("clearyes").addEventListener("click", () => { setSaved([]); $("confirmclear").hidden = true; toast("My playlist is empty"); $("dclose").focus(); });

// Escape closes the drawer; Tab stays inside it while it is open; shortcut
// keys pressed here go to the player bar.
document.addEventListener("keydown", e => {
  if (!$("overlay").hidden) {
    if (e.key === "Escape") { e.preventDefault(); closeDrawer(); return; }
    if (e.key === "Tab") {
      const items = [...$("drawer").querySelectorAll("button, a, input, select")].filter(x => !x.disabled && x.offsetParent);
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      return;
    }
  }
  const key = shortcut(e);
  if (!key) return;
  e.preventDefault();
  if (key !== "held") hub.postMessage({ type: "key", key });
});
render(); renderSettings(); applyLarge(); applyEmotions(false);
window.actions = { open: openDrawer, close: closeDrawer, saved: savedSongs, settings };
</script>
</body></html>
""".replace("__BASE__", BASE_JS)
