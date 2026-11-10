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
             "link_label": getattr(song, "link_source", "") or "Last.fm",
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
    nothing itself: rows and Play pills send commands to the player bar
    (render_bar) over a BroadcastChannel named after the queue id.
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

# Shared by the scripts of both pages.
COMMON_JS = r"""
const DATA = __DATA__;
const tracks = DATA.tracks;
const channel = new BroadcastChannel("moodtune:" + DATA.id);
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
};
const icon = name => '<svg class="s" viewBox="0 0 24 24" aria-hidden="true">' + ICONS[name] + '</svg>';
function artNode(track, cls) {
  if (track.art) { const img = el("img", cls); img.src = track.art; img.alt = ""; img.loading = "lazy"; return img; }
  const box = el("div", cls + " noart", "♫"); box.setAttribute("aria-hidden", "true"); return box;
}
// Keyboard shortcuts: Space = play/pause, Right/Left = next/previous, M = mute.
// Nothing happens while typing in a text box, or with a modifier key held;
// Space on a button, link or row is left to that control.
function shortcut(e) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return null;
  const t = e.target || {}, tag = (t.tagName || "").toLowerCase(), role = t.getAttribute ? t.getAttribute("role") || "" : "";
  if (t.isContentEditable || tag === "input" || tag === "textarea" || tag === "select" || role === "combobox" || role === "textbox") return null;
  const key = { " ": "space", ArrowRight: "next", ArrowLeft: "prev", m: "mute", M: "mute" }[e.key] || null;
  if (key === "space" && (tag === "button" || tag === "a" || tag === "summary" || role === "button" || role === "menuitem")) return null;
  if ((key === "next" || key === "prev") && role === "slider") return null;
  if (key && key !== "next" && key !== "prev" && e.repeat) return "held";
  return key;
}
"""

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
  .row { display:grid; grid-template-columns:30px 48px minmax(0,1fr) auto auto auto; align-items:center; gap:14px;
         padding:8px 12px 8px 6px; border-radius:12px; cursor:pointer; }
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
  .lastfm { font-size:12.5px; color:var(--muted); text-decoration:none; white-space:nowrap; }
  .lastfm:hover { color:var(--purple); text-decoration:underline; }
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
    .row { grid-template-columns:24px 44px minmax(0,1fr) auto; gap:10px; padding:8px 4px; }
    .row .tag, .row .lastfm { display:none; } .art { width:44px; height:44px; }
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
    <div id="journeyrows"></div>
    <div id="journey"></div>
  </div>
  <div id="rows"></div>
</div>
<script>
__COMMON__
const SECTIONS = { journey: "Your path, in play order", more: "Keep listening", mood: "" };
let current = -1, paused = true;
const send = msg => channel.postMessage(msg);

function row(t, i) {
  const r = el("div", "row"); r.dataset.index = i; r.tabIndex = 0; r.setAttribute("role", "button");
  r.setAttribute("aria-label", "Play " + t.title + " by " + t.artist);
  r.appendChild(el("div", "num", String(t.number)));
  r.appendChild(artNode(t, "art"));
  const meta = el("div", "meta"); meta.appendChild(el("div", "title", t.title)); meta.appendChild(el("div", "artist", t.artist)); r.appendChild(meta);
  r.appendChild(el("span", "tag", t.bucket));
  const pill = el("button", "pill", "▶ Play"); pill.title = "Play this song";
  pill.addEventListener("click", e => { e.stopPropagation(); send(i === current ? { type: "toggle" } : { type: "play", index: i }); }); r.appendChild(pill);
  const link = el("a", "lastfm", t.link_label || "Last.fm"); link.href = t.lastfm; link.target = "_blank"; link.rel = "noopener noreferrer";
  link.title = "Open on " + (t.link_label || "Last.fm"); link.addEventListener("click", e => e.stopPropagation()); r.appendChild(link);
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
  if (!tracks.length) { $("rows").appendChild(el("div", "empty", "No playable songs were found for this mood.")); $("bigplay").hidden = true; return; }
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
      if (label) target.appendChild(el("div", "section", label));
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
build(); refresh(); fit();
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
    <div class="meta"><div class="title" id="nowtitle">Nothing playing</div><div class="artist" id="nowartist">Press play or pick a song</div></div></div>
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
let pos = -1, failures = 0, shuffle = false, repeat = "all", pathDone = pathLen === 0, queueOpen = false;
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

function act(key) {
  if (key === "space") toggle(); else if (key === "next") next(false); else if (key === "prev") prev();
  else if (key === "mute") setMuted(!audio.muted);
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
  next(true);
});
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
$("qclose").addEventListener("click", () => openQueue(false));
$("shade").addEventListener("click", () => openQueue(false));
channel.onmessage = e => {
  const m = e.data;
  if (m.type === "play") playBase(m.index); else if (m.type === "toggle") toggle(); else if (m.type === "hello") announce();
  else if (m.type === "key") act(m.key);
};
refreshVolume(); refresh();
window.player = {
  audio, play: playBase, playAt, toggle, next: () => next(false), prev, setShuffle, cycleRepeat, setMuted, openQueue, insert, removeAt,
  state: () => ({ current: now() ? now().base : -1, pos, paused: audio.paused, time: audio.currentTime, src: audio.src, shuffle, repeat,
                  muted: audio.muted, pathDone, pathLen, queueOpen,
                  order: order.map(e => ({ base: e.base, added: e.added, title: e.t.title, link: e.t.lastfm })) }),
};
</script>
</body></html>
""".replace("__COMMON__", COMMON_JS)
