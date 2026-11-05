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

    app.py pins it to the bottom of the window with CSS. When a song ends the
    next one starts; at the end of the queue it loops from queue["loop_from"]
    (after an Uplift journey, the "keep listening" songs).
    """
    return BAR_HTML.replace("__FONTS__", FONT_CSS).replace("__DATA__", _payload(queue))


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
const PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 1.8v12.4L14 8z"/></svg>';
const PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 1.8h3.6v12.4H3zM9.4 1.8H13v12.4H9.4z"/></svg>';
function artNode(track, cls) {
  if (track.art) { const img = el("img", cls); img.src = track.art; img.alt = ""; img.loading = "lazy"; return img; }
  const box = el("div", cls + " noart", "♫"); box.setAttribute("aria-hidden", "true"); return box;
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
  r.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); send({ type: "play", index: i }); } });
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
  #bar { position:absolute; inset:0; background:rgba(255,253,249,.97); border-top:1px solid var(--line);
         box-shadow:0 -6px 24px rgba(45,42,82,.08); display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.4fr) minmax(0,1fr);
         align-items:center; gap:18px; padding:0 22px; }
  #now { display:flex; align-items:center; gap:12px; min-width:0; }
  .art { width:52px; height:52px; border-radius:10px; background:#ebe6dc; object-fit:cover; flex:none; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:20px; }
  .meta { min-width:0; } .title { font-size:14.5px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:12.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  #center { display:flex; flex-direction:column; align-items:center; gap:7px; min-width:0; }
  .controls { display:flex; align-items:center; gap:22px; }
  button { background:none; border:none; cursor:pointer; padding:4px; display:flex; border-radius:999px; }
  button svg { width:18px; height:18px; fill:var(--muted); } button:hover svg { fill:var(--purple); }
  button:focus-visible, input:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  #toggle { width:42px; height:42px; background:var(--purple); align-items:center; justify-content:center; }
  #toggle svg { fill:#fff !important; width:16px; height:16px; } #toggle:hover { background:#433e78; }
  .progress { display:flex; align-items:center; gap:10px; width:100%; font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
  #track { flex:1; height:5px; background:#e5e0f2; border-radius:3px; cursor:pointer; position:relative; }
  #fill { height:100%; width:0; background:var(--purple); border-radius:3px; }
  #right { display:flex; align-items:center; justify-content:flex-end; gap:10px; color:var(--muted); font-size:12px; }
  #volume { width:110px; accent-color:var(--purple); }
  #right svg { width:18px; height:18px; fill:var(--muted); flex:none; }
  @media (max-width: 760px) {
    #bar { grid-template-columns:minmax(0,1fr) auto; gap:10px; padding:0 12px; }
    #center { flex-direction:row; } .progress, #right { display:none; } .controls { gap:12px; }
    .art { width:44px; height:44px; }
  }
</style></head>
<body>
<div id="bar">
  <div id="now"><div class="art noart" id="nowart" aria-hidden="true">&#9835;</div>
    <div class="meta"><div class="title" id="nowtitle">Nothing playing</div><div class="artist" id="nowartist">Press play or pick a song</div></div></div>
  <div id="center">
    <div class="controls">
      <button id="prev" title="Previous" aria-label="Previous"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2h2v12H3zM14 2v12L6 8z"/></svg></button>
      <button id="toggle" title="Play" aria-label="Play"></button>
      <button id="next" title="Next" aria-label="Next"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M11 2h2v12h-2zM2 2l8 6-8 6z"/></svg></button>
    </div>
    <div class="progress"><span id="tcur">0:00</span><div id="track" role="slider" aria-label="Position in the song" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" tabindex="0"><div id="fill"></div></div><span id="tdur">0:30</span></div>
  </div>
  <div id="right"><span id="position"></span>
    <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6h3l4-3v10l-4-3H2zM11 5.2a4 4 0 0 1 0 5.6l-.8-.8a2.9 2.9 0 0 0 0-4z"/></svg>
    <input id="volume" type="range" min="0" max="1" step="0.05" value="0.8" aria-label="Volume"></div>
</div>
<script>
__COMMON__
const audio = new Audio();
audio.preload = "auto";
let current = -1, failures = 0;
const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };

// The volume is a per-browser convenience; the page works without storage.
try { const v = parseFloat(localStorage.getItem("moodtune.volume")); if (v >= 0 && v <= 1) $("volume").value = v; } catch (e) {}
audio.volume = +$("volume").value;
$("volume").addEventListener("input", () => { audio.volume = +$("volume").value; try { localStorage.setItem("moodtune.volume", $("volume").value); } catch (e) {} });

function announce() { channel.postMessage({ type: "state", current, paused: audio.paused }); }
function refresh() {
  const playing = current >= 0 && !audio.paused;
  $("toggle").innerHTML = playing ? PAUSE : PLAY;
  $("toggle").title = $("toggle").ariaLabel = playing ? "Pause" : "Play";
  if (current >= 0) {
    const t = tracks[current];
    $("nowtitle").textContent = t.title; $("nowartist").textContent = t.artist;
    const old = $("nowart"), art = artNode(t, "art"); art.id = "nowart"; old.replaceWith(art);
    $("position").textContent = (current + 1) + " of " + tracks.length;
  }
  announce();
}
function play(i) {
  if (!tracks.length) return;
  current = (i + tracks.length) % tracks.length;
  audio.src = tracks[current].preview;
  audio.play().catch(() => {});
  refresh();
}
function next() { play(current + 1 >= tracks.length ? DATA.loopFrom : current + 1); }
function prev() { if (audio.currentTime > 3 || current <= 0) { audio.currentTime = 0; audio.play().catch(() => {}); } else play(current - 1); }
function toggle() { if (current < 0) return play(0); if (audio.paused) audio.play().catch(() => {}); else audio.pause(); }

audio.addEventListener("ended", () => { failures = 0; next(); });
audio.addEventListener("error", () => { if (current >= 0 && ++failures < tracks.length) next(); });
audio.addEventListener("playing", () => { failures = 0; refresh(); });
audio.addEventListener("pause", refresh);
audio.addEventListener("timeupdate", () => {
  const d = audio.duration || 30, p = Math.min(100, audio.currentTime / d * 100);
  $("fill").style.width = p + "%"; $("track").setAttribute("aria-valuenow", Math.round(p));
  $("tcur").textContent = fmt(audio.currentTime); $("tdur").textContent = fmt(d);
});
$("track").addEventListener("click", e => { if (current < 0 || !audio.duration) return; const r = $("track").getBoundingClientRect(); audio.currentTime = (e.clientX - r.left) / r.width * audio.duration; });
$("track").addEventListener("keydown", e => { if (current < 0 || !audio.duration) return; if (e.key === "ArrowRight") audio.currentTime += 5; if (e.key === "ArrowLeft") audio.currentTime -= 5; });
$("toggle").addEventListener("click", toggle);
$("next").addEventListener("click", next); $("prev").addEventListener("click", prev);
channel.onmessage = e => {
  const m = e.data;
  if (m.type === "play") play(m.index); else if (m.type === "toggle") toggle(); else if (m.type === "hello") announce();
};
refresh();
window.player = { audio, play, toggle, next, state: () => ({ current, paused: audio.paused, time: audio.currentTime, src: audio.src }) };
</script>
</body></html>
""".replace("__COMMON__", COMMON_JS)
