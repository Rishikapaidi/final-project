"""Build the play queue and render the page as one HTML/JS component.

The queue is a list of tracks with a playable 30-second preview (see
previews.py). Everything below the header (the scan picture, the emotion bars,
the mood card, the track list and the player bar) is one self-contained
HTML/JS page shown with st.components.v1.html: Streamlit reruns the script on
every interaction, which would stop the music, so play, pause, next/previous
and "play the next song when this one ends" all happen in the browser's
<audio> element.
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


def render(tracks, subtitle, journey=None, loop_from=0, view=None):
    """The whole page below the header as one HTML string.

    view holds what the panels show: {"scan": {"image", "note"}, "bars":
    {"emotions", "groups", "caption"}, "card": mood.card(...), "note"}.
    All text is inserted as text, never as HTML.
    """
    data = {"tracks": tracks, "subtitle": subtitle, "journey": journey, "loopFrom": loop_from, "view": view or {}}
    payload = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return PLAYER_HTML.replace("__DATA__", payload)


PLAYER_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#121212; --panel:#181818; --row:#2a2a2a; --line:#282828; --text:#ffffff; --muted:#b3b3b3; --green:#1db954; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  #main { position:absolute; top:0; left:0; right:0; bottom:88px; display:grid;
          grid-template-columns:minmax(300px, 360px) minmax(0, 1fr); gap:18px; padding:4px 4px 0; }
  #left, #picks { overflow-y:auto; padding-bottom:18px; }
  #left::-webkit-scrollbar, #picks::-webkit-scrollbar { width:10px; }
  #left::-webkit-scrollbar-thumb, #picks::-webkit-scrollbar-thumb { background:#3a3a3a; border-radius:5px; }
  .card { background:var(--panel); border-radius:12px; padding:16px; margin-bottom:16px; }
  .card h3 { margin:0 0 12px; font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  #scanimg { width:100%; border-radius:8px; display:block; background:#000; }
  .scan-empty { height:150px; border-radius:8px; background:#101010; display:flex; align-items:center; justify-content:center; color:#666; font-size:13px; }
  .small { color:var(--muted); font-size:12.5px; line-height:1.45; margin:10px 0 0; }
  .legend { display:flex; gap:14px; font-size:12px; color:var(--muted); margin-bottom:12px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  .erow { display:grid; grid-template-columns:70px 1fr; gap:10px; align-items:center; margin-bottom:11px; }
  .ename { font-size:13.5px; text-transform:capitalize; }
  .ebars { display:flex; flex-direction:column; gap:3px; }
  .bar { display:grid; grid-template-columns:1fr 38px; gap:8px; align-items:center; }
  .bar .rail { background:#262626; border-radius:3px; height:6px; overflow:hidden; }
  .bar .fill { height:100%; border-radius:3px; }
  .bar .pct { font-size:11px; color:#8c8c8c; text-align:right; font-variant-numeric:tabular-nums; }
  .bar.face .fill { background:#7d8590; } .bar.text .fill { background:#5b8def; }
  .bar.combined .rail { height:9px; } .bar.combined .fill { background:var(--green); }
  .bar.combined .pct { color:#fff; font-weight:600; font-size:12px; }
  #mood { display:grid; grid-template-columns:auto 1fr; gap:18px; align-items:center; border-left:5px solid var(--mood, #1db954); }
  #mood .emoji { font-size:58px; line-height:1; }
  #mood h2 { margin:0; font-size:26px; text-transform:none; } #mood h2 b { color:var(--mood, #1db954); font-weight:700; }
  #mood .mode { display:inline-block; margin-top:8px; font-size:12.5px; color:#e6e6e6; background:#242424; border:1px solid #333; border-radius:14px; padding:3px 11px; }
  #mood .msg { margin:10px 0 0; color:#d0d0d0; font-size:14px; line-height:1.5; max-width:70ch; }
  #mood .note { margin:8px 0 0; color:#9a9a9a; font-size:12.5px; }
  .picks-head { display:flex; align-items:center; gap:14px; margin:2px 0 10px; }
  .picks-head h2 { margin:0; font-size:20px; } .picks-head p { margin:2px 0 0; color:var(--muted); font-size:13px; }
  #bigplay { width:46px; height:46px; border-radius:50%; border:none; background:var(--green); cursor:pointer;
             display:flex; align-items:center; justify-content:center; flex:none; transition:transform .1s; }
  #bigplay:hover { transform:scale(1.06); } #bigplay svg { width:20px; height:20px; fill:#000; }
  details.journey { background:#101010; border-radius:8px; margin:0 0 10px; padding:0 14px; }
  details.journey summary { cursor:pointer; padding:10px 0; font-size:13.5px; font-weight:600; color:var(--muted); }
  details.journey[open] summary { color:var(--text); }
  details.journey .cap { color:var(--muted); font-size:12px; margin:0 0 12px; }
  .section { color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; margin:14px 4px 6px; }
  .row { display:grid; grid-template-columns:30px 46px minmax(0,1fr) auto auto auto; align-items:center; gap:13px;
         padding:7px 12px 7px 4px; border-radius:6px; cursor:pointer; }
  .row:hover { background:#222; } .row.playing { background:var(--row); }
  .num { text-align:center; color:var(--muted); font-size:14.5px; font-variant-numeric:tabular-nums; }
  .row.playing .num, .row.playing .title { color:var(--green); }
  .art { width:46px; height:46px; border-radius:4px; background:#333; object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#777; font-size:20px; }
  .meta { min-width:0; } .title { font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:13px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  .tag { font-size:11px; color:var(--muted); border:1px solid #3a3a3a; border-radius:10px; padding:2px 8px; white-space:nowrap; }
  .pill { border:none; background:var(--green); color:#000; font-weight:700; font-size:12px; border-radius:14px;
          padding:5px 13px; cursor:pointer; white-space:nowrap; min-width:68px; }
  .pill:hover { filter:brightness(1.1); }
  .lastfm { font-size:12px; color:var(--muted); text-decoration:none; white-space:nowrap; }
  .lastfm:hover { color:var(--text); text-decoration:underline; }
  .eq { display:inline-flex; gap:2px; align-items:flex-end; height:14px; }
  .eq i { width:3px; background:var(--green); animation:eq 0.9s ease-in-out infinite; }
  .eq i:nth-child(2) { animation-delay:.25s; } .eq i:nth-child(3) { animation-delay:.5s; }
  .paused .eq i { animation-play-state:paused; }
  @keyframes eq { 0%,100% { height:4px; } 50% { height:14px; } }
  #bar { position:absolute; left:0; right:0; bottom:0; height:88px; background:var(--panel); border-top:1px solid var(--line);
         display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.5fr) minmax(0,1fr); align-items:center; gap:16px; padding:0 20px; }
  #now { display:flex; align-items:center; gap:12px; min-width:0; }
  #now .art { width:54px; height:54px; flex:none; }
  #center { display:flex; flex-direction:column; align-items:center; gap:7px; }
  .controls { display:flex; align-items:center; gap:20px; }
  .controls button { background:none; border:none; cursor:pointer; padding:0; display:flex; }
  .controls svg { width:20px; height:20px; fill:var(--muted); } .controls button:hover svg { fill:var(--text); }
  #toggle { width:36px; height:36px; border-radius:50%; background:#fff !important; align-items:center; justify-content:center; }
  #toggle svg { fill:#000 !important; width:16px; height:16px; }
  .progress { display:flex; align-items:center; gap:9px; width:100%; font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
  #track { flex:1; height:4px; background:#4d4d4d; border-radius:2px; cursor:pointer; position:relative; }
  #fill { height:100%; width:0; background:#fff; border-radius:2px; } #track:hover #fill { background:var(--green); }
  #position { text-align:right; color:var(--muted); font-size:12px; }
  .empty { color:var(--muted); padding:30px 8px; }
  @media (max-width: 860px) { #main { grid-template-columns:1fr; overflow-y:auto; } #left, #picks { overflow:visible; } }
</style></head>
<body>
<div id="main">
  <div id="left">
    <div class="card" id="scan"><h3>Your scan</h3><div id="scanbody"></div></div>
    <div class="card" id="analysis"><h3>Emotion analysis</h3><div id="legend" class="legend"></div><div id="bars"></div><p class="small" id="barcap"></p></div>
  </div>
  <div id="picks">
    <div class="card" id="mood"><div class="emoji" id="moodemoji"></div>
      <div><h2 id="moodhead"></h2><span class="mode" id="moodmode"></span><p class="msg" id="moodmsg"></p><p class="note" id="moodnote"></p></div></div>
    <div class="card" id="list">
      <div class="picks-head">
        <button id="bigplay" title="Play" aria-label="Play"></button>
        <div><h2>Top Picks For You</h2><p id="qsub"></p></div>
      </div>
      <div id="journey"></div>
      <div id="rows"></div>
    </div>
  </div>
</div>
<div id="bar">
  <div id="now"><div class="art noart" id="nowart">&#9835;</div>
    <div class="meta"><div class="title" id="nowtitle">Nothing playing</div><div class="artist" id="nowartist">Press play or pick a song</div></div></div>
  <div id="center">
    <div class="controls">
      <button id="prev" title="Previous" aria-label="Previous"><svg viewBox="0 0 16 16"><path d="M3 2h2v12H3zM14 2v12L6 8z"/></svg></button>
      <button id="toggle" title="Play / pause" aria-label="Play / pause"></button>
      <button id="next" title="Next" aria-label="Next"><svg viewBox="0 0 16 16"><path d="M11 2h2v12h-2zM2 2l8 6-8 6z"/></svg></button>
    </div>
    <div class="progress"><span id="tcur">0:00</span><div id="track"><div id="fill"></div></div><span id="tdur">0:30</span></div>
  </div>
  <div id="position"></div>
</div>
<script>
const DATA = __DATA__;
const tracks = DATA.tracks, VIEW = DATA.view || {};
const PLAY = '<svg viewBox="0 0 16 16"><path d="M3 1.5v13l11-6.5z"/></svg>';
const PAUSE = '<svg viewBox="0 0 16 16"><path d="M3 1.5h3.5v13H3zM9.5 1.5H13v13H9.5z"/></svg>';
const SECTIONS = { journey: "Your journey, in play order", more: "Keep listening", mood: "" };
const audio = new Audio();
audio.preload = "auto";
let current = -1, failures = 0;
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
const pct = p => Math.round(p * 100) + "%";

function artNode(track, cls) {
  if (track.art) { const img = el("img", cls); img.src = track.art; img.alt = ""; img.loading = "lazy"; return img; }
  return el("div", cls + " noart", "♫");
}

function buildScan() {
  const scan = VIEW.scan || {}, body = $("scanbody");
  if (scan.image) { const img = el("img"); img.id = "scanimg"; img.src = scan.image; img.alt = "The frame used for your scan"; body.appendChild(img); }
  else body.appendChild(el("div", "scan-empty", "No picture for this scan"));
  if (scan.note) body.appendChild(el("p", "small", scan.note));
}

function buildBars() {
  const bars = VIEW.bars; if (!bars) return;
  const names = { face: "Face", text: "Text", combined: "Combined" }, colours = { face: "#7d8590", text: "#5b8def", combined: "#1db954" };
  bars.groups.forEach(g => { const item = el("span", null, names[g.key]); const sw = el("i"); sw.style.background = colours[g.key]; item.prepend(sw); $("legend").appendChild(item); });
  bars.emotions.forEach((emotion, k) => {
    const row = el("div", "erow"); row.dataset.emotion = emotion; row.appendChild(el("div", "ename", emotion));
    const stack = el("div", "ebars");
    bars.groups.forEach(g => {
      const bar = el("div", "bar " + g.key); bar.dataset.group = g.key; bar.dataset.value = g.values[k];
      const rail = el("div", "rail"), fill = el("div", "fill"); fill.style.width = (g.values[k] * 100).toFixed(1) + "%"; rail.appendChild(fill);
      bar.appendChild(rail); bar.appendChild(el("span", "pct", pct(g.values[k]))); stack.appendChild(bar);
    });
    row.appendChild(stack); $("bars").appendChild(row);
  });
  $("barcap").textContent = bars.caption || "";
}

function buildMood() {
  const c = VIEW.card; if (!c) { $("mood").style.display = "none"; return; }
  $("mood").style.setProperty("--mood", c.colour);
  $("moodemoji").textContent = c.emoji;
  const parts = c.headline.split(/(: | and )/), head = $("moodhead");
  if (c.headline.startsWith("Mostly ")) { head.appendChild(document.createTextNode("Mostly ")); head.appendChild(el("b", null, c.headline.slice(7))); }
  else parts.forEach((p, i) => {
    if (i !== 2 && i !== 4) return head.appendChild(document.createTextNode(p));
    const word = el("b", null, p); if (i === 4) word.style.color = c.second_colour; head.appendChild(word);
  });
  $("moodmode").textContent = c.mode_line; $("moodmsg").textContent = c.message; $("moodnote").textContent = VIEW.note || "";
}

function buildRows() {
  $("qsub").textContent = DATA.subtitle;
  const rows = $("rows");
  if (!tracks.length) { rows.appendChild(el("div", "empty", "No playable songs were found for this mood.")); return; }
  let section = null;
  tracks.forEach((t, i) => {
    if (t.section !== section) {
      section = t.section;
      let label = SECTIONS[section];
      if (section === "more") label += " · " + t.bucket;
      if (label) rows.appendChild(el("div", "section", label));
    }
    const row = el("div", "row"); row.dataset.index = i;
    row.appendChild(el("div", "num", String(t.number)));
    row.appendChild(artNode(t, "art"));
    const meta = el("div", "meta"); meta.appendChild(el("div", "title", t.title)); meta.appendChild(el("div", "artist", t.artist)); row.appendChild(meta);
    row.appendChild(el("span", "tag", t.bucket));
    const pill = el("button", "pill", "▶ Play"); pill.title = "Play this song";
    pill.addEventListener("click", e => { e.stopPropagation(); if (i === current) toggle(); else play(i); }); row.appendChild(pill);
    const link = el("a", "lastfm", t.link_label || "Last.fm"); link.href = t.lastfm; link.target = "_blank"; link.rel = "noopener noreferrer";
    link.addEventListener("click", e => e.stopPropagation()); row.appendChild(link);
    row.addEventListener("click", () => play(i));
    rows.appendChild(row);
  });
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
  const add = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.appendChild(n); return n; };
  add("line", { x1: P.l, y1: H - P.b, x2: W - P.r, y2: H - P.b, stroke: "#444" }); add("line", { x1: P.l, y1: P.t, x2: P.l, y2: H - P.b, stroke: "#444" });
  add("text", { x: (P.l + W - P.r) / 2, y: H - 8, fill: "#b3b3b3", "font-size": 12, "text-anchor": "middle" }, "valence (negative → positive)");
  add("text", { x: 14, y: (P.t + H - P.b) / 2, fill: "#b3b3b3", "font-size": 12, "text-anchor": "middle", transform: `rotate(-90 14 ${(P.t + H - P.b) / 2})` }, "arousal (calm → energetic)");
  [x0, x1].forEach((v, k) => add("text", { x: k ? W - P.r : P.l, y: H - P.b + 15, fill: "#777", "font-size": 10, "text-anchor": k ? "end" : "start" }, v.toFixed(1)));
  [y0, y1].forEach((v, k) => add("text", { x: P.l - 6, y: k ? P.t + 9 : H - P.b, fill: "#777", "font-size": 10, "text-anchor": "end" }, v.toFixed(1)));
  add("polyline", { points: j.waypoints.map(p => X(p[0]) + "," + Y(p[1])).join(" "), fill: "none", stroke: "#777", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
  add("polyline", { points: songs.map(s => X(s.valence) + "," + Y(s.arousal)).join(" "), fill: "none", stroke: "#1db954", "stroke-width": 2 });
  songs.forEach(s => { add("circle", { cx: X(s.valence), cy: Y(s.arousal), r: 11, fill: "#1db954" }); add("text", { x: X(s.valence), y: Y(s.arousal) + 4, fill: "#000", "font-size": 12, "font-weight": 700, "text-anchor": "middle" }, String(s.number)); });
  box.appendChild(svg);
  box.appendChild(el("p", "cap", "Dashed line: the planned path. Green: the songs picked, in play order. Values come from listeners' mood tags."));
  $("journey").appendChild(box);
}

function refresh() {
  const playing = current >= 0 && !audio.paused;
  document.querySelectorAll(".row").forEach(r => {
    const i = +r.dataset.index, on = i === current;
    r.classList.toggle("playing", on); r.classList.toggle("paused", on && audio.paused);
    const num = r.querySelector(".num");
    if (on) num.innerHTML = '<span class="eq"><i></i><i></i><i></i></span>'; else num.textContent = String(tracks[i].number);
    r.querySelector(".pill").textContent = on && playing ? "❚❚ Pause" : "▶ Play";
  });
  $("toggle").innerHTML = playing ? PAUSE : PLAY; $("bigplay").innerHTML = playing ? PAUSE : PLAY;
  if (current >= 0) {
    const t = tracks[current];
    $("nowtitle").textContent = t.title; $("nowartist").textContent = t.artist;
    const old = $("nowart"), art = artNode(t, "art"); art.id = "nowart"; old.replaceWith(art);
    $("position").textContent = (current + 1) + " of " + tracks.length;
  }
}

function play(i) {
  if (!tracks.length) return;
  current = (i + tracks.length) % tracks.length;
  audio.src = tracks[current].preview;
  audio.play().catch(() => {});
  const row = document.querySelector('.row[data-index="' + current + '"]');
  if (row) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
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
  const d = audio.duration || 30;
  $("fill").style.width = Math.min(100, audio.currentTime / d * 100) + "%";
  $("tcur").textContent = fmt(audio.currentTime); $("tdur").textContent = fmt(d);
});
$("track").addEventListener("click", e => { if (current < 0 || !audio.duration) return; const r = $("track").getBoundingClientRect(); audio.currentTime = (e.clientX - r.left) / r.width * audio.duration; });
$("toggle").addEventListener("click", toggle); $("bigplay").addEventListener("click", toggle);
$("next").addEventListener("click", next); $("prev").addEventListener("click", prev);

buildScan(); buildBars(); buildMood(); buildRows(); buildJourney(); refresh();
window.player = { audio, state: () => ({ current, paused: audio.paused, time: audio.currentTime, src: audio.src }) };
</script>
</body></html>
"""
