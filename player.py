"""Build the play queue and render the music-player component.

The queue is a list of tracks with a playable 30-second preview (see
previews.py). The player itself is one self-contained HTML/JS page shown with
st.components.v1.html: Streamlit reruns the script on every interaction, which
would stop the music, so play, pause, next/previous and "play the next song
when this one ends" all happen in the browser's <audio> element.
"""
import json

import numpy as np
import pandas as pd

import previews
import uplift

RESONANCE_TRACKS = 30
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
    track = {"title": song.title, "artist": song.artist, "lastfm": song.lastfm_url, "bucket": song.bucket,
             "valence": round(float(song.valence_tags), 2), "arousal": round(float(song.arousal_tags), 2),
             "preview": preview["preview_url"], "art": preview["artwork_url"]}
    track.update(extra)
    return track


def playable_songs(rows, wanted, rng):
    """Up to `wanted` rows of `rows` that have a preview, in random order.

    Songs already known to have a preview are used first; unchecked songs are
    only looked up if those are not enough.
    """
    rows = rows.iloc[rng.permutation(len(rows))]
    known = [previews.cached(url) for url in rows.lastfm_url]
    ready = [(song, row) for song, row in zip(rows.itertuples(), known) if row and row["preview_url"]]
    unchecked = rows[[row is None for row in known]]
    for round_ in range(LOOKUP_ROUNDS):
        if len(ready) >= wanted or previews.rate_limited():
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


def uplift_queue(songs, fused, emotions, rng=None):
    """(tracks, journey) for an Uplift playlist, or (None, None) if Uplift does not apply."""
    rng = rng or np.random.default_rng()

    def playable(positions):
        found = previews.resolve(_keys(songs.iloc[positions]))
        return np.array([p for p in positions if songs.lastfm_url.iloc[p] in found], dtype=int)

    playlist, journey = uplift.build_playlist(songs, fused, emotions, rng, playable=playable)
    if playlist is None:
        return None, None
    tracks = [_track(song, previews.cached(song.lastfm_url), section="journey", number=song.step,
                     waypoint=[round(float(song.waypoint_valence), 2), round(float(song.waypoint_arousal), 2)])
              for song in playlist.itertuples()]
    return tracks, journey


def journey_summary(journey):
    """The parts of an Uplift journey the player needs for its chart."""
    return {"start": journey["start_emotion"], "targets": journey["targets"],
            "waypoints": [[round(float(v), 2), round(float(a), 2)] for v, a in journey["waypoints"]]}


def render(tracks, title, subtitle, journey=None, loop_from=0):
    """The player page as an HTML string. Titles are inserted as text, never as HTML."""
    data = {"tracks": tracks, "title": title, "subtitle": subtitle, "journey": journey, "loopFrom": loop_from}
    payload = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return PLAYER_HTML.replace("__DATA__", payload)


PLAYER_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#121212; --panel:#181818; --row:#2a2a2a; --text:#ffffff; --muted:#b3b3b3; --green:#1db954; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  #list { position:absolute; top:0; left:0; right:0; bottom:88px; overflow-y:auto; padding:8px 20px 24px; }
  #list::-webkit-scrollbar { width:10px; } #list::-webkit-scrollbar-thumb { background:#3a3a3a; border-radius:5px; }
  .head { display:flex; align-items:center; gap:18px; padding:10px 4px 14px; }
  #bigplay { width:54px; height:54px; border-radius:50%; border:none; background:var(--green); cursor:pointer;
             display:flex; align-items:center; justify-content:center; flex:none; transition:transform .1s; }
  #bigplay:hover { transform:scale(1.06); } #bigplay svg { width:24px; height:24px; fill:#000; }
  .head h2 { margin:0; font-size:22px; } .head p { margin:3px 0 0; color:var(--muted); font-size:13px; }
  details.journey { background:var(--panel); border-radius:8px; margin:0 0 12px; padding:0 14px; }
  details.journey summary { cursor:pointer; padding:11px 0; font-size:14px; font-weight:600; color:var(--muted); }
  details.journey[open] summary { color:var(--text); }
  details.journey .cap { color:var(--muted); font-size:12px; margin:0 0 12px; }
  .section { color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; margin:16px 4px 6px; }
  .row { display:grid; grid-template-columns:34px 48px 1fr auto auto; align-items:center; gap:14px;
         padding:7px 12px 7px 4px; border-radius:6px; cursor:pointer; }
  .row:hover { background:#1f1f1f; } .row.playing { background:var(--row); }
  .num { text-align:center; color:var(--muted); font-size:15px; font-variant-numeric:tabular-nums; }
  .row.playing .num, .row.playing .title { color:var(--green); }
  .art { width:48px; height:48px; border-radius:4px; background:#333; object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#777; font-size:20px; }
  .meta { min-width:0; } .title { font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:13px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  .tag { font-size:11px; color:var(--muted); border:1px solid #3a3a3a; border-radius:10px; padding:2px 8px; white-space:nowrap; }
  .lastfm { font-size:12px; color:var(--muted); text-decoration:none; white-space:nowrap; }
  .lastfm:hover { color:var(--text); text-decoration:underline; }
  .eq { display:inline-flex; gap:2px; align-items:flex-end; height:14px; }
  .eq i { width:3px; background:var(--green); animation:eq 0.9s ease-in-out infinite; }
  .eq i:nth-child(2) { animation-delay:.25s; } .eq i:nth-child(3) { animation-delay:.5s; }
  .paused .eq i { animation-play-state:paused; }
  @keyframes eq { 0%,100% { height:4px; } 50% { height:14px; } }
  #bar { position:absolute; left:0; right:0; bottom:0; height:88px; background:var(--panel); border-top:1px solid #282828;
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
  #right { text-align:right; color:var(--muted); font-size:12px; }
  .empty { color:var(--muted); padding:40px 8px; }
</style></head>
<body>
<div id="list">
  <div class="head">
    <button id="bigplay" title="Play" aria-label="Play"></button>
    <div><h2 id="qtitle"></h2><p id="qsub"></p></div>
  </div>
  <div id="journey"></div>
  <div id="rows"></div>
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
  <div id="right"></div>
</div>
<script>
const DATA = __DATA__;
const tracks = DATA.tracks;
const PLAY = '<svg viewBox="0 0 16 16"><path d="M3 1.5v13l11-6.5z"/></svg>';
const PAUSE = '<svg viewBox="0 0 16 16"><path d="M3 1.5h3.5v13H3zM9.5 1.5H13v13H9.5z"/></svg>';
const SECTIONS = { journey: "Your journey, in play order", more: "Keep listening", mood: "For your mood" };
const audio = new Audio();
audio.preload = "auto";
let current = -1, failures = 0;
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const fmt = s => { s = Math.max(0, Math.floor(s || 0)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };

function artNode(track, cls) {
  if (track.art) { const img = el("img", cls); img.src = track.art; img.alt = ""; img.loading = "lazy"; return img; }
  return el("div", cls + " noart", "♫");
}

function buildRows() {
  $("qtitle").textContent = DATA.title; $("qsub").textContent = DATA.subtitle;
  const rows = $("rows");
  if (!tracks.length) { rows.appendChild(el("div", "empty", "No playable songs were found for this mood right now. Try Re-scan in a minute.")); return; }
  let section = null;
  tracks.forEach((t, i) => {
    if (t.section !== section) {
      section = t.section;
      let label = SECTIONS[section] || section;
      if (section === "more") label += " · " + t.bucket;
      rows.appendChild(el("div", "section", label));
    }
    const row = el("div", "row"); row.dataset.index = i;
    row.appendChild(el("div", "num", String(t.number)));
    row.appendChild(artNode(t, "art"));
    const meta = el("div", "meta"); meta.appendChild(el("div", "title", t.title)); meta.appendChild(el("div", "artist", t.artist)); row.appendChild(meta);
    row.appendChild(el("span", "tag", t.bucket));
    const link = el("a", "lastfm", "Last.fm"); link.href = t.lastfm; link.target = "_blank"; link.rel = "noopener noreferrer";
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
  });
  $("toggle").innerHTML = playing ? PAUSE : PLAY; $("bigplay").innerHTML = playing ? PAUSE : PLAY;
  if (current >= 0) {
    const t = tracks[current];
    $("nowtitle").textContent = t.title; $("nowartist").textContent = t.artist;
    const old = $("nowart"), art = artNode(t, "art"); art.id = "nowart"; old.replaceWith(art);
    $("right").textContent = (current + 1) + " of " + tracks.length;
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

buildRows(); buildJourney(); refresh();
window.player = { audio, state: () => ({ current, paused: audio.paused, time: audio.currentTime, src: audio.src }) };
</script>
</body></html>
"""
