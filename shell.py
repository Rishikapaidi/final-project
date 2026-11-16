"""The app's shell: the top bar, the profile menu, the sidebar's playlists,
the Explore song list and the Library, as HTML/JS components.

Like the player (player.py), they are st.components.v1.html iframes on the
same origin as the page, and they share player.BASE_JS: liked songs,
playlists, recently played, your name and the settings are kept in this
browser only (localStorage, or an in-memory copy on the page if that is
blocked) and are never sent to the server. They talk to each other and to
the player bar over the page-wide BroadcastChannel "moodtune".

All but the Explore list take no data from the server, so Streamlit never
reloads them on a rerun. Moving between Home, Explore and Library is done by
the sidebar's Streamlit buttons (app.py); the playlists here press the
Library button on the page to get there.
"""
import json

import ui
from player import BASE_JS, FONT_CSS, MENU_CSS, MENU_JS, THEME_HEAD

# Send feedback (the avatar's menu) opens this page in a new tab: the project's GitHub Issues.
FEEDBACK_URL = "https://github.com/Rishikapaidi/final-project/issues"

# Shared by the components below: a button reset, icon buttons and focus rings.
BASE_CSS = """
  * { box-sizing:border-box; }
  html, body { margin:0; background:transparent; color:var(--ink); font-family:var(--sans); }
  svg.s { fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }
  svg.f { fill:currentColor; }
  .icon-btn { width:38px; height:38px; border:none; background:none; border-radius:999px; color:var(--muted); cursor:pointer;
              display:flex; align-items:center; justify-content:center; padding:0; flex:none; }
  .icon-btn:hover, .icon-btn[aria-expanded="true"] { color:var(--purple); background:var(--hover-soft); }
  .icon-btn svg { width:20px; height:20px; }
  button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible, [tabindex]:focus-visible {
       outline:3px solid var(--lavender); outline-offset:2px; }
"""


def _page(css, body, script, data=None):
    page = (f'<!DOCTYPE html>\n<html><head><meta charset="utf-8">{THEME_HEAD}<style>{FONT_CSS}{BASE_CSS}{css}</style></head>'
            f'<body>{body}<script>\n')
    if data is not None:
        page += "const DATA = " + json.dumps(data, ensure_ascii=True).replace("</", "<\\/") + ";\n"
    return page + BASE_JS + "\n" + script + "\n</script></body></html>"


# ---------------------------------------------------------------- top bar
def render_topbar():
    """The left of the top bar: the menu (hamburger) button, the logo and the search box.

    The hamburger folds the sidebar to icons (wide windows) or slides it in
    over the page (narrow windows); the choice for wide windows is kept in
    this browser. The search box filters the song list on the page as you
    type (see BASE_JS: searchQuery, onSearch); nothing is sent anywhere.
    """
    return _page(TOP_CSS, TOP_BODY.replace("__LOGO__", ui.LOGO_SVG), TOP_JS)


TOP_CSS = """
  html, body { height:100%; overflow:hidden; }
  #top { display:flex; align-items:center; gap:10px; height:100%; padding:0 6px 0 4px; }
  .brand { display:flex; align-items:center; gap:10px; flex:none; margin-right:8px; text-decoration:none; }
  .brand .logo { width:34px; height:34px; display:block; }
  .brand span { font-family:var(--serif); font-size:20px; font-weight:700; color:var(--purple); letter-spacing:.06em; white-space:nowrap; }
  .search { flex:1; min-width:0; max-width:560px; display:flex; align-items:center; gap:8px; background:var(--field); border:1px solid var(--line);
            border-radius:999px; padding:0 14px; color:var(--muted); height:42px; }
  .search:focus-within { border-color:var(--lavender); box-shadow:0 0 0 3px rgba(185,176,232,.35); }
  .search svg { width:17px; height:17px; flex:none; }
  #q { flex:1; min-width:0; border:none; outline:none; background:none; font:inherit; font-size:14.5px; color:var(--ink); padding:0; height:100%; }
  #q:focus-visible { outline:none; }
  #hamburger[aria-expanded] { background:none; } #hamburger:hover { background:var(--hover-soft); }
  @media (max-width: 520px) { .brand span { display:none; } .brand { margin-right:0; } .search { padding:0 10px; } }
  @media (max-width: 250px) { .brand { display:none; } }
"""
TOP_BODY = """
<div id="top">
  <button class="icon-btn" id="hamburger" aria-label="Show or hide the menu" title="Menu" aria-controls="moodtune-nav" aria-expanded="true"></button>
  <div class="brand" aria-label="MOOD TUNE">__LOGO__<span>MOOD TUNE</span></div>
  <label class="search" title="Search the songs on this page"><span id="searchicon"></span>
    <input id="q" type="search" placeholder="Search songs on this page" aria-label="Search the songs on this page, by title or artist" autocomplete="off"></label>
</div>
"""
TOP_JS = r"""
$("hamburger").innerHTML = icon("hamburger"); $("searchicon").outerHTML = icon("search");
const page = window.parent, root = page.document.documentElement;
const NAV = "moodtune.nav";                  // "mini" or "full", for wide windows
const narrow = () => page.innerWidth <= 760;
function applyNav() {
  root.classList.toggle("mt-nav-mini", !narrow() && load(NAV, "full") === "mini");
  if (!narrow()) root.classList.remove("mt-nav-open");
  $("hamburger").setAttribute("aria-expanded", narrow() ? root.classList.contains("mt-nav-open") : !root.classList.contains("mt-nav-mini"));
}
// Narrow windows: the sidebar slides in over the page, with a shade to tap to close it.
function shade() {
  let s = page.document.getElementById("moodtune-navshade");
  if (!s) { s = page.document.createElement("div"); s.id = "moodtune-navshade"; page.document.body.appendChild(s); }
  s.onclick = () => openNav(false);
  return s;
}
function openNav(on) { shade(); root.classList.toggle("mt-nav-open", on); applyNav(); }
$("hamburger").addEventListener("click", () => {
  if (narrow()) openNav(!root.classList.contains("mt-nav-open"));
  else { save(NAV, root.classList.contains("mt-nav-mini") ? "full" : "mini"); applyNav(); }
});
// Picking a page in the slid-in sidebar closes it; so does Escape. One listener, replaced by each new top bar.
try {
  if (page.__moodtuneNav) { page.document.removeEventListener("click", page.__moodtuneNav.click, true); page.document.removeEventListener("keydown", page.__moodtuneNav.key); }
  page.__moodtuneNav = {
    click: e => { try { if (narrow() && e.target.closest && e.target.closest(".st-key-nav button")) openNav(false); } catch (err) {} },
    key: e => { try { if (e.key === "Escape" && root.classList.contains("mt-nav-open")) openNav(false); } catch (err) {} },
  };
  page.document.addEventListener("click", page.__moodtuneNav.click, true);
  page.document.addEventListener("keydown", page.__moodtuneNav.key);
  page.addEventListener("resize", applyNav);
} catch (e) {}
applyNav();

// Search: the query lives on the page (not in storage) and goes to the lists over the hub.
$("q").value = searchQuery();
function search(q) { try { page.__moodtuneSearch = q; } catch (e) {} hub.postMessage({ type: "search", q }); }
$("q").addEventListener("input", () => search($("q").value.trim()));
$("q").addEventListener("keydown", e => { if (e.key === "Escape" && $("q").value) { e.preventDefault(); $("q").value = ""; search(""); } });
// Shortcut keys pressed here (outside the box) go to the player bar.
document.addEventListener("keydown", e => { const key = shortcut(e); if (!key) return; e.preventDefault(); if (key !== "held") hub.postMessage({ type: "key", key }); });
window.topbar = { search: q => { $("q").value = q; search(q); }, toggleNav: () => $("hamburger").click() };
"""


# ---------------------------------------------------------------- profile
def render_profile():
    """The right of the top bar: the avatar, its menu and the drawer it opens.

    The avatar shows the first letter of the name you give it (kept in this
    browser), or a person icon. The menu: the name ("Local profile", with a
    pencil to change it), Settings, Liked songs and History (open the
    Library), Keyboard shortcuts, Privacy, About, Send feedback (FEEDBACK_URL,
    a new tab) and Reset profile (after a confirm, clears everything kept in
    this browser and the settings cookie). While the menu or the drawer is
    open the iframe is pinned over the whole window.

    Settings never rerun the app: the theme and "Larger text" put a class on
    the page, "Show all 7 emotions" opens the mood card's expander, autoplay
    and repeat are read by the player bar. The theme, default language and
    default mode also go into the cookie "moodtune_prefs", which the server
    reads when the app opens (app.py), so they apply from the next visit
    (the theme straight away as well).
    """
    return _page(PROFILE_CSS, PROFILE_BODY.replace("__FEEDBACK_URL__", FEEDBACK_URL), PROFILE_JS)


PROFILE_CSS = """
  html, body { height:100%; overflow:hidden; }
  #buttons { display:flex; align-items:center; justify-content:flex-end; height:56px; padding:0 6px; }
  .avatar { width:38px; height:38px; border-radius:50%; border:none; background:var(--purple); color:var(--on-purple); font:inherit; font-size:16px;
            font-weight:700; cursor:pointer; display:flex; align-items:center; justify-content:center; padding:0; flex:none; text-transform:uppercase; }
  .avatar:hover { background:var(--purple-hover); } .avatar svg { width:20px; height:20px; }
  .avatar.big { width:46px; height:46px; font-size:19px; cursor:default; } .avatar.big:hover { background:var(--purple); }
  #overlay { position:fixed; inset:0; } #overlay[hidden] { display:none; }
  #backdrop { position:absolute; inset:0; }
  #overlay.drawer #backdrop { background:rgba(43,40,64,.38); }
  /* the profile menu, under the avatar */
  #pmenu { position:absolute; width:300px; max-height:calc(100vh - 80px); overflow:auto; background:var(--field); border:1px solid var(--line); border-radius:16px; padding:8px;
           box-shadow:0 18px 50px rgba(45,42,82,.22); }
  #pmenu[hidden] { display:none; }
  .who { display:flex; align-items:center; gap:12px; padding:10px 10px 12px; border-bottom:1px solid var(--line); margin-bottom:6px; }
  .who b { display:block; font-size:15px; } .who small { display:block; color:var(--muted); font-size:12.5px; margin-top:2px; }
  .pi { display:flex; align-items:center; gap:12px; width:100%; border:none; background:none; font:inherit; font-size:14.5px; color:var(--ink);
        padding:10px 12px; border-radius:10px; cursor:pointer; text-align:left; }
  .pi:hover, .pi:focus { background:var(--lav-soft); outline:none; } .pi svg { width:19px; height:19px; color:var(--muted); flex:none; }
  a.pi { text-decoration:none; } .pi .ext { margin-left:auto; color:var(--muted); font-size:13px; }
  .pi.danger, .pi.danger svg { color:var(--danger); }
  .sep { border-top:1px solid var(--line); margin:6px 4px; }
  .whotext { flex:1; min-width:0; } .whotext b { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #m-name { width:34px; height:34px; } #m-name svg { width:17px; height:17px; }
  select { font:inherit; font-size:14px; color:var(--ink); background:var(--field); border:1px solid var(--line); border-radius:10px; padding:7px 10px; flex:none; }
  .keys { display:grid; grid-template-columns:auto 1fr; gap:10px 16px; align-items:center; margin:0; }
  .keys dt { margin:0; } .keys dd { margin:0; font-size:14px; }
  kbd { display:inline-block; min-width:30px; text-align:center; font:inherit; font-size:13px; font-weight:600; border:1px solid var(--line);
        border-bottom-width:2px; border-radius:7px; padding:3px 8px; background:var(--field); }
  #v-privacy h3, #v-reset h3 { font-family:var(--serif); font-size:18px; font-weight:600; margin:18px 0 6px; }
  #v-privacy p, #v-reset p, #v-reset li { font-size:14px; line-height:1.6; margin:0 0 8px; }
  #v-reset ul { padding-left:20px; margin:0 0 10px; }
  .actions { display:flex; gap:8px; margin-top:12px; }
  /* the drawer: Settings and About */
  #drawer { position:absolute; top:0; right:0; bottom:0; width:min(440px, 100%); background:var(--cream); display:flex; flex-direction:column;
            box-shadow:-14px 0 44px rgba(45,42,82,.22); animation:slide .22s ease; }
  #drawer[hidden] { display:none; }
  @keyframes slide { from { transform:translateX(28px); opacity:0; } }
  @media (prefers-reduced-motion: reduce) { #drawer { animation:none; } }
  header { display:flex; align-items:center; gap:6px; padding:22px 14px 6px 24px; }
  h2 { font-family:var(--serif); font-weight:600; font-size:27px; margin:0; flex:1; letter-spacing:-.01em; }
  .view { overflow:auto; flex:1; padding:6px 24px 28px; } .view[hidden] { display:none; }
  .hint { color:var(--muted); font-size:13px; line-height:1.5; margin:0 0 14px; }
  .btn { border:1px solid var(--line); background:var(--field); color:var(--ink); border-radius:999px; font:inherit; font-size:13.5px; font-weight:600;
         padding:8px 16px; cursor:pointer; display:inline-flex; align-items:center; gap:7px; }
  .btn:hover { border-color:var(--lavender); }
  .btn.danger { color:var(--danger); } .btn.danger.primary { background:#9b2f2f; border-color:#9b2f2f; color:#fff; }
  .confirm { background:var(--field); border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin:0 0 12px; font-size:14px; }
  .confirm[hidden] { display:none; } .confirm p { margin:0 0 10px; }
  #back { margin-left:-10px; } #back[hidden] { display:none; }
  .setting { display:flex; align-items:center; justify-content:space-between; gap:16px; background:var(--field); border:1px solid var(--line);
             border-radius:14px; padding:14px 16px; margin:0 0 10px; cursor:pointer; }
  .setting b { display:block; font-size:14.5px; font-weight:600; } .setting small { display:block; color:var(--muted); font-size:12.5px; line-height:1.45; margin-top:3px; }
  input[type="checkbox"][role="switch"] { appearance:none; flex:none; width:44px; height:26px; border-radius:999px; background:var(--switch-off); position:relative;
         cursor:pointer; margin:0; transition:background .15s; }
  input[type="checkbox"][role="switch"]::after { content:""; position:absolute; top:3px; left:3px; width:20px; height:20px; border-radius:50%; background:var(--field);
         box-shadow:0 1px 3px rgba(0,0,0,.2); transition:left .15s; }
  input[type="checkbox"][role="switch"]:checked { background:var(--purple); }
  input[type="checkbox"][role="switch"]:checked::after { left:21px; }
  fieldset.setting { display:block; cursor:default; } fieldset.setting legend { float:left; width:100%; padding:0; margin:0 0 10px; }
  .seg { display:flex; background:var(--lav-soft); border-radius:999px; padding:3px; clear:both; }
  .seg label { flex:1; text-align:center; font-size:13.5px; padding:7px 0; border-radius:999px; cursor:pointer; color:var(--muted); font-weight:600; }
  .seg input { position:absolute; opacity:0; pointer-events:none; }
  .seg label:has(input:checked) { background:var(--field); color:var(--purple); box-shadow:0 1px 3px rgba(45,42,82,.15); }
  .seg label:has(input:focus-visible) { outline:3px solid var(--lavender); }
  .group { font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:18px 2px 8px; }
  .link-row { display:flex; width:100%; align-items:center; justify-content:space-between; background:var(--field); border:1px solid var(--line); border-radius:14px;
              padding:14px 16px; font:inherit; font-size:14.5px; font-weight:600; color:var(--ink); cursor:pointer; text-align:left; margin:0 0 10px; }
  .link-row:hover { border-color:var(--lavender); }
  .link-row small { display:block; font-weight:400; color:var(--muted); font-size:12.5px; margin-top:3px; }
  .link-row[disabled] { cursor:default; color:var(--muted); } .link-row[disabled]:hover { border-color:var(--line); }
  #v-about h3 { font-family:var(--serif); font-size:18px; font-weight:600; margin:20px 0 6px; }
  #v-about p, #v-about li { font-size:14px; line-height:1.6; margin:0 0 8px; }
  #v-about ul { padding-left:20px; margin:0 0 8px; }
"""
PROFILE_BODY = """
<div id="buttons"><button class="avatar" id="avatar" aria-haspopup="menu" aria-expanded="false" aria-controls="pmenu"></button></div>
<div id="overlay" hidden>
  <div id="backdrop"></div>
  <div id="pmenu" role="menu" aria-label="Your profile" hidden>
    <div class="who"><div class="avatar big" id="bigavatar" aria-hidden="true"></div>
      <div class="whotext"><b id="whoname"></b><small>Local profile</small></div>
      <button class="icon-btn" id="m-name" role="menuitem" title="Change your name" aria-label="Change your name"></button></div>
    <button class="pi" id="m-settings" role="menuitem"><span data-icon="gear"></span>Settings</button>
    <button class="pi" id="m-liked" role="menuitem"><span data-icon="heart"></span>Liked songs</button>
    <button class="pi" id="m-history" role="menuitem"><span data-icon="history"></span>History</button>
    <button class="pi" id="m-keys" role="menuitem"><span data-icon="keyboard"></span>Keyboard shortcuts</button>
    <button class="pi" id="m-privacy" role="menuitem"><span data-icon="shield"></span>Privacy</button>
    <button class="pi" id="m-about" role="menuitem"><span data-icon="info"></span>About MOOD TUNE</button>
    <a class="pi" id="m-feedback" role="menuitem" href="__FEEDBACK_URL__" target="_blank" rel="noopener noreferrer"><span data-icon="message"></span>Send feedback<span class="ext" aria-hidden="true">↗</span></a>
    <div class="sep" role="separator"></div>
    <button class="pi danger" id="m-reset" role="menuitem"><span data-icon="reset"></span>Reset profile</button>
  </div>
  <section id="drawer" role="dialog" aria-modal="true" aria-labelledby="dtitle" hidden>
    <header><button class="icon-btn" id="back" title="Back to Settings" aria-label="Back to Settings" hidden></button>
      <h2 id="dtitle">Settings</h2><button class="icon-btn" id="dclose" title="Close" aria-label="Close"></button></header>
    <div class="view" id="v-settings" data-title="Settings">
      <p class="hint">Changes apply straight away, without stopping the music. They are kept in this browser only.</p>
      <div class="group">Appearance</div>
      <fieldset class="setting"><legend><b>Theme</b><small>Light is the cream and purple look; Dark is easier on the eyes at night.</small></legend>
        <div class="seg" role="radiogroup" aria-label="Theme">
          <label><input type="radio" name="theme" value="light">Light</label><label><input type="radio" name="theme" value="dark">Dark</label></div></fieldset>
      <label class="setting"><span><b>Larger text</b><small>Makes the page's text and cards bigger.</small></span>
        <input type="checkbox" role="switch" id="s-large"></label>
      <label class="setting"><span><b>Show all 7 emotions</b><small>Open "See all 7 emotions" on the mood card by default.</small></span>
        <input type="checkbox" role="switch" id="s-emotions"></label>
      <div class="group">When MOOD TUNE opens</div>
      <label class="setting"><span><b>Default language</b><small>"Songs in" starts with it the next time MOOD TUNE opens.</small></span>
        <select id="s-language" aria-label="Default language"></select></label>
      <fieldset class="setting"><legend><b>Default mode</b><small>What MOOD TUNE picks for a new mood reading, from the next time it opens. Auto picks "Lift my mood" for a low mood and "Meet me here" otherwise.</small></legend>
        <div class="seg" role="radiogroup" aria-label="Default mode">
          <label><input type="radio" name="mode" value="auto">Auto</label><label><input type="radio" name="mode" value="resonance">Meet me here</label><label><input type="radio" name="mode" value="uplift">Lift my mood</label></div></fieldset>
      <div class="group">Playback</div>
      <label class="setting"><span><b>Autoplay next song</b><small>When a song ends, the next one starts.</small></span>
        <input type="checkbox" role="switch" id="s-autoplay"></label>
      <fieldset class="setting"><legend><b>Repeat</b><small>Where the player starts. The repeat button in the player changes it for now.</small></legend>
        <div class="seg" role="radiogroup" aria-label="Default repeat">
          <label><input type="radio" name="repeat" value="off">Off</label><label><input type="radio" name="repeat" value="all">All</label><label><input type="radio" name="repeat" value="one">This song</label></div></fieldset>
      <div class="group">Your data</div>
      <button class="link-row" id="s-clear"><span>Clear liked songs <small id="s-count"></small></span><span aria-hidden="true">›</span></button>
      <button class="link-row" id="s-clearrecent"><span>Clear recently played <small id="s-recentcount"></small></span><span aria-hidden="true">›</span></button>
      <div class="confirm" id="confirm" hidden><p id="confirmtext"></p>
        <button class="btn danger primary" id="confirmyes">Clear</button> <button class="btn" id="confirmno">Cancel</button></div>
      <button class="link-row" id="openabout"><span>About MOOD TUNE</span><span aria-hidden="true">›</span></button>
    </div>
    <div class="view" id="v-keys" data-title="Keyboard shortcuts" hidden>
      <p class="hint">They work anywhere on the page, except while you are typing in a box.</p>
      <dl class="keys">
        <dt><kbd>Space</kbd></dt><dd>Play or pause</dd>
        <dt><kbd>→</kbd></dt><dd>Next song</dd>
        <dt><kbd>←</kbd></dt><dd>Previous song (or the start of this one)</dd>
        <dt><kbd>M</kbd></dt><dd>Mute or unmute</dd>
        <dt><kbd>L</kbd></dt><dd>Add the song playing to Liked songs, or remove it</dd>
        <dt><kbd>Esc</kbd></dt><dd>Close a menu, the queue or this panel</dd>
      </dl>
    </div>
    <div class="view" id="v-privacy" data-title="Privacy" hidden>
      <h3>Your camera and photos</h3>
      <p>The camera image is processed in memory and never saved. The same goes for a photo you upload: it is read once, in memory, to find a face.</p>
      <h3>Your words</h3>
      <p>What you type is read on this computer and kept only for this visit.</p>
      <h3>What stays in this browser</h3>
      <p>Liked songs, your playlists, recently played (the last 50 songs, without times), your name and your settings are kept in this browser's storage, on this computer.
        Nothing about your mood, your scan or the time is stored.</p>
      <h3>What the server sees</h3>
      <p>Your theme, default language and default mode are also kept in a small cookie, so MOOD TUNE can open with them. The server reads only those three; nothing else you keep here is sent anywhere.</p>
      <h3>Music</h3>
      <p>Previews and artwork are loaded from Apple's servers. Song links open Last.fm or Spotify.</p>
      <p class="hint">Reset profile (in this menu) clears everything kept here.</p>
    </div>
    <div class="view" id="v-about" data-title="About MOOD TUNE" hidden>
      <p>MOOD TUNE reads your expression from the webcam (or a photo) and, if you like, a sentence about your day, and combines the two into a mood reading.
        Then it suggests music: songs that match how you feel ("Meet me here"), or a five-song path from a low mood towards a lighter one ("Lift my mood").
        Explore lets you browse the songs by mood and by language.</p>
      <h3>Where the songs come from</h3>
      <ul>
        <li>English songs: the Muse dataset (<code>muse_v3.csv</code>), with listeners' mood tags. Links go to Last.fm.</li>
        <li>Hindi, Tamil, Telugu, Malayalam and Korean songs: "Spotify Tracks Dataset (Updated Every Week)" by gauthamvijayaraj on Kaggle, Apache 2.0 licence. Links go to Spotify.</li>
        <li>Previews and artwork: 30-second previews from the iTunes Search API, played from Apple's servers.</li>
        <li>Fonts: Fraunces and Inter, SIL Open Font License.</li>
        <li>Face model: HSEmotion (enet_b0_8_best_afew) by Andrey Savchenko, code Apache-2.0, trained on AffectNet, whose licence allows non-commercial research and education only.</li>
      </ul>
      <h3>Limits</h3>
      <ul>
        <li>The face model (HSEmotion, trained on the AffectNet photos) reads about 6 in 10 faces correctly on its own benchmark. On the grayscale FER2013 test set it reads 55%, and it sometimes takes a neutral face for angry or sad. Subtle expressions are hard for any model.</li>
        <li>Only 30-second previews play, not whole songs.</li>
        <li>Each song's language is as labelled in its dataset; MOOD TUNE does not check it.</li>
        <li>Telugu and Malayalam have too few songs for "Lift my mood", so they offer "Meet me here" only.</li>
      </ul>
      <h3>Privacy</h3>
      <ul>
        <li>Your webcam picture (or photo) is analysed in memory and never saved.</li>
        <li>Your words are kept only for this visit.</li>
        <li>Liked songs, your playlists, recently played (the last 50 songs, without times or moods), your name and these settings stay in this browser, on this computer; only the theme, default language and default mode go to the server, in a cookie.</li>
      </ul>
    </div>
    <div class="view" id="v-reset" data-title="Reset profile" hidden>
      <p>This clears everything MOOD TUNE keeps in this browser:</p>
      <ul>
        <li id="r-liked"></li><li id="r-lists"></li><li id="r-recent"></li>
        <li>your name, and all settings (theme, defaults, volume, the folded sidebar), and the settings cookie</li>
      </ul>
      <p class="hint">It can't be undone. The music playing and the songs from your mood reading stay as they are.</p>
      <div class="actions"><button class="btn danger primary" id="resetyes">Reset profile</button> <button class="btn" id="resetno">Cancel</button></div>
    </div>
  </section>
</div>
"""
PROFILE_JS = r"""
let opener = null, mode = null, fromSettings = false;
const EXTRA_ICONS = {
  keyboard: '<rect x="2" y="6" width="20" height="12" rx="2"/><line x1="6" y1="10" x2="6" y2="10"/><line x1="10" y1="10" x2="10" y2="10"/><line x1="14" y1="10" x2="14" y2="10"/><line x1="18" y1="10" x2="18" y2="10"/><line x1="7" y1="14" x2="17" y2="14"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  reset: '<path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 4 3 9 8 9"/>',
};
const anyIcon = name => ICONS[name] ? icon(name) : '<svg class="s" viewBox="0 0 24 24" aria-hidden="true">' + EXTRA_ICONS[name] + '</svg>';
document.querySelectorAll("[data-icon]").forEach(e => { e.outerHTML = anyIcon(e.dataset.icon); });
$("dclose").innerHTML = icon("close"); $("back").innerHTML = icon("back"); $("m-name").innerHTML = icon("pencil");
LANGS.forEach(l => { const o = el("option", null, l); o.value = l; $("s-language").appendChild(o); });

function renderAvatar() {
  const name = profileName();
  [$("avatar"), $("bigavatar")].forEach(a => { if (name) a.textContent = [...name][0]; else a.innerHTML = icon("user"); });
  $("avatar").title = name ? name + ": your profile" : "Your profile";
  $("avatar").ariaLabel = name ? "Your profile, " + name : "Your profile";
  $("whoname").textContent = name || "Your profile";
  $("m-name").title = $("m-name").ariaLabel = name ? "Change your name" : "Add your name";
}
on("profile", renderAvatar);

// The page's own theme (its CSS is in ui.py): one class on the page, and the cookie the server reads when the app opens.
const PREFS_COOKIE = "moodtune_prefs";
function applyPageTheme() {
  try { const r = window.parent.document.documentElement, dark = settings().theme === "dark"; r.classList.toggle("mt-dark", dark); r.classList.toggle("mt-light", !dark); } catch (e) {}
}
function writeCookie() {
  const s = settings(), isDefault = s.theme === DEFAULTS.theme && s.language === DEFAULTS.language && s.mode === DEFAULTS.mode;
  const value = encodeURIComponent("theme=" + s.theme + "&lang=" + s.language + "&mode=" + s.mode);
  try { window.parent.document.cookie = PREFS_COOKIE + "=" + (isDefault ? "; max-age=0" : value + "; max-age=31536000") + "; path=/; samesite=lax"; } catch (e) {}
}

// Opening: the iframe is pinned over the whole window, and the avatar is drawn where it was.
function pin(on) {
  const f = window.frameElement;
  if (!f) return;
  if (on) {
    const r = f.getBoundingClientRect();
    $("buttons").style.cssText = `position:fixed; left:${r.left}px; top:${r.top}px; width:${r.width}px;`;
    Object.assign(f.style, { position: "fixed", left: "0", top: "0", width: "100vw", height: "100vh", zIndex: "1002" });
  } else {
    Object.assign(f.style, { position: "", left: "", top: "", width: "", height: "", zIndex: "" });
    $("buttons").style.cssText = "";
  }
}
function openMenu() {
  pin(true); mode = "menu"; opener = $("avatar");
  const a = $("avatar").getBoundingClientRect();
  $("overlay").hidden = false; $("overlay").className = "menu"; $("drawer").hidden = true; $("pmenu").hidden = false;
  $("pmenu").style.top = (a.bottom + 8) + "px"; $("pmenu").style.right = Math.max(8, window.innerWidth - a.right) + "px";
  $("avatar").setAttribute("aria-expanded", "true");
  $("m-settings").focus();
}
function openDrawer(view) {
  if (mode === null) pin(true);
  mode = "drawer"; opener = opener || $("avatar");
  $("overlay").hidden = false; $("overlay").className = "drawer"; $("pmenu").hidden = true; $("drawer").hidden = false;
  $("avatar").setAttribute("aria-expanded", "false");
  fromSettings = false;
  showView(view);
}
function showView(view) {
  document.querySelectorAll(".view").forEach(v => { v.hidden = v.id !== view; });
  $("dtitle").textContent = $(view).dataset.title;
  $("back").hidden = !(view === "v-about" && fromSettings);
  $("confirm").hidden = true;
  $(view).scrollTop = 0;
  renderSettings(); renderReset();
  ($("back").hidden ? $("dclose") : $("back")).focus();
}
function closeAll() {
  if (mode === null) return;
  $("overlay").hidden = true; $("confirm").hidden = true; mode = null;
  $("avatar").setAttribute("aria-expanded", "false");
  pin(false);
  if (opener) opener.focus();
  opener = null;
}
$("avatar").addEventListener("click", () => (mode ? closeAll() : openMenu()));
$("backdrop").addEventListener("click", closeAll);
$("dclose").addEventListener("click", closeAll);
$("m-settings").addEventListener("click", () => openDrawer("v-settings"));
$("m-keys").addEventListener("click", () => openDrawer("v-keys"));
$("m-privacy").addEventListener("click", () => openDrawer("v-privacy"));
$("m-about").addEventListener("click", () => openDrawer("v-about"));
$("m-reset").addEventListener("click", () => openDrawer("v-reset"));
$("m-liked").addEventListener("click", () => { closeAll(); openInLibrary("liked"); });
$("m-history").addEventListener("click", () => { closeAll(); openInLibrary("recent"); });
$("m-feedback").addEventListener("click", () => closeAll());
$("m-name").addEventListener("click", () => {
  closeAll();
  askName({ title: profileName() ? "Change your name" : "Add your name", label: "Your name", value: profileName(), ok: "Save", max: 40, allowEmpty: true,
            hint: "Only its first letter is shown, on the avatar. Kept in this browser only; leave it empty to remove it.",
            onOk: name => { setProfileName(name); toast(name ? "Hello, " + name : "Your name is removed"); } });
});
$("openabout").addEventListener("click", () => { fromSettings = true; showView("v-about"); });
$("back").addEventListener("click", () => showView("v-settings"));

// Settings. Theme, "Larger text" and "Show all 7 emotions" act on the page itself, so they never rerun
// the app; autoplay and repeat are read by the player bar; the default language and mode are for the next visit.
function renderSettings() {
  const s = settings(), n = savedSongs().length, r = recentSongs().length;
  $("s-autoplay").checked = s.autoplay; $("s-emotions").checked = s.emotions; $("s-large").checked = s.large;
  $("s-language").value = s.language;
  ["repeat", "theme", "mode"].forEach(k => document.querySelectorAll(`input[name="${k}"]`).forEach(x => { x.checked = x.value === s[k]; }));
  $("s-count").textContent = n ? plural(n, "song", "songs") + " in Liked songs" : "Nothing liked yet";
  $("s-recentcount").textContent = r ? plural(r, "song", "songs") + " in Recently played" : "Nothing played yet";
  $("s-clear").disabled = !n; $("s-clearrecent").disabled = !r;
}
function renderReset() {
  const n = savedSongs().length, lists = playlists(), songs = lists.reduce((a, p) => a + p.songs.length, 0), r = recentSongs().length;
  $("r-liked").textContent = "Liked songs (" + plural(n, "song", "songs") + ")";
  $("r-lists").textContent = "your playlists (" + plural(lists.length, "playlist", "playlists") + ", " + plural(songs, "song", "songs") + ")";
  $("r-recent").textContent = "Recently played (" + plural(r, "song", "songs") + ")";
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
on("settings", () => { renderSettings(); applyLarge(); applyPageTheme(); writeCookie(); });
["saved", "recent", "playlists"].forEach(t => on(t, () => { renderSettings(); renderReset(); }));
$("s-autoplay").addEventListener("change", e => setSetting("autoplay", e.target.checked));
$("s-emotions").addEventListener("change", e => { setSetting("emotions", e.target.checked); applyEmotions(true); });
$("s-large").addEventListener("change", e => setSetting("large", e.target.checked));
$("s-language").addEventListener("change", e => { setSetting("language", e.target.value); toast("MOOD TUNE will open with songs in " + e.target.value); });
["repeat", "theme", "mode"].forEach(k => document.querySelectorAll(`input[name="${k}"]`).forEach(x => x.addEventListener("change", () => setSetting(k, x.value))));
let confirmRun = null, confirmBack = null;
function ask(text, run, back) { $("confirmtext").textContent = text; confirmRun = run; confirmBack = back; $("confirm").hidden = false; $("confirmno").focus(); }
$("s-clear").addEventListener("click", () => { const n = savedSongs().length; if (n) ask("Remove all " + plural(n, "song", "songs") + " from Liked songs?", () => { setSaved([]); toast("Liked songs is empty"); }, $("s-clear")); });
$("s-clearrecent").addEventListener("click", () => { const n = recentSongs().length; if (n) ask("Clear Recently played (" + plural(n, "song", "songs") + ")?", () => { clearRecent(); toast("Recently played is cleared"); }, $("s-clearrecent")); });
$("confirmno").addEventListener("click", () => { $("confirm").hidden = true; if (confirmBack) confirmBack.focus(); });
$("confirmyes").addEventListener("click", () => { $("confirm").hidden = true; if (confirmRun) confirmRun(); $("dclose").focus(); });

// Reset profile: everything MOOD TUNE keeps in this browser, and the settings cookie.
const LOCAL_KEYS = ["moodtune.saved", "moodtune.playlists", "moodtune.recent", "moodtune.profile", "moodtune.settings", "moodtune.volume", "moodtune.nav"];
function resetProfile() {
  const memory = pageMemory();
  LOCAL_KEYS.forEach(k => { delete memory[k]; try { localStorage.removeItem(k); } catch (e) {} });
  try { window.parent.document.documentElement.classList.remove("mt-nav-mini"); } catch (e) {}
  ["saved", "playlists", "recent", "profile", "settings"].forEach(changed);
  writeCookie();
}
$("resetyes").addEventListener("click", () => { resetProfile(); closeAll(); toast("Your profile is reset"); });
$("resetno").addEventListener("click", closeAll);

// Escape closes; Tab stays inside the menu or drawer; arrows move in the menu; shortcut keys go to the player bar.
document.addEventListener("keydown", e => {
  if (mode) {
    if (e.key === "Escape") { e.preventDefault(); closeAll(); return; }
    const box = mode === "menu" ? $("pmenu") : $("drawer");
    if (mode === "menu" && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      const items = [...box.querySelectorAll(".pi")], k = items.indexOf(document.activeElement);
      e.preventDefault(); items[(k + (e.key === "ArrowDown" ? 1 : items.length - 1)) % items.length].focus(); return;
    }
    if (e.key === "Tab") {
      const items = [...box.querySelectorAll("button, a, input, select")].filter(x => !x.disabled && x.offsetParent);
      const first = items[0], last = items[items.length - 1];
      if (mode === "menu") { e.preventDefault(); closeAll(); return; }
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
renderAvatar(); renderSettings(); renderReset(); applyLarge(); applyEmotions(false); applyPageTheme(); writeCookie();
window.actions = { open: v => (v === "menu" ? openMenu() : openDrawer(["v-about", "v-keys", "v-privacy", "v-reset"].includes(v) ? v : "v-settings")),
                   close: closeAll, settings, name: profileName, reset: resetProfile };
"""


# ---------------------------------------------------------------- sidebar playlists
def render_navlists():
    """Under the sidebar's Home / Explore / Library: "New playlist", then Liked songs,
    your playlists and Recently played. Picking one opens it in the Library."""
    return _page(NAVLISTS_CSS, NAVLISTS_BODY, NAVLISTS_JS)


NAVLISTS_CSS = """
  html, body { height:100%; }
  body { overflow-y:auto; overflow-x:hidden; padding:4px 2px 10px; }
  .new { display:flex; align-items:center; gap:10px; width:100%; border:1px solid var(--line); background:var(--field); color:var(--ink); font:inherit;
         font-size:14px; font-weight:600; border-radius:999px; padding:9px 16px; cursor:pointer; margin:0 0 12px; }
  .new:hover { border-color:var(--lavender); background:var(--hover); } .new svg { width:18px; height:18px; flex:none; }
  ul { list-style:none; margin:0; padding:0; }
  .item { display:grid; grid-template-columns:20px minmax(0,1fr); align-items:center; gap:12px; width:100%; border:none; background:none; font:inherit;
          color:var(--ink); text-align:left; padding:8px 12px; border-radius:10px; cursor:pointer; }
  .item:hover { background:var(--hover-soft); } .item.on { background:var(--lav-soft); }
  .item svg { width:18px; height:18px; color:var(--muted); } .item.liked svg { color:var(--terra); }
  .item b { display:block; font-size:14px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .item small { display:block; font-size:12px; color:var(--muted); margin-top:1px; }
"""
NAVLISTS_BODY = """
<button class="new" id="new"><span id="plusicon"></span>New playlist</button>
<nav aria-label="Your songs"><ul id="lists"></ul></nav>
"""
NAVLISTS_JS = r"""
$("plusicon").outerHTML = icon("plus");
let active = "";
function collections() {
  return [{ id: "liked", name: "Liked songs", n: savedSongs().length, icon: "heart", cls: "liked" }]
    .concat(playlists().map(p => ({ id: p.id, name: p.name, n: p.songs.length, icon: "list" })))
    .concat([{ id: "recent", name: "Recently played", n: recentSongs().length, icon: "history" }]);
}
function render() {
  const ul = $("lists"); ul.replaceChildren();
  collections().forEach(c => {
    const li = el("li"), b = el("button", "item" + (c.cls ? " " + c.cls : "") + (c.id === active ? " on" : ""));
    b.dataset.id = c.id; b.innerHTML = icon(c.icon);
    const text = el("span"); text.appendChild(el("b", null, c.name)); text.appendChild(el("small", null, plural(c.n, "song", "songs"))); b.appendChild(text);
    b.ariaLabel = c.name + ", " + plural(c.n, "song", "songs"); if (c.id === active) b.setAttribute("aria-current", "true");
    b.addEventListener("click", () => openInLibrary(c.id));
    li.appendChild(b); ul.appendChild(li);
  });
}
["saved", "playlists", "recent"].forEach(t => on(t, render));
hub.addEventListener("message", e => { if (e.data && e.data.type === "libraryView") { active = e.data.id || ""; render(); } });
$("new").addEventListener("click", () => askName({ title: "New playlist", label: "Playlist name", ok: "Create", hint: "Kept in this browser only.",
  onOk: name => { const id = newPlaylist(name); toast("Created “" + name + "”"); openInLibrary(id); } }));
document.addEventListener("keydown", e => { const key = shortcut(e); if (!key) return; e.preventDefault(); if (key !== "held") hub.postMessage({ type: "key", key }); });
render();
window.navlists = { open: openInLibrary, render };
"""


# ---------------------------------------------------------------- song rows (Explore and Library)
ROW_CSS = """
  .row { display:grid; grid-template-columns:30px 48px minmax(0,1fr) auto auto auto; align-items:center; gap:10px;
         padding:8px 8px 8px 6px; border-radius:12px; cursor:pointer; }
  .row[hidden] { display:none !important; }
  .row:hover { background:var(--hover); } .row.playing { background:var(--lav-soft); }
  .num { text-align:center; color:var(--muted); font-size:14.5px; font-variant-numeric:tabular-nums; }
  .row.playing .num, .row.playing .title { color:var(--purple); font-weight:600; }
  .art { width:48px; height:48px; border-radius:10px; background:var(--art-bg); object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:20px; }
  .meta { min-width:0; } .title { font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .artist { font-size:13px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  .lang { font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:1px 7px; margin-left:8px; background:var(--field); }
  .row .icon-btn { width:34px; height:34px; } .row .icon-btn svg { width:19px; height:19px; }
  .heart[aria-pressed="true"] { color:var(--terra); } .heart[aria-pressed="true"] svg { fill:currentColor; }
  .pill { border:none; background:var(--purple); color:var(--on-purple); font-weight:600; font-size:12.5px; border-radius:999px; font-family:var(--sans);
          padding:6px 14px; cursor:pointer; white-space:nowrap; min-width:76px; }
  .pill:hover { background:var(--purple-hover); }
  .eq { display:inline-flex; gap:2px; align-items:flex-end; height:14px; }
  .eq i { width:3px; background:currentColor; animation:eq .9s ease-in-out infinite; }
  .eq i:nth-child(2) { animation-delay:.25s; } .eq i:nth-child(3) { animation-delay:.5s; }
  .paused .eq i { animation-play-state:paused; }
  @keyframes eq { 0%,100% { height:4px; } 50% { height:14px; } }
  #matches { font-size:13px; color:var(--muted); margin:14px 0 4px; } #matches[hidden] { display:none; }
  #nomatch { color:var(--muted); font-size:14px; padding:14px 8px 6px; } #nomatch[hidden] { display:none; }
  #bigplay { width:52px; height:52px; border-radius:50%; border:none; background:var(--purple); cursor:pointer; flex:none;
             display:flex; align-items:center; justify-content:center; box-shadow:0 6px 16px rgba(45,42,82,.25); transition:transform .12s; color:var(--on-purple); }
  #bigplay:hover { transform:scale(1.05); } #bigplay svg { width:20px; height:20px; fill:var(--on-purple); }
  #bigplay[hidden] { display:none; }
  @media (max-width: 640px) {
    .row { grid-template-columns:22px 44px minmax(0,1fr) auto auto auto; gap:6px; padding:8px 2px; }
    .lang { display:none; } .art { width:44px; height:44px; } .pill { min-width:0; padding:6px 11px; }
  }
"""
ROW_JS = r"""
// What the player bar says is playing (by link), for the playing row and its pill.
let nowLink = "", nowPaused = true;
const playToggle = () => hub.postMessage({ type: "key", key: "space" });
const playSongsNow = songs => hub.postMessage({ type: "playSongs", songs });
const queueSongs = (songs, where) => hub.postMessage({ type: "insertSongs", songs, where });
// One song row. opts: { number, showLanguage, onPlay(), remove: { label, run } }
function songRow(t, opts) {
  const r = el("div", "row"); r.dataset.link = t.lastfm; r.tabIndex = 0; r.setAttribute("role", "button");
  r.setAttribute("aria-label", "Play " + t.title + " by " + t.artist);
  const num = el("div", "num", opts.number); num.dataset.n = opts.number; r.appendChild(num);
  r.appendChild(artNode(t, "art"));
  const meta = el("div", "meta"), artist = el("div", "artist", t.artist);
  meta.appendChild(el("div", "title", t.title));
  if (t.language && opts.showLanguage) artist.appendChild(el("span", "lang", t.language));
  meta.appendChild(artist); r.appendChild(meta);
  const heart = el("button", "icon-btn heart"); heart.innerHTML = icon("heart"); heart.dataset.link = t.lastfm;
  heart.addEventListener("click", e => { e.stopPropagation(); toggleSaved(t); });
  heart.addEventListener("keydown", e => e.stopPropagation()); r.appendChild(heart);
  const pill = el("button", "pill", "▶ Play"); pill.title = "Play this song";
  const play = () => (t.lastfm === nowLink ? playToggle() : opts.onPlay());
  pill.addEventListener("click", e => { e.stopPropagation(); play(); }); r.appendChild(pill);
  const more = el("button", "icon-btn more"); more.innerHTML = icon("more");
  more.title = "More options"; more.ariaLabel = "More options for " + t.title; more.setAttribute("aria-haspopup", "menu"); more.setAttribute("aria-expanded", "false");
  more.addEventListener("click", e => {
    e.stopPropagation();
    if (menuOpenFor(more)) return closeSongMenu(true);
    openSongMenu(more, t, { playNext: () => queueSongs([songOf(t)], "next"), addToQueue: () => queueSongs([songOf(t)], "queue"), remove: opts.remove });
  });
  more.addEventListener("keydown", e => e.stopPropagation()); r.appendChild(more);
  r.addEventListener("click", () => opts.onPlay());
  r.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); play(); } });
  return r;
}
function refreshRows() {
  document.querySelectorAll(".row").forEach(r => {
    const on = !!nowLink && r.dataset.link === nowLink, num = r.querySelector(".num");
    r.classList.toggle("playing", on); r.classList.toggle("paused", on && nowPaused);
    if (on) num.innerHTML = '<span class="eq" aria-hidden="true"><i></i><i></i><i></i></span>'; else num.textContent = num.dataset.n;
    r.querySelector(".pill").textContent = on && !nowPaused ? "❚❚ Pause" : "▶ Play";
  });
  refreshBigPlay();
}
function refreshHearts() {
  document.querySelectorAll(".heart").forEach(h => {
    const on = isSaved(h.dataset.link), title = h.closest(".row").querySelector(".title").textContent;
    h.setAttribute("aria-pressed", on); h.title = on ? "Remove from Liked songs" : "Add to Liked songs";
    h.ariaLabel = (on ? "Remove " : "Add ") + title + (on ? " from Liked songs" : " to Liked songs");
  });
}
on("saved", refreshHearts);
hub.addEventListener("message", e => { if (e.data && e.data.type === "now") { nowLink = e.data.link; nowPaused = e.data.paused; refreshRows(); } });
// Grow the iframe to fit its content, so the page scrolls instead of the list.
function fit() { if (window.frameElement) window.frameElement.style.height = Math.ceil(document.documentElement.scrollHeight) + "px"; }
new ResizeObserver(fit).observe(document.body);
document.addEventListener("keydown", e => { const key = shortcut(e); if (!key) return; e.preventDefault(); if (key !== "held") hub.postMessage({ type: "key", key }); });
"""

CARD_CSS = """
  html, body { overflow:hidden; }
  #card { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:26px 26px 18px;
          box-shadow:0 1px 2px rgba(45,42,82,.05), 0 8px 26px rgba(45,42,82,.07); margin:2px 2px 14px; }
  .kicker { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 6px; }
  .head { display:flex; align-items:flex-start; gap:18px; justify-content:space-between; }
  h2 { font-family:var(--serif); font-weight:600; font-size:30px; line-height:1.15; margin:0; letter-spacing:-.01em; }
  .sub { margin:6px 0 0; font-size:13.5px; color:var(--muted); line-height:1.5; }
  #rows { margin-top:14px; }
  .empty { color:var(--muted); font-size:14px; line-height:1.55; background:var(--card2); border:1px dashed var(--line); border-radius:14px; padding:18px; text-align:center; margin:14px 0 4px; }
  .empty[hidden] { display:none; }
  @media (max-width: 640px) { #card { padding:20px 14px 12px; border-radius:18px; } h2 { font-size:25px; } }
"""


# ---------------------------------------------------------------- Explore list
def render_songlist(list_id, title, subtitle, tracks):
    """A song list for Explore: Play all, and a row per song (heart, Play, "..." menu).

    Songs play through the player bar, after the song playing, like Play next;
    the queue it already has stays as it is. tracks: player._track dicts.
    """
    data = {"id": list_id, "title": title, "subtitle": subtitle,
            "tracks": [{k: t[k] for k in ("title", "artist", "lastfm", "link_label", "language", "bucket", "preview", "art")} for t in tracks]}
    return _page(CARD_CSS + ROW_CSS + MENU_CSS, LIST_BODY, MENU_JS + ROW_JS + LIST_JS, data)


LIST_BODY = """
<div id="card">
  <div class="head"><div><p class="kicker">Explore</p><h2 id="title"></h2><p class="sub" id="sub"></p></div>
    <button id="bigplay" title="Play all" aria-label="Play all"></button></div>
  <p id="matches" aria-live="polite" hidden></p>
  <div id="rows"></div>
  <div id="nomatch" hidden>No songs in this list match your search.</div>
</div>
<div id="menu" role="menu" hidden></div>
"""
LIST_JS = r"""
const tracks = DATA.tracks;
$("title").textContent = DATA.title; $("sub").textContent = DATA.subtitle;
tracks.forEach((t, k) => $("rows").appendChild(songRow(t, { number: String(k + 1), showLanguage: false, onPlay: () => playSongsNow([songOf(t)]) })));
const inList = () => tracks.some(t => t.lastfm === nowLink);
function refreshBigPlay() {
  const playing = inList() && !nowPaused;
  $("bigplay").innerHTML = playing ? PAUSE : PLAY; $("bigplay").title = $("bigplay").ariaLabel = playing ? "Pause" : "Play all";
}
$("bigplay").addEventListener("click", () => (inList() ? playToggle() : playSongsNow(tracks.map(songOf))));
function filter(q) {
  let shown = 0;
  document.querySelectorAll(".row").forEach((r, k) => { const hit = matches(q, tracks[k]); r.hidden = !hit; if (hit) shown++; });
  $("matches").hidden = !q; $("matches").textContent = q ? plural(shown, "song", "songs") + " of " + tracks.length + " match “" + q + "”" : "";
  $("nomatch").hidden = !q || shown > 0;
  closeSongMenu(false);
}
onSearch(filter);
refreshRows(); refreshHearts(); if (searchQuery()) filter(searchQuery()); fit();
hub.postMessage({ type: "hello" });
window.songlist = { id: DATA.id, count: () => tracks.length };
"""


# ---------------------------------------------------------------- Library
def render_library():
    """The Library: Liked songs, My playlist and your other playlists, and Recently
    played, as cards; each opens as a list (Play all, Rename / Delete or Clear,
    and a row per song). Everything is read from this browser."""
    return _page(CARD_CSS + ROW_CSS + MENU_CSS + LIBRARY_CSS, LIBRARY_BODY, MENU_JS + ROW_JS + LIBRARY_JS)


LIBRARY_CSS = """
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:16px; }
  .coll { display:flex; flex-direction:column; gap:10px; border:1px solid var(--line); background:var(--field); border-radius:18px; padding:12px; cursor:pointer;
          text-align:left; font:inherit; color:var(--ink); transition:border-color .15s, box-shadow .15s; }
  .coll:hover { border-color:#cfc8ef; box-shadow:0 6px 18px rgba(45,42,82,.1); }
  .coll[hidden] { display:none; }
  .cover { aspect-ratio:1; width:100%; border-radius:12px; overflow:hidden; display:grid; grid-template-columns:1fr 1fr; background:var(--lav-soft); }
  .cover img { width:100%; height:100%; object-fit:cover; display:block; }
  .cover.one img { grid-column:1 / 3; grid-row:1 / 3; }
  .cover .ph { grid-column:1 / 3; display:flex; align-items:center; justify-content:center; color:var(--purple); }
  .cover .ph svg { width:42%; height:42%; }
  .cover.liked { background:linear-gradient(135deg, #f8e9e1, #f3cdbb); } .cover.liked .ph { color:var(--terra); }
  .cover.liked .ph svg { fill:currentColor; }
  .cover.recent { background:linear-gradient(135deg, #eeebfa, #d9d3f3); }
  html.dark .cover.liked { background:linear-gradient(135deg, #3a2721, #5c3324); } html.dark .cover.recent { background:linear-gradient(135deg, #2e2a52, #3d3770); }
  .cover.new { background:var(--card2); border:1.5px dashed #cfc8ef; }
  .coll b { font-size:15px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .coll small { font-size:12.5px; color:var(--muted); margin-top:-6px; }
  #back { display:inline-flex; align-items:center; gap:6px; border:none; background:none; font:inherit; font-size:14px; color:var(--muted);
          cursor:pointer; padding:6px 10px 6px 4px; border-radius:999px; margin:-6px 0 12px -4px; }
  #back:hover { color:var(--purple); background:var(--hover-soft); } #back svg { width:17px; height:17px; }
  .vhead { display:flex; gap:20px; align-items:flex-end; }
  .vhead .cover { width:132px; flex:none; }
  .actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:16px 0 0; }
  .btn { border:1px solid var(--line); background:var(--field); color:var(--ink); border-radius:999px; font:inherit; font-size:13.5px; font-weight:600;
         padding:8px 16px; cursor:pointer; display:inline-flex; align-items:center; gap:7px; }
  .btn:hover { border-color:var(--lavender); } .btn[hidden] { display:none; }
  .btn.primary { background:var(--purple); color:var(--on-purple); border-color:var(--purple); } .btn.primary:hover { background:var(--purple-hover); }
  .btn.danger { color:var(--danger); } .btn.danger.primary { background:#9b2f2f; border-color:#9b2f2f; color:#fff; }
  .btn svg { width:14px; height:14px; } .btn.primary svg { fill:currentColor; }
  .confirm { background:var(--field); border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin:12px 0 0; font-size:14px; }
  .confirm[hidden] { display:none; } .confirm p { margin:0 0 10px; }
  #view[hidden], #home[hidden] { display:none; }
  @media (max-width: 640px) { .grid { grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px; } .vhead .cover { width:96px; } .vhead { gap:14px; } }
"""
LIBRARY_BODY = """
<div id="card">
  <div id="home">
    <p class="kicker">Your library</p>
    <div class="grid" id="grid"></div>
    <div id="nomatchlists" class="empty" hidden></div>
  </div>
  <div id="view" hidden>
    <button id="back"><span id="backicon"></span>Library</button>
    <div class="vhead"><div class="cover" id="vcover"></div>
      <div><p class="kicker" id="vkind"></p><h2 id="vtitle"></h2><p class="sub" id="vcount"></p></div></div>
    <div class="actions">
      <button class="btn primary" id="vplay"></button>
      <button class="btn" id="vrename">Rename</button>
      <button class="btn danger" id="vclear">Clear</button>
      <button class="btn danger" id="vdelete">Delete playlist</button>
    </div>
    <div class="confirm" id="confirm" hidden><p id="confirmtext"></p>
      <button class="btn danger primary" id="confirmyes"></button> <button class="btn" id="confirmno">Cancel</button></div>
    <p id="matches" aria-live="polite" hidden></p>
    <div id="rows"></div>
    <p class="empty" id="empty" hidden></p>
    <div id="nomatch" hidden>No songs here match your search.</div>
  </div>
</div>
<div id="menu" role="menu" hidden></div>
"""
LIBRARY_JS = r"""
$("backicon").outerHTML = icon("back"); $("vplay").innerHTML = icon("play") + "Play all";
const EMPTY = {
  liked: "No liked songs yet. Tap ♡ next to a song, or press L while it plays.",
  playlist: "This playlist is empty. Use “…” next to a song and pick “Add to playlist…”.",
  recent: "Nothing played yet. The songs you play show up here (the last 50), without times.",
};
let openId = "", query = searchQuery(), confirmRun = null;
function collections() {
  return [{ id: "liked", kind: "liked", name: "Liked songs", songs: savedSongs() }]
    .concat(playlists().map(p => ({ id: p.id, kind: "playlist", name: p.name, songs: p.songs })))
    .concat([{ id: "recent", kind: "recent", name: "Recently played", songs: recentSongs() }]);
}
const find = id => collections().find(c => c.id === id);
function cover(c, box) {
  box.replaceChildren(); box.className = "cover" + (c.kind === "liked" ? " liked" : c.kind === "recent" ? " recent" : "");
  const arts = c.kind === "liked" ? [] : c.songs.map(s => s.art).filter(Boolean).slice(0, 4);
  if (arts.length) {
    if (arts.length < 4) box.classList.add("one");
    (arts.length >= 4 ? arts : arts.slice(0, 1)).forEach(a => { const img = el("img"); img.src = a; img.alt = ""; img.loading = "lazy"; box.appendChild(img); });
  } else {
    const ph = el("div", "ph"); ph.innerHTML = icon(c.kind === "liked" ? "heart" : c.kind === "recent" ? "history" : "list"); box.appendChild(ph);
  }
}
function count(c) { return c.kind === "recent" ? (c.songs.length ? "The last " + plural(c.songs.length, "song", "songs") + " you played" : "Nothing played yet") : plural(c.songs.length, "song", "songs"); }
function renderGrid() {
  const grid = $("grid"); grid.replaceChildren();
  let shown = 0;
  collections().forEach(c => {
    const b = el("button", "coll"); b.dataset.id = c.id; b.ariaLabel = c.name + ", " + count(c);
    const box = el("div"); cover(c, box); b.appendChild(box);
    b.appendChild(el("b", null, c.name)); b.appendChild(el("small", null, count(c)));
    b.hidden = !!query && !norm(c.name).includes(norm(query)); if (!b.hidden) shown++;
    b.addEventListener("click", () => openCollection(c.id));
    grid.appendChild(b);
  });
  const add = el("button", "coll"); add.id = "newcard"; add.ariaLabel = "New playlist";
  const box = el("div", "cover new"), ph = el("div", "ph"); ph.innerHTML = icon("plus"); box.appendChild(ph); add.appendChild(box);
  add.appendChild(el("b", null, "New playlist")); add.appendChild(el("small", null, "Kept in this browser"));
  add.addEventListener("click", createPlaylist); grid.appendChild(add);
  $("nomatchlists").hidden = !query || shown > 0;
  $("nomatchlists").textContent = "No playlists are called “" + query + "”. Open one to search its songs.";
}
function createPlaylist() {
  askName({ title: "New playlist", label: "Playlist name", ok: "Create", hint: "Kept in this browser only.",
            onOk: name => { const id = newPlaylist(name); toast("Created “" + name + "”"); openCollection(id); } });
}
function openCollection(id) {
  if (!find(id)) id = "";
  openId = id; $("confirm").hidden = true;
  try { window.parent.__moodtuneOpen = ""; } catch (e) {}
  hub.postMessage({ type: "libraryView", id });
  render();
  if (id) $("back").focus();
}
function removeAction(c, t) {
  if (c.kind === "playlist") return { label: "Remove from " + c.name, run: () => { removeFromPlaylist(c.id, t.lastfm); toast("Removed “" + t.title + "” from " + c.name); } };
  if (c.kind === "recent") return { label: "Remove from Recently played", run: () => removeRecent(t.lastfm) };
  return null;   // Liked songs: the heart (and the menu's "Remove from Liked songs") already do it
}
function renderView() {
  const c = find(openId);
  cover(c, $("vcover"));
  $("vkind").textContent = c.kind === "playlist" ? "Playlist" : c.kind === "liked" ? "Your hearts" : "History";
  $("vtitle").textContent = c.name; $("vcount").textContent = count(c);
  const n = c.songs.length, own = c.kind === "playlist" && c.id !== MY_PLAYLIST;
  $("vplay").hidden = !n; $("vrename").hidden = !own; $("vdelete").hidden = !own; $("vclear").hidden = own || !n;
  const rows = $("rows"); rows.replaceChildren();
  c.songs.forEach((s, k) => { const t = songTrack(s); rows.appendChild(songRow(t, { number: String(k + 1), showLanguage: true, onPlay: () => playSongsNow([s]), remove: removeAction(c, t) })); });
  $("empty").hidden = n > 0; $("empty").textContent = EMPTY[c.kind];
  filter(query); refreshRows(); refreshHearts();
}
function render() {
  if (openId && !find(openId)) openId = "";
  $("home").hidden = !!openId; $("view").hidden = !openId;
  if (openId) renderView(); else renderGrid();
}
function refreshBigPlay() {}
function filter(q) {
  query = q;
  if (!openId) return renderGrid();
  const songs = find(openId).songs; let shown = 0;
  document.querySelectorAll(".row").forEach((r, k) => { const hit = matches(q, songTrack(songs[k])); r.hidden = !hit; if (hit) shown++; });
  $("matches").hidden = !q || !songs.length; $("matches").textContent = q ? plural(shown, "song", "songs") + " of " + songs.length + " match “" + q + "”" : "";
  $("nomatch").hidden = !q || !songs.length || shown > 0;
  closeSongMenu(false);
}
onSearch(filter);
function ask(text, yes, run) { $("confirmtext").textContent = text; $("confirmyes").textContent = yes; confirmRun = run; $("confirm").hidden = false; $("confirmno").focus(); }
$("confirmno").addEventListener("click", () => { $("confirm").hidden = true; });
$("confirmyes").addEventListener("click", () => { $("confirm").hidden = true; if (confirmRun) confirmRun(); });
$("vplay").addEventListener("click", () => { const c = find(openId); if (c && c.songs.length) playSongsNow(c.songs); });
$("vrename").addEventListener("click", () => { const c = find(openId); if (c) askName({ title: "Rename playlist", label: "Playlist name", value: c.name, ok: "Rename", onOk: name => renamePlaylist(c.id, name) }); });
$("vdelete").addEventListener("click", () => { const c = find(openId); if (c) ask("Delete “" + c.name + "”" + (c.songs.length ? " and its " + plural(c.songs.length, "song", "songs") : "") + "? This can't be undone.", "Delete",
  () => { deletePlaylist(c.id); toast("Deleted “" + c.name + "”"); openCollection(""); }); });
$("vclear").addEventListener("click", () => { const c = find(openId); if (!c) return;
  ask(c.kind === "recent" ? "Clear Recently played?" : "Remove all " + plural(c.songs.length, "song", "songs") + " from " + c.name + "?", "Clear",
      () => { if (c.kind === "liked") setSaved([]); else if (c.kind === "recent") clearRecent(); else clearPlaylist(c.id); toast(c.name + " is empty"); }); });
$("back").addEventListener("click", () => { const from = openId; openCollection(""); const b = document.querySelector(`.coll[data-id="${from}"]`); if (b) b.focus(); });
["saved", "playlists", "recent"].forEach(t => on(t, render));
hub.addEventListener("message", e => { if (e.data && e.data.type === "openCollection") openCollection(e.data.id); });
window.addEventListener("pagehide", () => hub.postMessage({ type: "libraryView", id: "" }));
let start = ""; try { start = window.parent.__moodtuneOpen || ""; } catch (e) {}
openCollection(start); fit();
hub.postMessage({ type: "hello" });
window.library = { open: openCollection, current: () => openId, collections };
"""


# ---------------------------------------------------------------- Quick picks (Home)
def render_quickpicks(list_id, title, subtitle, tracks):
    """Home's Quick picks: a grid of songs (art, title, artist; 3 columns on a wide
    window) with Play all. A tile plays its song through the player bar, after the
    song playing, like Play next."""
    data = {"id": list_id, "title": title, "subtitle": subtitle,
            "tracks": [{k: t[k] for k in ("title", "artist", "lastfm", "link_label", "language", "bucket", "preview", "art")} for t in tracks]}
    return _page(QUICKPICKS_CSS, QUICKPICKS_BODY, QUICKPICKS_JS, data)


QUICKPICKS_CSS = """
  html, body { overflow:hidden; }
  #card { background:var(--card); border:1px solid var(--line); border-radius:22px; padding:22px 22px 16px;
          box-shadow:0 1px 2px rgba(45,42,82,.05), 0 8px 26px rgba(45,42,82,.07); margin:2px 2px 14px; }
  .head { display:flex; align-items:center; gap:16px; justify-content:space-between; margin:0 4px 14px; }
  .kicker { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 4px; }
  h2 { font-family:var(--serif); font-weight:600; font-size:26px; line-height:1.15; margin:0; }
  .sub { margin:4px 0 0; font-size:13.5px; color:var(--muted); }
  #playall { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--line); background:var(--field); color:var(--ink); border-radius:999px;
             font:inherit; font-size:14px; font-weight:600; padding:9px 18px; cursor:pointer; flex:none; }
  #playall:hover { border-color:var(--lavender); } #playall svg { width:15px; height:15px; fill:currentColor; stroke:none; }
  #grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:4px 14px; }
  .tile { display:grid; grid-template-columns:52px minmax(0, 1fr); align-items:center; gap:12px; padding:7px 8px; border-radius:12px;
          cursor:pointer; border:none; background:none; font:inherit; color:var(--ink); text-align:left; width:100%; }
  .tile:hover { background:var(--hover); } .tile.playing { background:var(--lav-soft); }
  .tile[hidden] { display:none; }
  .cover { position:relative; width:52px; height:52px; }
  .art { width:52px; height:52px; border-radius:9px; background:var(--art-bg); object-fit:cover; display:block; }
  .noart { display:flex; align-items:center; justify-content:center; color:#8f8aa3; font-size:20px; }
  .cover .over { position:absolute; inset:0; border-radius:9px; background:rgba(28,25,52,.45); display:flex; align-items:center; justify-content:center;
                 opacity:0; transition:opacity .12s; }
  .cover .over svg { width:18px; height:18px; fill:#fff; }
  .tile:hover .over, .tile:focus-visible .over, .tile.playing .over { opacity:1; }
  .meta { min-width:0; } .title { font-size:14.5px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tile.playing .title { color:var(--purple); font-weight:700; }
  .artist { font-size:12.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }
  #nomatch { color:var(--muted); font-size:14px; padding:10px 6px 4px; } #nomatch[hidden] { display:none; }
  @media (max-width: 860px) { #grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 520px) { #grid { grid-template-columns:minmax(0, 1fr); } #card { padding:18px 12px 12px; border-radius:18px; }
                              h2 { font-size:22px; } .head { flex-wrap:wrap; } }
"""
QUICKPICKS_BODY = """
<div id="card">
  <div class="head"><div><p class="kicker">Quick picks</p><h2 id="title"></h2><p class="sub" id="sub"></p></div>
    <button id="playall"></button></div>
  <div id="grid" role="list"></div>
  <div id="nomatch" hidden>No quick picks match your search.</div>
</div>
"""
QUICKPICKS_JS = r"""
const tracks = DATA.tracks;
let nowLink = "", nowPaused = true;
$("title").textContent = DATA.title; $("sub").textContent = DATA.subtitle;
$("playall").innerHTML = PLAY + "Play all";
$("playall").addEventListener("click", () => hub.postMessage({ type: "playSongs", songs: tracks.map(songOf) }));
tracks.forEach(t => {
  const b = el("button", "tile"); b.dataset.link = t.lastfm; b.setAttribute("role", "listitem");
  b.ariaLabel = "Play " + t.title + " by " + t.artist;
  const cover = el("div", "cover"); cover.appendChild(artNode(t, "art"));
  const over = el("div", "over"); over.setAttribute("aria-hidden", "true"); over.innerHTML = PLAY; cover.appendChild(over); b.appendChild(cover);
  const meta = el("div", "meta"); meta.appendChild(el("div", "title", t.title)); meta.appendChild(el("div", "artist", t.artist)); b.appendChild(meta);
  b.addEventListener("click", () => (t.lastfm === nowLink ? hub.postMessage({ type: "key", key: "space" }) : hub.postMessage({ type: "playSongs", songs: [songOf(t)] })));
  $("grid").appendChild(b);
});
function refresh() {
  document.querySelectorAll(".tile").forEach(b => {
    const on = !!nowLink && b.dataset.link === nowLink;
    b.classList.toggle("playing", on);
    b.querySelector(".over").innerHTML = on && !nowPaused ? PAUSE : PLAY;
  });
}
hub.addEventListener("message", e => { if (e.data && e.data.type === "now") { nowLink = e.data.link; nowPaused = e.data.paused; refresh(); } });
function filter(q) {
  let shown = 0;
  document.querySelectorAll(".tile").forEach((b, k) => { const hit = matches(q, tracks[k]); b.hidden = !hit; if (hit) shown++; });
  $("nomatch").hidden = !q || shown > 0;
}
onSearch(filter);
function fit() { if (window.frameElement) window.frameElement.style.height = Math.ceil(document.documentElement.scrollHeight) + "px"; }
new ResizeObserver(fit).observe(document.body);
document.addEventListener("keydown", e => { const key = shortcut(e); if (!key) return; e.preventDefault(); if (key !== "held") hub.postMessage({ type: "key", key }); });
if (searchQuery()) filter(searchQuery());
refresh(); fit(); hub.postMessage({ type: "hello" });
window.quickpicks = { id: DATA.id, tracks: () => tracks.map(t => ({ title: t.title, link: t.lastfm, language: t.language, bucket: t.bucket })) };
"""


# ---------------------------------------------------------------- one-off "play these songs" (Home's mood chips)
def render_play_command(nonce, songs, message, label):
    """A hidden page that tells the player bar to replace its queue with `songs`
    and play the first, once: the nonce is remembered on the page, so drawing it
    again on a later rerun does nothing. songs: player._track dicts; label: what
    the queue panel calls the list."""
    data = {"nonce": nonce, "message": message, "label": label,
            "songs": [{"title": t["title"], "artist": t["artist"], "link": t["lastfm"], "preview": t["preview"],
                       "art": t["art"] or "", "language": t["language"]} for t in songs]}
    return _page("", "", PLAYCMD_JS, data)


PLAYCMD_JS = r"""
let done = {};
try { const page = window.parent; done = page.__moodtunePlayed || (page.__moodtunePlayed = {}); } catch (e) {}
if (!done[DATA.nonce]) {
  done[DATA.nonce] = 1;
  hub.postMessage({ type: "replaceQueue", songs: DATA.songs, label: DATA.label });
  if (DATA.message) toast(DATA.message);
}
window.playcommand = { nonce: DATA.nonce, count: DATA.songs.length };
"""
