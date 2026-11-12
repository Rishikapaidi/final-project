"""The page's look: CSS for the Streamlit page and the small HTML pieces it shows.

Everything interactive is a Streamlit widget (styled here with CSS) or one of
the two components in player.py. Nothing here touches Streamlit's state or
the models; the functions only turn values into HTML. Text that comes from
outside (song titles, the user's words) never goes through here.
"""
import html
from datetime import datetime, timedelta, timezone

import mood

APP_NAME = "MOOD TUNE"

# The app's own mark: four sound-wave bars, lavender on deep purple.
LOGO_SVG = ('<svg class="logo" viewBox="0 0 40 40" width="38" height="38" role="img" aria-label="MOOD TUNE logo">'
            '<rect width="40" height="40" rx="12" fill="#2d2a52"/>'
            '<rect x="8" y="16" width="4.5" height="8" rx="2.25" fill="#b9b0e8"/>'
            '<rect x="15" y="9" width="4.5" height="22" rx="2.25" fill="#b9b0e8"/>'
            '<rect x="22" y="13" width="4.5" height="14" rx="2.25" fill="#e8a283"/>'
            '<rect x="29" y="18" width="4.5" height="4" rx="2.25" fill="#b9b0e8"/></svg>')

# Colours of the three groups of emotion bars.
BAR_COLOURS = {"face": "#8d88a6", "text": "#7b6fd0", "combined": "#2d2a52"}
BAR_NAMES = {"face": "Face", "text": "Text", "combined": "Combined"}

PAGE_CSS = """
<style>
  :root { --hero:#2d2a52; --field:#fff; --hover:#f4f1fb; --hover-soft:rgba(45,42,82,.06); --on-purple:#fff; --card2:#fcfbf8;
          --rail:#eeeaf6; --uplift-on:#fffaf7;
          --cream:#f6f3ee; --card:#ffffff; --ink:#2b2840; --muted:#625e78; --line:#e7e1d8; --purple:#2d2a52;
          --purple-2:#3a3668; --lavender:#b9b0e8; --lav-soft:#eeebfa; --terra:#b85c38; --terra-soft:#f8e9e1; --terra-ink:#7a3a20;
          --serif:"Fraunces", Georgia, "Times New Roman", serif; }
  #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
  .stApp { background:var(--cream); }
  h1, h2, h3 { font-family:var(--serif) !important; }
  /* no anchor-link icons next to the page's headings */
  [data-testid="stHeaderActionElements"] { display:none !important; }

  /* the shell: a slim top bar, a sidebar on the left, and the page between them and the player bar */
  :root { --top:64px; --nav:244px; --nav-mini:84px; }
  [data-testid="stMain"] { padding-left:var(--nav); transition:padding-left .2s ease; }
  html.mt-nav-mini [data-testid="stMain"] { padding-left:var(--nav-mini); }
  [data-testid="stMainBlockContainer"] { max-width:1180px; padding:calc(var(--top) + 14px) 28px 132px; }
  /* the page's style block and the three pinned parts take no room, so cancel the gap after each */
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child,
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-nav),
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-playerbar) { margin-bottom:-1rem; }

  /* 1. top bar: menu button, logo and search (a component), "Songs in", the avatar (a component) */
  .st-key-topbar { position:fixed; top:0; left:0; right:0; height:var(--top); z-index:1000; background:var(--cream);
                   border-bottom:1px solid var(--line); padding:0 14px 0 10px; flex-wrap:nowrap; }
  [data-testid="stLayoutWrapper"]:has(> .st-key-topleft) { flex:1 1 auto; min-width:0; }
  .st-key-topleft iframe, .st-key-topactions iframe { display:block; width:100%; height:56px; border:none; }
  [data-testid="stLayoutWrapper"]:has(> .st-key-topactions) { flex:none; width:56px; }
  .st-key-language { flex:none; width:auto; }
  .st-key-language [data-testid="stSelectbox"] { display:flex; flex-direction:row; align-items:center; justify-content:flex-end; gap:10px; }
  .st-key-language [data-testid="stWidgetLabel"] { margin:0; min-height:0; flex:none; }
  .st-key-language [data-testid="stWidgetLabel"] p { font-size:14px; color:var(--muted); }
  .st-key-language [data-testid="stSelectbox"] > div:last-child { width:150px; flex:none; }

  /* 2. sidebar: Home / Explore / Library (Streamlit buttons), then New playlist and your playlists (a component) */
  .st-key-nav { position:fixed; top:var(--top); left:0; bottom:86px; width:var(--nav); z-index:998; background:var(--cream);
                padding:12px 12px 0; gap:2px; flex-wrap:nowrap; overflow:hidden; transition:width .2s ease, transform .2s ease; }
  .st-key-nav .stButton button { justify-content:flex-start; gap:0; border:none; border-radius:12px; padding:9px 16px; min-height:44px;
                                 background:transparent; color:var(--ink); box-shadow:none; }
  .st-key-nav .stButton button:hover { background:var(--hover-soft); color:var(--purple); }
  .st-key-nav .stButton button[kind="primary"], .st-key-nav .stButton button[data-testid="stBaseButton-primary"] {
       background:var(--lav-soft); color:var(--purple); }
  .st-key-nav .stButton button > div { width:100%; justify-content:flex-start; }
  .st-key-nav .stButton button p { font-size:15px; font-weight:500; }
  .st-key-nav .stButton button[kind="primary"] p { font-weight:700; }
  .st-key-nav .stButton button [data-testid="stIconMaterial"] { font-size:23px; margin-right:16px; }
  .st-key-nav .stButton button:focus-visible { outline:3px solid var(--lavender); outline-offset:1px; }
  .st-key-nav [data-testid="stLayoutWrapper"]:has(> .st-key-navlists) { flex:1 1 auto; min-height:0; display:flex; }
  .st-key-navlists { flex:1 1 auto; min-height:0; border-top:1px solid var(--line); margin-top:10px; padding-top:12px; }
  .st-key-navlists [data-testid="stElementContainer"] { flex:1 1 auto !important; height:auto !important; min-height:0; }
  .st-key-navlists iframe { display:block; width:100%; height:100% !important; border:none; }
  /* folded: icons with a small label under them */
  html.mt-nav-mini .st-key-nav { width:var(--nav-mini); padding:12px 6px 0; }
  html.mt-nav-mini .st-key-nav .stButton button { flex-direction:column; padding:10px 2px 8px; min-height:66px; justify-content:center; }
  html.mt-nav-mini .st-key-nav .stButton button > div { justify-content:center; }
  html.mt-nav-mini .st-key-nav .stButton button > div > span { flex-direction:column; gap:5px; }
  html.mt-nav-mini .st-key-nav .stButton button [data-testid="stIconMaterial"] { margin:0; }
  html.mt-nav-mini .st-key-nav .stButton button p { font-size:11px; }
  html.mt-nav-mini .st-key-nav [data-testid="stLayoutWrapper"]:has(> .st-key-navlists) { display:none; }
  #moodtune-navshade { display:none; }
  .greeting { margin:26px 0 4px; }
  .greeting .date { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  .greeting h1 { font-size:34px; font-weight:600; color:var(--ink); margin:4px 0 0; padding:0; letter-spacing:-.01em; }

  /* 2. hero card */
  .st-key-hero { background:var(--hero); border-radius:26px; padding:34px 34px 30px; gap:22px;
                 box-shadow:0 2px 4px rgba(45,42,82,.08), 0 18px 40px rgba(45,42,82,.18); }
  .st-key-hero h2 { color:#fbf8f3; font-size:38px; font-weight:600; margin:0; padding:0; line-height:1.1; letter-spacing:-.01em; }
  .hero-sub { color:#ddd8f5; font-size:16.5px !important; margin:12px 0 4px !important; }
  .st-key-hero [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
  .st-key-hero [data-testid="stLayoutWrapper"]:has(> .st-key-facetile),
  .st-key-hero [data-testid="stLayoutWrapper"]:has(> .st-key-texttile) { flex:1 1 auto; }
  /* Streamlit's "Press Enter to submit form · 12/180" goes under the box instead of over the text */
  .st-key-user_text [data-testid="stTextInput"] > div:has(> [data-testid="InputInstructions"]) { position:static; height:auto; }
  .st-key-user_text [data-testid="InputInstructions"] { position:static; color:#d6d0f2 !important; font-size:12px; padding:6px 2px 0; text-align:right; }
  .st-key-facetile, .st-key-texttile { background:var(--purple-2); border:1px solid rgba(255,255,255,.1); border-radius:20px;
                                      padding:20px; flex:1; gap:14px; }
  .tile-foot { color:#cfc8ef; font-size:12.5px !important; line-height:1.45; margin:0 !important; }
  .tile-label { font-size:12px !important; letter-spacing:.12em; text-transform:uppercase; color:#cfc8ef; font-weight:600; margin:0 !important; }
  .scanimg { width:100%; max-height:280px; object-fit:contain; border-radius:14px; display:block; background:#1f1c3d; }
  .scan-empty { height:200px; border-radius:14px; background:#25224a; display:flex; align-items:center; justify-content:center;
                color:#cfc8ef; font-size:14px; text-align:center; padding:0 20px; }
  /* the bottom margin keeps the note clear of "Tap to scan again" (markdown blocks pull up by 1rem) */
  .tile-note { color:#e3def7; font-size:13px !important; line-height:1.45; margin:10px 0 22px !important; }
  .tile-note.with-tip { margin-bottom:0 !important; }
  .tile-note.tile-tip { color:#cfc8ef; font-size:12.5px !important; margin-top:4px !important; }
  .listening { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; height:220px;
               border-radius:14px; background:#25224a; color:#f1edfb; font-size:17px; }
  .pulse { width:58px; height:58px; border-radius:50%; background:var(--lavender); animation:pulse 1.3s ease-in-out infinite; }
  .listening-detail { font-size:13.5px; color:#cfc8ef; margin-top:-6px; }
  @keyframes pulse { 0%, 100% { transform:scale(.75); opacity:.55; } 50% { transform:scale(1.08); opacity:1; } }
  .st-key-hero [data-testid="stWidgetLabel"] p { color:#f1edfb; font-size:15px; font-weight:500; }
  .st-key-hero [data-testid="stTextInput"] [data-baseweb="input"] { background:#fff; border-radius:12px; border:2px solid transparent; }
  .st-key-hero [data-testid="stTextInput"] [data-baseweb="input"]:focus-within { border-color:var(--lavender); }
  .st-key-hero [data-testid="stTextInput"] input { color:#2b2840; background:#fff; font-size:15px; padding:12px 14px; }
  .st-key-hero [data-testid="InputInstructions"] { color:var(--muted); }
  .st-key-hero [data-testid="stForm"] { border:none; padding:0; }
  .st-key-hero button { border-radius:999px !important; font-weight:600; padding:8px 18px; min-height:42px; }
  .st-key-hero button[kind="primaryFormSubmit"], .st-key-hero button[data-testid="stBaseButton-primaryFormSubmit"] {
       background:var(--lavender); color:#2d2a52; border:none; }
  .st-key-hero button[kind="primaryFormSubmit"]:hover { background:#cdc6f1; color:#2d2a52; }
  .st-key-rescan button { background:#fff; color:#2d2a52; border:none; }
  /* a button's label follows the button's colour (Streamlit colours the label's text block separately) */
  button [data-testid="stMarkdownContainer"], button [data-testid="stMarkdownContainer"] p { color:inherit !important; }
  .st-key-rescan button:hover { background:#f1edfb; color:#2d2a52; }
  .st-key-facetile [data-testid="stPopoverButton"] { background:transparent; color:#f1edfb; border:1px solid rgba(255,255,255,.4); }
  .st-key-facetile [data-testid="stPopoverButton"]:hover { border-color:#fff; color:#fff; background:rgba(255,255,255,.06); }
  .st-key-hero button:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  .reading { display:inline-flex; align-items:center; gap:10px; color:#f1edfb; font-size:14px;
             background:rgba(255,255,255,.08); border-radius:999px; padding:7px 16px; }
  .reading .dot { width:10px; height:10px; border-radius:50%; background:var(--lavender); animation:pulse 1.1s ease-in-out infinite; }

  /* white cards: mood reading and the choice */
  .st-key-moodcard, .st-key-choice { background:var(--card); border:1px solid var(--line); border-radius:24px; padding:28px;
       box-shadow:0 1px 2px rgba(45,42,82,.05), 0 8px 26px rgba(45,42,82,.07); margin-top:10px; }
  .card-kicker { font-size:12px !important; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 6px !important; }
  .mood-head { display:flex; align-items:center; gap:16px; min-width:0; }
  .mood-title { min-width:0; overflow-wrap:anywhere; }
  .mood-head .emoji { font-size:50px; line-height:1; }
  .mood-title { font-family:var(--serif); font-size:30px; font-weight:600; color:var(--ink); line-height:1.15; }
  .mood-title b { font-weight:700; }
  .mood-grid { display:grid; grid-template-columns:230px minmax(0,1fr); gap:34px; align-items:center; margin:6px 0 18px; }
  .gauge-box { position:relative; text-align:center; }
  .gauge { width:100%; display:block; }
  .gauge-num { position:absolute; left:0; right:0; top:34%; font-size:15px; color:var(--muted); line-height:1.2; }
  .mood-main { min-width:0; }
  .gauge-num span { display:block; font-family:var(--serif); font-size:40px; font-weight:600; color:var(--ink); line-height:1; }
  .band { display:inline-block; margin-top:16px; font-size:13px; font-weight:600; border-radius:999px; padding:4px 14px; }
  .band-light { background:#e7f3ea; color:#2c5e3a; } .band-in-between { background:var(--lav-soft); color:var(--purple); }
  .band-low { background:var(--terra-soft); color:var(--terra-ink); }
  .mood-sentence { font-size:15.5px !important; color:var(--ink); margin:10px 0 16px !important; }
  .leans { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px 26px; max-width:560px; }
  .lean-top { display:flex; justify-content:space-between; font-size:13.5px; color:var(--muted); margin-bottom:6px; }
  .lean-top b { color:var(--ink); font-variant-numeric:tabular-nums; }
  .lean .rail { background:var(--rail); border-radius:4px; height:8px; overflow:hidden; }
  .lean .fill { height:100%; border-radius:4px; background:#7b6fd0; }
  .lean[data-source="face"] .fill { background:#8d88a6; }
  .blend { font-size:13px !important; color:var(--muted); margin:12px 0 0 !important; }
  .st-key-moodcard [data-testid="stExpander"] details { border:1px solid var(--line); border-radius:14px; background:var(--card2); }
  .st-key-moodcard [data-testid="stExpander"] summary p { font-weight:600; color:var(--purple); }
  .legend { display:flex; gap:16px; font-size:12.5px; color:var(--muted); margin:2px 0 14px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
  .erow { display:grid; grid-template-columns:76px 1fr; gap:12px; align-items:center; margin-bottom:12px; }
  .ename { font-size:14px; text-transform:capitalize; color:var(--ink); }
  .ebars { display:flex; flex-direction:column; gap:4px; }
  .bar { display:grid; grid-template-columns:1fr 42px; gap:10px; align-items:center; }
  .bar .rail { background:var(--rail); border-radius:4px; height:7px; overflow:hidden; }
  .bar .fill { height:100%; border-radius:4px; }
  .bar .pct { font-size:12px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  .bar.combined .rail { height:10px; } .bar.combined .pct { color:var(--ink); font-weight:600; }
  .barcap { color:var(--muted); font-size:13px !important; margin:6px 0 0 !important; }

  /* 4. the choice cards: the visible card is HTML; a transparent Streamlit button covers it */
  .st-key-choice h2 { font-size:26px; font-weight:600; color:var(--ink); margin:0; padding:0; }
  .choice-sub { color:var(--muted); font-size:14.5px !important; margin:8px 0 6px !important; }
  .st-key-choice_uplift, .st-key-choice_resonance { position:relative; }
  .choice-card { border:2px solid var(--line); border-radius:18px; padding:18px 20px; min-height:128px; background:var(--card);
                 transition:border-color .15s, box-shadow .15s, background .15s; }
  .choice-card .icon { font-size:24px; line-height:1; }
  .choice-card h3 { font-size:21px; font-weight:600; margin:8px 0 0; padding:0; color:var(--ink); }
  .choice-card p { font-size:14px; color:var(--muted); margin:4px 0 0; line-height:1.45; }
  .choice-card .tag { display:inline-block; font-size:11.5px; font-weight:600; letter-spacing:.04em; border-radius:999px;
                      padding:3px 10px; background:var(--lav-soft); color:var(--purple); float:right; }
  .choice-card .fallback { margin-top:10px; font-size:13px; color:var(--terra-ink); background:var(--terra-soft); border-radius:10px; padding:8px 11px; }
  .choice-card.uplift.on { border-color:var(--terra); background:var(--uplift-on); box-shadow:0 6px 18px rgba(184,92,56,.14); }
  .choice-card.resonance.on { border-color:var(--purple); background:var(--hover); box-shadow:0 6px 18px rgba(45,42,82,.12); }
  .choice-card .check { display:none; font-size:12.5px; font-weight:600; margin-top:10px; }
  .choice-card.on .check { display:block; } .choice-card.uplift .check { color:var(--terra-ink); } .choice-card.resonance .check { color:var(--purple); }
  .st-key-choice_uplift:hover .choice-card:not(.on), .st-key-choice_resonance:hover .choice-card:not(.on) { border-color:#cfc8ef; }
  .st-key-choice_uplift [data-testid="stElementContainer"]:has(.stButton),
  .st-key-choice_resonance [data-testid="stElementContainer"]:has(.stButton) { position:absolute; inset:0; z-index:2; margin:0; }
  .st-key-choice_uplift .stButton, .st-key-choice_resonance .stButton,
  .st-key-choice_uplift .stButton button, .st-key-choice_resonance .stButton button { width:100%; height:100%; }
  .st-key-choice_uplift .stButton button, .st-key-choice_resonance .stButton button { opacity:0; cursor:pointer; }
  .st-key-choice_uplift [data-testid="stMarkdown"], .st-key-choice_resonance [data-testid="stMarkdown"],
  .st-key-choice_uplift [data-testid="stMarkdownContainer"], .st-key-choice_resonance [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
  .st-key-choice_uplift:has(button:focus-visible) .choice-card, .st-key-choice_resonance:has(button:focus-visible) .choice-card {
       outline:3px solid var(--lavender); outline-offset:3px; }
  .st-key-choose_auto button { background:none; border:none; color:var(--purple); text-decoration:underline; padding:0; min-height:0; }

  /* 5-6. the path card iframe grows to fit; the player bar iframe is pinned to the bottom */
  .st-key-path iframe { display:block; width:100%; border:none; }
  /* Streamlit fixes the wrapper at the component's first height; let it follow the card's own height
     (otherwise the end of the list hangs past the page's bottom padding, behind the player bar) */
  .st-key-path [data-testid="stElementContainer"] { height:auto !important; }
  .st-key-playerbar { position:fixed; left:0; right:0; bottom:0; z-index:999; }
  /* keep things scrolled to (or focused with the keyboard) clear of the player bar */
  [data-testid="stMain"] { scroll-padding-bottom:110px; scroll-padding-top:16px; }
  .st-key-playerbar iframe { display:block; width:100%; height:86px; border:none; }
  /* Settings > Larger text: the page's cards are drawn 15% bigger (the top bar, sidebar and player bar keep their size) */
  html.moodtune-large .greeting, html.moodtune-large .st-key-hero, html.moodtune-large .st-key-notices,
  html.moodtune-large .st-key-choice, html.moodtune-large .st-key-path, html.moodtune-large .st-key-quickpicks,
  html.moodtune-large .page-head, html.moodtune-large .st-key-explore, html.moodtune-large .st-key-explorelist,
  html.moodtune-large .st-key-library { zoom:1.15; }

  /* Home: the chips row, the reading inside the "Detect my mood" card, Quick picks */
  .st-key-chips { flex-wrap:wrap; row-gap:8px; padding:2px 2px 6px; margin:4px 0 0; align-items:center; }
  .st-key-chips::-webkit-scrollbar { display:none; }
  .st-key-chips [data-testid="stElementContainer"] { flex:none; width:auto !important; }
  .st-key-chips .stButton button { border-radius:10px; min-height:36px; padding:5px 14px; border:1px solid var(--line); background:var(--field);
                                   color:var(--ink); box-shadow:none; white-space:nowrap; }
  .st-key-chips .stButton button p { font-size:14px; font-weight:500; white-space:nowrap; }
  .st-key-chips .stButton button:hover { border-color:var(--lavender); background:var(--hover); color:var(--purple); }
  .st-key-chips .stButton button[kind="primary"] { background:var(--purple); border-color:var(--purple); color:var(--on-purple); }
  .st-key-chips .stButton button[kind="primary"] p { font-weight:600; }
  .st-key-chips .stButton button:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  .chip-sep { display:block; width:1px; height:26px; background:var(--line); margin:0 6px; }
  .st-key-chips [data-testid="stElementContainer"]:has(.chip-sep) [data-testid="stMarkdownContainer"] { margin:0 !important; }
  [data-testid="stLayoutWrapper"]:has(> .st-key-playcmd) { display:none; }
  .st-key-hero .st-key-moodcard { margin-top:4px; box-shadow:none; border:none; }
  .result-wait { border:1px dashed rgba(255,255,255,.28); border-radius:20px; padding:20px 22px; color:#e3def7; }
  .result-wait b { display:block; font-size:15.5px; color:#fbf8f3; font-weight:600; }
  .result-wait span { display:block; font-size:13.5px; margin-top:4px; color:#cfc8ef; }
  .st-key-quickpicks { margin-top:10px; }
  .st-key-quickpicks iframe { display:block; width:100%; border:none; }
  .st-key-quickpicks [data-testid="stElementContainer"] { height:auto !important; }

  /* Explore and Library */
  .page-head { margin:22px 0 6px; }
  .page-head h1 { font-size:34px; font-weight:600; color:var(--ink); margin:0; padding:0; letter-spacing:-.01em; }
  .page-head p { color:var(--muted); font-size:15px; margin:6px 0 0; max-width:70ch; }
  .section-head { margin:18px 0 2px; }
  .section-head h2 { font-size:22px; font-weight:600; color:var(--ink); margin:0; padding:0; }
  .section-head p { color:var(--muted); font-size:13.5px !important; margin:4px 0 0 !important; }
  .st-key-moodcards, .st-key-langcards { display:grid !important; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:12px; }
  .st-key-langcards { grid-template-columns:repeat(auto-fill, minmax(176px, 1fr)); }
  [class*="st-key-xcard_"] { position:relative; }
  .xcard { border:2px solid var(--line); border-radius:18px; background:var(--card); padding:14px 16px; min-height:112px; height:100%;
           transition:border-color .15s, box-shadow .15s, background .15s; }
  .xcard .accent { width:34px; height:5px; border-radius:999px; margin-bottom:12px; }
  .xcard .emoji { font-size:26px; line-height:1; float:right; }
  .xcard h3 { font-size:18px; font-weight:600; margin:0; padding:0; color:var(--ink); text-transform:capitalize; }
  .xcard .native { font-size:14px; color:var(--muted); margin-top:2px; }
  .xcard p { font-size:12.5px; color:var(--muted); margin:6px 0 0; }
  .xcard .modes { display:block; font-size:12px; font-weight:600; margin-top:8px; color:var(--purple); line-height:1.45; }
  .xcard .modes span { display:block; }
  .xcard .modes.only { display:inline-block; border-radius:999px; padding:2px 9px; background:var(--terra-soft); color:var(--terra-ink); }
  .xcard.on { border-color:var(--purple); background:var(--hover); box-shadow:0 6px 18px rgba(45,42,82,.12); }
  [class*="st-key-xcard_"]:hover .xcard:not(.on) { border-color:#cfc8ef; }
  [class*="st-key-xcard_"] [data-testid="stMarkdown"], [class*="st-key-xcard_"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; height:100%; }
  [class*="st-key-xcard_"] [data-testid="stElementContainer"]:has(.stButton) { position:absolute; inset:0; z-index:2; margin:0; }
  [class*="st-key-xcard_"] .stButton, [class*="st-key-xcard_"] .stButton button { width:100%; height:100%; }
  [class*="st-key-xcard_"] .stButton button { opacity:0; cursor:pointer; }
  [class*="st-key-xcard_"]:has(button:focus-visible) .xcard { outline:3px solid var(--lavender); outline-offset:3px; }
  .st-key-explorelist { margin-top:14px; }
  .st-key-explorelist iframe, .st-key-library iframe { display:block; width:100%; border:none; }
  .st-key-explorelist [data-testid="stElementContainer"], .st-key-library [data-testid="stElementContainer"] { height:auto !important; }
  .st-key-library { margin-top:8px; }

  /* the name dialog (New playlist, Rename, Your name), drawn on the page by the components */
  #moodtune-dialog { position:fixed; inset:0; z-index:1004; display:flex; align-items:center; justify-content:center; padding:16px; }
  #moodtune-dialog .mtd-shade { position:absolute; inset:0; background:rgba(43,40,64,.4); }
  #moodtune-dialog .mtd-box { position:relative; width:min(420px, 100%); background:var(--cream); border-radius:20px; padding:22px 22px 18px;
                              box-shadow:0 24px 60px rgba(45,42,82,.3); }
  #moodtune-dialog h2 { font-size:24px; font-weight:600; margin:0 0 14px; padding:0; color:var(--ink); }
  #moodtune-dialog .mtd-label { display:block; font-size:13px; color:var(--muted); font-weight:600; margin:0 0 6px; }
  #moodtune-dialog input { width:100%; border:1px solid var(--line); border-radius:12px; padding:11px 14px; font:inherit; font-size:15px;
                           background:var(--field); color:var(--ink); }
  #moodtune-dialog input:focus { outline:3px solid var(--lavender); border-color:transparent; }
  #moodtune-dialog .mtd-hint { font-size:12.5px; color:var(--muted); margin:8px 0 16px; }
  #moodtune-dialog .mtd-actions { display:flex; justify-content:flex-end; gap:8px; }
  #moodtune-dialog button { border-radius:999px; font:inherit; font-size:14px; font-weight:600; padding:9px 18px; cursor:pointer;
                            border:1px solid var(--line); background:var(--field); color:var(--ink); }
  #moodtune-dialog .mtd-ok { background:var(--purple); color:var(--on-purple); border-color:var(--purple); }
  #moodtune-dialog button:focus-visible { outline:3px solid var(--lavender); outline-offset:2px; }
  /* short messages from the song list and the player ("Added to the queue", "Link copied") */
  #moodtune-toast { position:fixed; left:50%; bottom:104px; transform:translateX(-50%); z-index:1003; pointer-events:none;
                    background:var(--purple); color:var(--on-purple); font-size:14px; padding:10px 18px; border-radius:999px; max-width:calc(100vw - 32px);
                    box-shadow:0 10px 30px rgba(45,42,82,.25); opacity:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #moodtune-toast.show { animation:moodtune-toast 2.8s ease forwards; }
  @keyframes moodtune-toast { 0% { opacity:0; transform:translate(-50%, 8px); } 8%, 82% { opacity:1; transform:translate(-50%, 0); } 100% { opacity:0; } }
  @media (prefers-reduced-motion: reduce) { #moodtune-toast.show { animation-duration:3.2s; animation-timing-function:steps(1, end); } }

  /* alerts */
  .st-key-notices [data-testid="stAlert"] { border-radius:14px; }

  /* narrow windows: the sidebar slides in over the page (the hamburger), with a shade behind it */
  @media (max-width: 760px) {
    [data-testid="stMain"], html.mt-nav-mini [data-testid="stMain"] { padding-left:0; }
    .st-key-nav, html.mt-nav-mini .st-key-nav { width:264px; padding:12px 12px 0; transform:translateX(-105%); z-index:1001;
                                                box-shadow:14px 0 40px rgba(45,42,82,.18); visibility:hidden; }
    html.mt-nav-open .st-key-nav { transform:none; visibility:visible; }
    html.mt-nav-open #moodtune-navshade { display:block; position:fixed; inset:0; z-index:1000; background:rgba(43,40,64,.32); }
    .st-key-language [data-testid="stWidgetLabel"] { display:none; }
    .st-key-language [data-testid="stSelectbox"] > div:last-child { width:112px; }
    .st-key-topbar { padding:0 8px 0 4px; }
    /* the chips scroll sideways in one row */
    .st-key-chips { flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none; }
  }
  @media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding:calc(var(--top) + 6px) 16px 116px; }
    .page-head h1 { font-size:28px; }
    .st-key-moodcards { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .st-key-langcards { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .greeting h1 { font-size:28px; }
    .st-key-hero { padding:22px 16px; border-radius:22px; }
    .st-key-hero h2 { font-size:30px; }
    .st-key-moodcard, .st-key-choice { padding:20px 16px; border-radius:20px; }
    .mood-title { font-size:25px; } .mood-head .emoji { font-size:42px; }
    .mood-grid { grid-template-columns:minmax(0,1fr); gap:18px; } .gauge-box { max-width:230px; margin:0 auto; }
  }
</style>
"""


# Settings > Theme > Dark, for the page itself (the components have their own, in player.DARK_CSS).
# __D__ stands for the selector that turns it on (see page_css).
DARK_PAGE_CSS = """
<style>
  __D__ {
          --hero:#262246; --purple-2:#302b59; --field:#2a2649; --hover:#2b2750; --hover-soft:rgba(255,255,255,.08); --on-purple:#191630;
          --card2:#1b1833; --rail:#3a3560; --uplift-on:#2f211d;
          --cream:#15132a; --card:#211e3b; --ink:#ece9f7; --muted:#aaa4c6; --line:#3a3560; --purple:#c3b9f2;
          --lav-soft:#322d5a; --terra:#e0845f; --terra-soft:#3a2721; --terra-ink:#f2b49a; }
  __D__ .stApp, __D__ [data-testid="stAppViewContainer"], __D__ [data-testid="stMain"] { background:var(--cream); color:var(--ink); }
  __D__ [data-testid="stMarkdownContainer"], __D__ [data-testid="stWidgetLabel"] p, __D__ [data-testid="stText"] { color:var(--ink); }
  __D__ [data-testid="stCaptionContainer"] { color:var(--muted); }
  __D__ .st-key-hero [data-testid="stMarkdownContainer"] { color:#f1edfb; }
  __D__ [data-testid="stSelectbox"] [role="group"] { background:var(--field); border-color:var(--line); color:var(--ink); }
  __D__ [data-testid="stSelectbox"] input, __D__ [data-testid="stSelectbox"] button { color:var(--ink); }
  __D__ [role="listbox"], __D__ [role="listbox"] ul { background:var(--card) !important; color:var(--ink); }
  __D__ [role="option"] { background:var(--card); color:var(--ink); }
  __D__ [role="option"][aria-selected="true"], __D__ [role="option"]:hover { background:var(--lav-soft) !important; }
  __D__ [data-testid="stPopoverBody"] { background:var(--card); color:var(--ink); border-color:var(--line); }
  __D__ [data-testid="stFileUploaderDropzone"] { background:var(--card2); color:var(--ink); }
  __D__ [data-testid="stFileUploaderDropzone"] small, __D__ [data-testid="stFileUploaderDropzone"] span { color:var(--muted); }
  __D__ [data-testid="stExpander"] details { border-color:var(--line); }
  __D__ [data-testid="stExpander"] summary, __D__ [data-testid="stExpander"] summary svg { color:var(--purple); }
  __D__ [data-testid="stAlertContainer"] { background:#2b2750; color:var(--ink); }
  __D__ [data-testid="stAlertContainer"] p, __D__ [data-testid="stAlertContainer"] svg { color:var(--ink); }
  __D__ .stButton button[kind="secondary"], __D__ [data-testid="stBaseButton-secondary"] { background:var(--field); color:var(--ink); border-color:var(--line); }
  __D__ .st-key-rescan button { background:#fff !important; color:#2d2a52 !important; border-color:#fff !important; }
  __D__ .st-key-nav .stButton button { color:var(--ink); }
  __D__ .st-key-nav .stButton button[kind="primary"] { background:var(--lav-soft); color:var(--purple); }
  __D__ .st-key-hero .st-key-moodcard { border:1px solid var(--line); }
  __D__ .gauge path:first-child { stroke:var(--rail); } __D__ .gauge path:last-child { stroke:var(--purple); }
  __D__ .bar.combined .fill, __D__ .legend i.combined { background:var(--purple) !important; }
  __D__ .band-light { background:#203a29; color:#a9dbb7; }
</style>
"""


def page_css(dark_first=False):
    """The page's CSS. The components put the class mt-dark (Settings > Theme > Dark) or
    mt-light on the page; with dark_first (the theme cookie says dark), the dark theme
    also applies before they have, so the first paint is already dark."""
    selector = ":is(html.mt-dark, html:not(.mt-light))" if dark_first else "html.mt-dark"
    return PAGE_CSS + DARK_PAGE_CSS.replace("__D__", selector)


def page_head_html(title, text):
    return f'<div class="page-head"><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>'


def section_head_html(title, text):
    return f'<div class="section-head"><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p></div>'


# Explore's tiles. Each is drawn here and covered by a transparent Streamlit
# button (see the CSS), like the choice cards.
def mood_tile_html(emotion, count, selected):
    """A mood tile: its colour, emoji, name and how many songs it has (with a preview, in the chosen language)."""
    return (f'<div class="xcard mood{" on" if selected else ""}" aria-hidden="true">'
            f'<span class="emoji">{mood.EMOJI[emotion]}</span><div class="accent" style="background:{mood.COLOURS[emotion]}"></div>'
            f'<h3>{emotion}</h3><p>{count:,} song{"" if count == 1 else "s"}</p></div>')


def language_tile_html(name, native, count, uplift, selected):
    """A language tile: its name (and in its own script), how many songs, and which of the two modes it offers."""
    native_html = f'<div class="native">{html.escape(native)}</div>' if native != name else '<div class="native">&nbsp;</div>'
    modes = ('<span class="modes"><span>✓ Lift my mood</span><span>✓ Meet me here</span></span>' if uplift
             else '<span class="modes only">Meet me here only</span>')
    return (f'<div class="xcard lang{" on" if selected else ""}" aria-hidden="true"><h3>{html.escape(name)}</h3>{native_html}'
            f'<p>{count:,} song{"" if count == 1 else "s"} with a preview</p>{modes}</div>')


def local_now(tz_name=None, offset_minutes=None, now=None):
    """The listener's local time: the browser's time zone if known, else its UTC offset, else the server's clock."""
    now = now or datetime.now(timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            return now.astimezone(ZoneInfo(tz_name))
        except Exception:
            pass
    if offset_minutes is not None:
        # The browser's offset follows JavaScript's getTimezoneOffset(): minutes *behind* UTC.
        return now.astimezone(timezone(-timedelta(minutes=offset_minutes)))
    return now.astimezone()


def greeting(local):
    hour = local.hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


def greeting_html(local):
    date = f"{local:%A}, {local.day} {local:%B}"
    return f'<div class="greeting"><div class="date">{date}</div><h1>{greeting(local)}</h1></div>'


HERO_HTML = ('<div><h2>Detect my mood</h2>'
             '<p class="hero-sub">MOOD TUNE reads your expression and your words together.</p></div>')

# Under the scan and the words, before there is a reading.
RESULT_WAIT_HTML = ('<div class="result-wait"><b>Your mood reading appears here</b>'
                    '<span>Look at the camera, or tell MOOD TUNE how your day was.</span></div>')


def scan_screen(title, detail=""):
    return (f'<div class="listening"><div class="pulse"></div><div>{title}</div>'
            f'<div class="listening-detail">{detail}</div></div>')


def face_tile_html(image, note, tip=None):
    """The scan picture (an in-memory data URI, never a file), its note and an optional tip line under it."""
    if image:
        body = f'<img class="scanimg" src="{image}" alt="The picture used for your scan, with the face box and label">'
    else:
        body = '<div class="scan-empty">No picture for this scan</div>'
    note_html = f'<p class="tile-note">{html.escape(note)}</p>' if note else ""
    if note and tip:
        note_html = (f'<p class="tile-note with-tip">{html.escape(note)}</p>'
                     f'<p class="tile-note tile-tip">{html.escape(tip)}</p>')
    return f'<p class="tile-label">Your expression</p>{body}{note_html}'


READING_HTML = '<div class="reading"><span class="dot"></span>Reading your text...</div>'


def headline_html(card):
    """'Mostly <b>sad</b>' or 'Mixed: <b>happy</b> and <b>sad</b>', each emotion in its colour."""
    head = card["headline"]
    if head.startswith("Mostly "):
        return f'Mostly <b style="color:{card["colour"]}">{html.escape(head[7:])}</b>'
    first, second = head[len("Mixed: "):].split(" and ")
    return (f'Mixed: <b style="color:{card["colour"]}">{html.escape(first)}</b> and '
            f'<b style="color:{card["second_colour"]}">{html.escape(second)}</b>')


# The gauge's bands, by the share of sad + angry + fear + disgust (uplift.negative_share).
BANDS = [(0.25, "Light", "Lighter feelings lead in this reading."),
         (0.50, "In between", "Lighter and heavier feelings sit side by side in this reading."),
         (1.01, "Low", "Heavier feelings lead in this reading right now.")]


def band(share):
    """(label, sentence) for a negative share: under 25% Light, 25-50% In between, 50% or more Low.

    The middle band isn't called "Mixed", so it doesn't read like the
    headline's "Mixed: X and Y" (which is about the top two emotions).
    """
    return next((label, sentence) for limit, label, sentence in BANDS if share < limit)


def percent(share):
    """Whole percent, rounded down, so the number shown never crosses a band edge the share hasn't."""
    return int(share * 100 + 1e-9)


def gauge_svg(share):
    """A half-circle gauge filled to `share`."""
    return ('<svg class="gauge" viewBox="0 0 200 112" role="img" aria-hidden="true">'
            '<path d="M18 100 A82 82 0 0 1 182 100" pathLength="100" fill="none" stroke="#eeebfa" stroke-width="16" stroke-linecap="round"/>'
            f'<path d="M18 100 A82 82 0 0 1 182 100" pathLength="100" fill="none" stroke="#2d2a52" stroke-width="16" '
            f'stroke-linecap="round" stroke-dasharray="{max(share * 100, 0.01):.2f} 200"/></svg>')


def lean_html(key, name, share):
    return (f'<div class="lean" data-source="{key}" data-share="{share:.6f}"><div class="lean-top"><span>{name} leans low</span>'
            f'<b>{percent(share)}%</b></div><div class="rail"><div class="fill" style="width:{share * 100:.1f}%"></div></div></div>')


def mood_card_html(card, low):
    """The mood reading: the "X% low" gauge with its band, the headline, and how low each source leans.

    low: {"combined": share, "face": share or None, "text": share or None},
    each the negative share of that source's calibrated probabilities.
    """
    share = low["combined"]
    label, sentence = band(share)
    sources = [(k, n, low[k]) for k, n in (("face", "Face"), ("text", "Text")) if low.get(k) is not None]
    leans = "".join(lean_html(k, n, v) for k, n, v in sources)
    blend = '<p class="blend">Your result blends both signals.</p>' if len(sources) == 2 else ""
    return (f'<p class="card-kicker">Your mood reading</p>'
            f'<div class="mood-grid">'
            f'<div class="gauge-box" id="gauge" data-share="{share:.6f}" role="img" aria-label="{percent(share)}% low, {label}">'
            f'{gauge_svg(share)}<div class="gauge-num"><span id="gaugepct">{percent(share)}%</span> low</div>'
            f'<div class="band band-{label.lower().replace(' ', '-')}" id="band">{label}</div></div>'
            f'<div class="mood-main"><div class="mood-head"><span class="emoji" id="moodemoji">{card["emoji"]}</span>'
            f'<div class="mood-title" id="moodhead" role="heading" aria-level="2">{headline_html(card)}</div></div>'
            f'<p class="mood-sentence" id="bandtext">{sentence}</p>'
            f'<div class="leans">{leans}</div>{blend}</div></div>')


def bars_html(bars):
    """The Face / Text / Combined bars for all seven emotions, sorted by the combined value."""
    legend = "".join(f'<span><i class="{g["key"]}" style="background:{BAR_COLOURS[g["key"]]}"></i>{BAR_NAMES[g["key"]]}</span>'
                     for g in bars["groups"])
    rows = []
    for k, emotion in enumerate(bars["emotions"]):
        stack = "".join(
            f'<div class="bar {g["key"]}" data-group="{g["key"]}" data-value="{g["values"][k]}">'
            f'<div class="rail"><div class="fill" style="width:{g["values"][k] * 100:.1f}%;background:{BAR_COLOURS[g["key"]]}"></div></div>'
            f'<span class="pct">{round(g["values"][k] * 100)}%</span></div>'
            for g in bars["groups"])
        rows.append(f'<div class="erow" data-emotion="{emotion}"><div class="ename">{emotion}</div>'
                    f'<div class="ebars">{stack}</div></div>')
    return (f'<div class="legend" id="legend">{legend}</div><div id="bars">{"".join(rows)}</div>'
            f'<p class="barcap" id="barcap">{html.escape(bars["caption"])}</p>')


CHOICE_HTML = '<div><h2>What do you need from music?</h2><p class="choice-sub">Pick one, or keep the recommended one.</p></div>'

CHOICES = {
    "Uplift": ("uplift", "🌅", "Lift my mood", "Start where you are and move, song by song, towards something lighter."),
    "Resonance": ("resonance", "🌙", "Meet me here", "Songs that match how you feel right now."),
}


def choice_card_html(mode, selected, recommended, fallback=None):
    cls, icon, title, text = CHOICES[mode]
    tag = '<span class="tag">Recommended</span>' if recommended else ""
    note = f'<div class="fallback">{html.escape(fallback)}</div>' if fallback else ""
    return (f'<div class="choice-card {cls}{" on" if selected else ""}" aria-hidden="true">{tag}'
            f'<div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p>{note}'
            f'<div class="check">✓ Selected</div></div>')
