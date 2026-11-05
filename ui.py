"""The page's look: CSS for the Streamlit page and the small HTML pieces it shows.

Everything interactive is a Streamlit widget (styled here with CSS) or one of
the two components in player.py. Nothing here touches Streamlit's state or
the models; the functions only turn values into HTML. Text that comes from
outside (song titles, the user's words) never goes through here.
"""
import html
from datetime import datetime, timedelta, timezone

APP_NAME = "MOOD TUNE"
TAGLINE = "Music that meets your mood"

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
  :root { --cream:#f6f3ee; --card:#ffffff; --ink:#2b2840; --muted:#625e78; --line:#e7e1d8; --purple:#2d2a52;
          --purple-2:#3a3668; --lavender:#b9b0e8; --lav-soft:#eeebfa; --terra:#b85c38; --terra-soft:#f8e9e1; --terra-ink:#7a3a20;
          --serif:"Fraunces", Georgia, "Times New Roman", serif; }
  #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
  .stApp { background:var(--cream); }
  [data-testid="stMainBlockContainer"] { max-width:1180px; padding:18px 28px 132px; }
  h1, h2, h3 { font-family:var(--serif) !important; }
  /* no anchor-link icons next to the page's headings */
  [data-testid="stHeaderActionElements"] { display:none !important; }

  /* 1. top bar and greeting */
  .st-key-topbar { border-bottom:1px solid var(--line); padding-bottom:12px; }
  .app-head { display:flex; align-items:center; gap:12px; }
  .app-head .logo { flex:none; }
  .app-name { font-family:var(--serif); font-size:22px; font-weight:700; color:var(--purple); letter-spacing:.06em; line-height:1.1; }
  .app-tagline { font-size:13px; color:var(--muted); margin-top:2px; }
  .st-key-language [data-testid="stSelectbox"] { display:flex; flex-direction:row; align-items:center; justify-content:flex-end; gap:10px; }
  .st-key-language [data-testid="stWidgetLabel"] { margin:0; min-height:0; flex:none; }
  .st-key-language [data-testid="stWidgetLabel"] p { font-size:14px; color:var(--muted); }
  .st-key-language [data-testid="stSelectbox"] > div:last-child { width:150px; flex:none; }
  .greeting { margin:26px 0 4px; }
  .greeting .date { font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; }
  .greeting h1 { font-size:34px; font-weight:600; color:var(--ink); margin:4px 0 0; padding:0; letter-spacing:-.01em; }

  /* 2. hero card */
  .st-key-hero { background:var(--purple); border-radius:26px; padding:34px 34px 30px; gap:22px;
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
  .tile-note { color:#e3def7; font-size:13px !important; line-height:1.45; margin:10px 0 0 !important; }
  .listening { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; height:220px;
               border-radius:14px; background:#25224a; color:#f1edfb; font-size:17px; }
  .pulse { width:58px; height:58px; border-radius:50%; background:var(--lavender); animation:pulse 1.3s ease-in-out infinite; }
  .listening-detail { font-size:13.5px; color:#cfc8ef; margin-top:-6px; }
  @keyframes pulse { 0%, 100% { transform:scale(.75); opacity:.55; } 50% { transform:scale(1.08); opacity:1; } }
  .st-key-hero [data-testid="stWidgetLabel"] p { color:#f1edfb; font-size:15px; font-weight:500; }
  .st-key-hero [data-testid="stTextInput"] [data-baseweb="input"] { background:#fff; border-radius:12px; border:2px solid transparent; }
  .st-key-hero [data-testid="stTextInput"] [data-baseweb="input"]:focus-within { border-color:var(--lavender); }
  .st-key-hero [data-testid="stTextInput"] input { color:var(--ink); background:#fff; font-size:15px; padding:12px 14px; }
  .st-key-hero [data-testid="InputInstructions"] { color:var(--muted); }
  .st-key-hero [data-testid="stForm"] { border:none; padding:0; }
  .st-key-hero button { border-radius:999px !important; font-weight:600; padding:8px 18px; min-height:42px; }
  .st-key-hero button[kind="primaryFormSubmit"], .st-key-hero button[data-testid="stBaseButton-primaryFormSubmit"] {
       background:var(--lavender); color:var(--purple); border:none; }
  .st-key-hero button[kind="primaryFormSubmit"]:hover { background:#cdc6f1; color:var(--purple); }
  .st-key-rescan button { background:#fff; color:var(--purple); border:none; }
  .st-key-rescan button:hover { background:#f1edfb; color:var(--purple); }
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
  .band-light { background:#e7f3ea; color:#2c5e3a; } .band-mixed { background:var(--lav-soft); color:var(--purple); }
  .band-low { background:var(--terra-soft); color:var(--terra-ink); }
  .mood-sentence { font-size:15.5px !important; color:var(--ink); margin:10px 0 16px !important; }
  .leans { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px 26px; max-width:560px; }
  .lean-top { display:flex; justify-content:space-between; font-size:13.5px; color:var(--muted); margin-bottom:6px; }
  .lean-top b { color:var(--ink); font-variant-numeric:tabular-nums; }
  .lean .rail { background:#eeeaf6; border-radius:4px; height:8px; overflow:hidden; }
  .lean .fill { height:100%; border-radius:4px; background:#7b6fd0; }
  .lean[data-source="face"] .fill { background:#8d88a6; }
  .blend { font-size:13px !important; color:var(--muted); margin:12px 0 0 !important; }
  .st-key-moodcard [data-testid="stExpander"] details { border:1px solid var(--line); border-radius:14px; background:#fcfbf8; }
  .st-key-moodcard [data-testid="stExpander"] summary p { font-weight:600; color:var(--purple); }
  .legend { display:flex; gap:16px; font-size:12.5px; color:var(--muted); margin:2px 0 14px; flex-wrap:wrap; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
  .erow { display:grid; grid-template-columns:76px 1fr; gap:12px; align-items:center; margin-bottom:12px; }
  .ename { font-size:14px; text-transform:capitalize; color:var(--ink); }
  .ebars { display:flex; flex-direction:column; gap:4px; }
  .bar { display:grid; grid-template-columns:1fr 42px; gap:10px; align-items:center; }
  .bar .rail { background:#eeeaf6; border-radius:4px; height:7px; overflow:hidden; }
  .bar .fill { height:100%; border-radius:4px; }
  .bar .pct { font-size:12px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  .bar.combined .rail { height:10px; } .bar.combined .pct { color:var(--ink); font-weight:600; }
  .barcap { color:var(--muted); font-size:13px !important; margin:6px 0 0 !important; }

  /* 4. the choice cards: the visible card is HTML; a transparent Streamlit button covers it */
  .st-key-choice h2 { font-size:26px; font-weight:600; color:var(--ink); margin:0; padding:0; }
  .choice-sub { color:var(--muted); font-size:14.5px !important; margin:8px 0 6px !important; }
  .st-key-choice_uplift, .st-key-choice_resonance { position:relative; }
  .choice-card { border:2px solid var(--line); border-radius:18px; padding:18px 20px; min-height:128px; background:#fff;
                 transition:border-color .15s, box-shadow .15s, background .15s; }
  .choice-card .icon { font-size:24px; line-height:1; }
  .choice-card h3 { font-size:21px; font-weight:600; margin:8px 0 0; padding:0; color:var(--ink); }
  .choice-card p { font-size:14px; color:var(--muted); margin:4px 0 0; line-height:1.45; }
  .choice-card .tag { display:inline-block; font-size:11.5px; font-weight:600; letter-spacing:.04em; border-radius:999px;
                      padding:3px 10px; background:var(--lav-soft); color:var(--purple); float:right; }
  .choice-card .fallback { margin-top:10px; font-size:13px; color:var(--terra-ink); background:var(--terra-soft); border-radius:10px; padding:8px 11px; }
  .choice-card.uplift.on { border-color:var(--terra); background:#fffaf7; box-shadow:0 6px 18px rgba(184,92,56,.14); }
  .choice-card.resonance.on { border-color:var(--purple); background:#fbfaff; box-shadow:0 6px 18px rgba(45,42,82,.12); }
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
  .st-key-playerbar { position:fixed; left:0; right:0; bottom:0; z-index:999; }
  /* keep things scrolled to (or focused with the keyboard) clear of the player bar */
  [data-testid="stMain"] { scroll-padding-bottom:110px; scroll-padding-top:16px; }
  .st-key-playerbar iframe { display:block; width:100%; height:86px; border:none; }

  /* alerts */
  .st-key-notices [data-testid="stAlert"] { border-radius:14px; }

  @media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding:12px 16px 116px; }
    .greeting h1 { font-size:28px; }
    .st-key-hero { padding:22px 16px; border-radius:22px; }
    .st-key-hero h2 { font-size:30px; }
    .st-key-moodcard, .st-key-choice { padding:20px 16px; border-radius:20px; }
    .mood-title { font-size:25px; } .mood-head .emoji { font-size:42px; }
    .mood-grid { grid-template-columns:minmax(0,1fr); gap:18px; } .gauge-box { max-width:230px; margin:0 auto; }
    .app-tagline { display:none; }
  }
</style>
"""


def topbar_html():
    return (f'<div class="app-head">{LOGO_SVG}<div><div class="app-name">{APP_NAME}</div>'
            f'<div class="app-tagline">{TAGLINE}</div></div></div>')


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


HERO_HTML = ('<div><h2>How are you, really?</h2>'
             '<p class="hero-sub">MOOD TUNE combines your expression and your words.</p></div>')


def scan_screen(title, detail=""):
    return (f'<div class="listening"><div class="pulse"></div><div>{title}</div>'
            f'<div class="listening-detail">{detail}</div></div>')


def face_tile_html(image, note):
    """The scan picture (an in-memory data URI, never a file) and its note."""
    if image:
        body = f'<img class="scanimg" src="{image}" alt="The picture used for your scan, with the face box and label">'
    else:
        body = '<div class="scan-empty">No picture for this scan</div>'
    note_html = f'<p class="tile-note">{html.escape(note)}</p>' if note else ""
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
         (0.50, "Mixed", "There's a mix of lighter and heavier feelings in this reading."),
         (1.01, "Low", "Heavier feelings lead in this reading right now.")]


def band(share):
    """(label, sentence) for a negative share: under 25% Light, 25-50% Mixed, 50% or more Low."""
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
            f'<div class="band band-{label.lower()}" id="band">{label}</div></div>'
            f'<div class="mood-main"><div class="mood-head"><span class="emoji" id="moodemoji">{card["emoji"]}</span>'
            f'<div class="mood-title" id="moodhead" role="heading" aria-level="2">{headline_html(card)}</div></div>'
            f'<p class="mood-sentence" id="bandtext">{sentence}</p>'
            f'<div class="leans">{leans}</div>{blend}</div></div>')


def bars_html(bars):
    """The Face / Text / Combined bars for all seven emotions, sorted by the combined value."""
    legend = "".join(f'<span><i style="background:{BAR_COLOURS[g["key"]]}"></i>{BAR_NAMES[g["key"]]}</span>'
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
