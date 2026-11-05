"""Logic end-to-end test of the redesigned app with a fake camera and a fake preview service."""
import hashlib, io, os, sys, tempfile
import numpy as np, pandas as pd, cv2
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import previews, json

# Optional: make the app's unseeded random picks repeatable (each call gets the next seed),
# so two versions of the code can be compared on the same scenarios.
if os.environ.get("TEST_SEED"):
    _real_rng, _seed = np.random.default_rng, [int(os.environ["TEST_SEED"])]
    def _seeded_rng(*args, **kw):
        if args or kw: return _real_rng(*args, **kw)
        _seed[0] += 1; return _real_rng(_seed[0])
    np.random.default_rng = _seeded_rng

# fake preview service: ~70% of songs have a preview, decided by a hash of the title
lookups = []
def fake_search(title, artist):
    lookups.append(title)
    if int(hashlib.md5(f"{title}|{artist}".encode()).hexdigest(), 16) % 10 < 7:
        return {"preview_url": "https://example.test/preview.m4a", "artwork_url": "https://example.test/art.jpg",
                "itunes_title": title, "itunes_artist": artist}
    return None
previews.search_itunes = fake_search
previews.CACHE_PATH = os.path.join(tempfile.mkdtemp(), "previews.csv")
previews._cache = None

EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
songs = pd.read_csv(f"{ROOT}/data/songs/songs_clean.csv", keep_default_na=False)
z = np.load(f"{ROOT}/data/fusion/text_test.npz"); probs, labels = z["probs"], z["labels"].astype(str)
tp = pd.read_csv(f"{ROOT}/models/text_emotion_distilbert/test_predictions.csv", keep_default_na=False)
texts = {}
for k, e in enumerate(EMO):
    ok = (labels == e) & (probs.argmax(1) == k) & (tp.text.str.len().between(25, 110).values) & ~tp.text.str.contains(r"\[NAME\]").values
    texts[e] = tp.text[np.flatnonzero(ok)[probs[ok, k].argmax()]]

p = pd.read_parquet(f"{ROOT}/data/face_emotion/raw/privateTest-0000.parquet")
fz = np.load(f"{ROOT}/data/fusion/face_test.npz"); fprobs, flabels = fz["probs"], fz["labels"].astype(str)
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def frame_for(img48):
    canvas = np.full((480, 640), 128, np.uint8); canvas[120:360, 200:440] = cv2.resize(img48, (240, 240), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
def face_frame(emotion):
    k = EMO.index(emotion)
    for i in np.argsort(-fprobs[:, k]):
        if flabels[i] != emotion: continue
        f = frame_for(np.asarray(Image.open(io.BytesIO(p.image.iloc[i]["bytes"])).convert("L")))
        if len(haar.detectMultiScale(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5)) == 1:
            return f
happy_frame, sad_frame = face_frame("happy"), face_frame("sad")
blank = np.full((480, 640, 3), 128, np.uint8)

class FakeCap:
    frame, opened = None, 0
    def __init__(self, *a): FakeCap.opened += 1
    def read(self): return True, FakeCap.frame.copy()
    def release(self): pass
    def isOpened(self): return True
cv2.VideoCapture = FakeCap

def start(frame):
    FakeCap.frame = frame
    at = AppTest.from_file(f"{ROOT}/app.py", default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    return at

def describe(at):
    if "queue" not in at.session_state:
        return "NO QUEUE | " + " | ".join(w.value for w in at.warning)
    q = at.session_state["queue"]; t = q["tracks"]
    chip = headline(at)
    sections = pd.Series([x["section"] for x in t]).value_counts().to_dict()
    journey = [x for x in t if x["section"] == "journey"]
    out = f"{q['mode']:9} | tracks {len(t)} {sections} | all have preview: {all(x['preview'] for x in t)}"
    if journey:
        v = [x["valence"] for x in journey]
        out += f" | journey buckets {[x['bucket'] for x in journey]} numbers {[x['number'] for x in journey]} valence {v} rising {all(b > a for a, b in zip(v, v[1:]))}"
        more = [x["bucket"] for x in t if x["section"] == "more"]
        if more: out += f" | then {len(more)} more from {sorted(set(more))}"
    else:
        out += f" | buckets {pd.Series([x['bucket'] for x in t]).value_counts().to_dict()}"
    out += f" | components {len(at.get('iframe'))} (path card + player bar) | note: {q['note']!r} | headline: {chip!r}"
    out += "\nRESULT      mood card: " + check_card(at)
    return out

SUBMIT = "FormSubmitter:textform-Check my mood"
CHOICE_KEYS = {"Uplift": "choose_uplift", "Resonance": "choose_resonance", "Auto": "choose_auto"}
def set_text(at, text):
    """Type into "How was your day?" and press "Check my mood"."""
    at.text_input(key="user_text").set_value(text)
    at.button(key=SUBMIT).click().run(); assert not at.exception, [e.value for e in at.exception]
def set_mode(at, mode):
    """Click a choice card ("Lift my mood" / "Meet me here"), or "Let MOOD TUNE choose" for Auto."""
    if mode == "Auto" and at.session_state["mode_choice"] == "Auto": return
    at.button(key=CHOICE_KEYS[mode]).click().run(); assert not at.exception, [e.value for e in at.exception]

import re
def card_html(at):
    return next((m.value for m in at.markdown if 'id="gauge"' in m.value), "")
def headline(at):
    m = re.search(r'id="moodhead"[^>]*>(.*?)</div></div>', card_html(at))
    return re.sub("<.*?>", "", m.group(1)) if m else ""

# --- the mood card, checked against the probabilities computed here, independently of the app
import fusion
from fusion import EMOTIONS as CANON
CONFIG = json.load(open(f"{ROOT}/models/fusion/fusion_config.json"))
NEGATIVE = ["sad", "angry", "fear", "disgust"]
def negative(probs):
    return float(sum(probs[CANON.index(e)] for e in NEGATIVE))
def expected_card(at):
    face = at.session_state["scan"]["face_probs"]
    text = at.session_state["text_results"].get(at.session_state["user_text"].strip())
    face_cal = None if face is None else fusion.calibrate(fusion.align_face(face), CONFIG["face_temperature"])
    text_cal = None if text is None else fusion.calibrate(fusion.align_text(text), CONFIG["text_temperature"])
    uniform = np.full(len(CANON), 1 / len(CANON))
    weight = CONFIG["face_weight"] if face is not None and text is not None else (1.0 if text is None else 0.0)
    _, fused = fusion.fuse(face if face is not None else uniform, text if text is not None else uniform, face_weight=weight,
                           method=CONFIG["method"], face_temperature=CONFIG["face_temperature"], text_temperature=CONFIG["text_temperature"])
    share = sum(fused[e] for e in NEGATIVE)
    ranked = sorted(fused, key=fused.get, reverse=True)
    head = f"Mostly {ranked[0]}" if fused[ranked[0]] >= 0.5 else f"Mixed: {ranked[0]} and {ranked[1]}"
    return {"share": share, "band": "Light" if share < 0.25 else "In between" if share < 0.5 else "Low", "head": head,
            "face": None if face_cal is None else negative(face_cal), "text": None if text_cal is None else negative(text_cal), "fused": fused}
def check_card(at):
    h = card_html(at); want = expected_card(at)
    share = float(re.search(r'id="gauge" data-share="([0-9.]+)"', h).group(1))
    pct = int(re.search(r'id="gaugepct">(\d+)%', h).group(1))
    band = re.search(r'id="band">([^<]+)<', h).group(1)
    leans = {k: float(v) for k, v in re.findall(r'class="lean" data-source="(\w+)" data-share="([0-9.]+)"', h)}
    shown = {k: int(v) for k, v in re.findall(r'data-source="(\w+)" data-share="[0-9.]+"><div class="lean-top"><span>\w+ leans low</span><b>(\d+)%', h)}
    blend = "Your result blends both signals" in h
    # the expander's Combined bars, against the fused probabilities
    bars = next(m.value for m in at.markdown if 'id="legend"' in m.value)
    combined = {e: int(v) for e, v in re.findall(r'data-emotion="(\w+)">.*?<div class="bar combined"[^>]*>.*?<span class="pct">(\d+)%', bars)}
    ok = {
        "gauge = negative share": abs(share - want["share"]) < 1e-6,
        "gauge % (rounded down)": pct == int(want["share"] * 100 + 1e-9),
        "band": band == want["band"],
        "face lean": (want["face"] is None and "face" not in leans) or ("face" in leans and abs(leans["face"] - want["face"]) < 1e-6 and shown["face"] == int(want["face"] * 100 + 1e-9)),
        "text lean": (want["text"] is None and "text" not in leans) or ("text" in leans and abs(leans["text"] - want["text"]) < 1e-6 and shown["text"] == int(want["text"] * 100 + 1e-9)),
        "blend note": blend == (want["face"] is not None and want["text"] is not None),
        "headline rule": headline(at) == want["head"],
        "combined bars": combined == {e: round(want["fused"][e] * 100) for e in combined} and len(combined) == 7,
    }
    lean_text = ", ".join(f"{k} {v:.3f}" for k, v in leans.items())
    return (f"gauge {pct}% low ({share:.4f}) {band} | leans {lean_text or 'none'} | blend note {blend} | "
            f"{'ALL MATCH' if all(ok.values()) else 'MISMATCH ' + str([k for k, v in ok.items() if not v])}")

lang_of = dict(zip(songs.lastfm_url, songs.language))
dump = {}
def record(name, at):
    dump[name] = [t["lastfm"] for t in at.session_state["queue"]["tracks"]] if "queue" in at.session_state else None

def set_language(at, language):
    at.selectbox(key="language").set_value(language).run(); assert not at.exception, [e.value for e in at.exception]

def languages_of(at):
    return pd.Series([lang_of[t["lastfm"]] for t in at.session_state["queue"]["tracks"]]).value_counts().to_dict()

print("RESULT == AUTO MODE, text only (no face)")
for e in EMO:
    at = start(blank); set_text(at, texts[e])
    print(f"RESULT [{e:8}] {describe(at)}"); record(f"text {e}", at)

print("RESULT == FORCED MODES")
at = start(blank); set_text(at, texts["sad"]); set_mode(at, "Resonance")
print(f"RESULT [sad text, Resonance forced] {describe(at)}"); record("sad text resonance", at)
at = start(blank); set_text(at, texts["happy"]); set_mode(at, "Uplift")
print(f"RESULT [happy text, Uplift forced ] {describe(at)}"); record("happy text uplift", at)

print("RESULT == FACES")
at = start(sad_frame);  print(f"RESULT [sad face only, auto        ] {describe(at)}"); record("sad face", at)
at = start(happy_frame); print(f"RESULT [happy face only, auto      ] {describe(at)}"); record("happy face", at)
at = start(happy_frame); set_text(at, "I miss her so much, everything feels empty and pointless now.")
print(f"RESULT [happy face + sad text, auto] {describe(at)}"); record("happy face sad text", at)
at = start(sad_frame); set_text(at, "This is absolutely infuriating, they lied to us again!")
print(f"RESULT [sad face + angry text, auto] {describe(at)}"); record("sad face angry text", at)
at = start(blank); print(f"RESULT [no face, no text           ] {describe(at)}")

if os.environ.get("TEST_DUMP"):
    json.dump(dump, open(os.environ["TEST_DUMP"], "w"), indent=0)
    print(f"RESULT English queues written to {os.environ['TEST_DUMP']}")
if os.environ.get("TEST_ENGLISH_ONLY"):
    sys.exit(0)

print("RESULT == LANGUAGES")
for language in ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Korean"]:
    for label, frame, text, mode in [("sad face, auto", sad_frame, "", None), ("happy face, auto", happy_frame, "", None),
                                     ("happy text, Uplift forced", blank, texts["happy"], "Uplift")]:
        at = start(frame)
        if text: set_text(at, text)
        if mode: set_mode(at, mode)
        set_language(at, language)
        q = at.session_state["queue"]
        langs = languages_of(at)
        print(f"RESULT [{language:9} {label:26}] {q['mode']:9} | tracks {len(q['tracks'])} | languages {langs} "
              f"| only {language}: {set(langs) == {language}} | subtitle {q['subtitle']!r} | note: {q['note']!r}")

print("RESULT == CHANGING THE LANGUAGE REBUILDS THE QUEUE WITHOUT THE CAMERA")
FakeCap.opened = 0
at = start(sad_frame); a = FakeCap.opened; scan_id = at.session_state["scan_id"]
seen = []
for language in ["Hindi", "Telugu", "Korean", "English"]:
    before = [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
    set_language(at, language)
    after = [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
    seen.append(f"{language}: rebuilt {before != after}, {at.session_state['queue']['mode']}, languages {languages_of(at)}")
print(f"RESULT camera opened after load {a}, after 4 language changes {FakeCap.opened} | same scan: {at.session_state['scan_id'] == scan_id}")
for line in seen: print(f"RESULT    {line}")

print("RESULT == CHOICE CARDS")
def cards(at):
    h = " ".join(m.value for m in at.markdown if "choice-card" in m.value)
    return {"selected": re.findall(r'class="choice-card \w+ on".*?<h3>(.*?)</h3>', h),
            "recommended": re.findall(r'<span class="tag">Recommended</span><div class="icon">.*?</div><h3>(.*?)</h3>', h),
            "fallback": re.findall(r'<div class="fallback">(.*?)</div>', h)}
def tracks(at): return [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
for label, frame in [("sad face", sad_frame), ("happy face", happy_frame)]:
    FakeCap.opened = 0
    at = start(frame); q0, m0, c0 = tracks(at), at.session_state["queue"]["mode"], cards(at)
    rec = "Uplift" if m0 == "Uplift" else "Resonance"
    other = "Resonance" if rec == "Uplift" else "Uplift"
    set_mode(at, rec); same = tracks(at) == q0
    set_mode(at, other); q1, m1, c1 = tracks(at), at.session_state["queue"]["mode"], cards(at)
    has_auto = any(b.key == "choose_auto" for b in at.button)
    set_mode(at, "Auto"); q2, m2 = tracks(at), at.session_state["queue"]["mode"]
    print(f"RESULT [{label}] Auto: {m0}, cards {c0} | clicking the recommended card keeps the queue: {same}")
    print(f"RESULT [{label}] clicking the other card: {m1}, rebuilt {q1 != q0}, cards {c1}, 'Let MOOD TUNE choose' shown {has_auto} "
          f"| back to Auto: {m2}, rebuilt {q2 != q1} | camera opened {FakeCap.opened} time(s) in all (1 = only the first scan)")
at = start(sad_frame); set_language(at, "Telugu")
print(f"RESULT [sad face, Telugu] {at.session_state['queue']['mode']} | cards {cards(at)}")
set_mode(at, "Uplift"); print(f"RESULT [sad face, Telugu, Lift my mood clicked] {at.session_state['queue']['mode']} | note {at.session_state['queue']['note']!r} | cards {cards(at)['selected']}")

print("RESULT == CAMERA IS ONLY OPENED ON LOAD AND RE-SCAN")
FakeCap.opened = 0
at = start(sad_frame); a = FakeCap.opened
key1 = at.session_state["queue_key"]; q1 = [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
set_mode(at, "Resonance"); set_mode(at, "Uplift"); set_text(at, "I feel awful today"); b = FakeCap.opened
set_mode(at, "Auto"); set_text(at, ""); c = FakeCap.opened
q2 = [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
at.run(); q3 = [t["lastfm"] for t in at.session_state["queue"]["tracks"]]
at.button(key="rescan").click().run(); d = FakeCap.opened
print(f"RESULT camera opened: after load {a}, after 3 setting changes {b}, after 2 more {c}, after Re-scan {d}")
print(f"RESULT queue unchanged by a plain rerun: {q2 == q3} | queue rebuilt after settings round-trip (new random picks): {q1 != q2}")
print(f"RESULT fake preview lookups made in total: {len(lookups)} | distinct songs cached: {len(previews._cache)}")
