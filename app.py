"""Emotion-based music player.

Detects the listener's mood from the webcam (and optional text), fuses the two
predictions, and plays matching songs (Resonance) or a short playlist that
moves towards a better mood (Uplift). See UPLIFT.md, models/README.md and
data/README.md for how each part works.
"""
import base64
import hashlib
import json
import math
import os
import time
import uuid
from urllib.parse import parse_qs, unquote

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from fusion import EMOTIONS, align_face, align_text, calibrate, fuse
import face
import face_models
import mood
import player
import previews
import shell
import textmodel
import ui
import uplift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_emotion_distilbert", "model")
SONGS_FILE = os.path.join(BASE_DIR, "data", "songs", "songs_clean.csv")
# The fusion settings fitted for the face model in use (HSEmotion's own, or the CNN's).
FUSION_CONFIG = face_models.fusion_config(face_models.resolved())
# Emotions whose fused probability is at least this share go into the playlist mix.
MIN_EMOTION_SHARE = 0.15
# Guided webcam scan: a short "get ready", then a short capture. The face model
# reads SCAN_FRAMES frames spread evenly over the capture rather than every frame:
# each frame costs about 33 ms of face detection and each face about 50 ms of
# HSEmotion, so reading all ~30 frames would add several seconds.
GET_READY_SECONDS = 0.5
CAPTURE_SECONDS = 1.0
WARM_UP_FRAMES = 5         # first frames of the capture are dropped (exposure still settling)
MAX_CAPTURE_FRAMES = 120   # safety cap (about 4 s at 30 frames a second)
SCAN_FRAMES = 10
SCAN_IMAGE_WIDTH = 480
BOX_GREEN = (84, 185, 29)  # the accent green, in OpenCV's blue-green-red order
# Auto picks Uplift when the mood is low (uplift.should_uplift), otherwise Resonance.
MODES = ["Auto", "Resonance", "Uplift"]
LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Korean"]
# Telugu and Malayalam have too few playable songs near the Uplift waypoints.
UPLIFT_LANGUAGES = {"English", "Hindi", "Tamil", "Korean"}
# The path card grows to fit its songs (see player.render_path); this is its first height.
PATH_HEIGHT = 900
BAR_HEIGHT = 86
TOP_HEIGHT = 56            # the top bar's two components
NAVLISTS_HEIGHT = 360      # the sidebar's playlists (the page CSS stretches it to fill the sidebar)
LIBRARY_HEIGHT = 640       # first heights; the Library and the Explore list grow to fit
EXPLORE_LIST_HEIGHT = 900
QUICKPICKS_HEIGHT = 360
PHOTO_TYPES = ["jpg", "jpeg", "png"]
# The sidebar's pages: key, label, icon.
PAGES = [("home", "Home", ":material/home:"), ("explore", "Explore", ":material/explore:"),
         ("library", "Library", ":material/library_music:")]
# Explore: the moods (the song buckets) in this order, and each language in its own script.
EXPLORE_MOODS = ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]
LANGUAGE_NAMES = {"English": "English", "Hindi": "हिन्दी", "Tamil": "தமிழ்",
                  "Telugu": "తెలుగు", "Malayalam": "മലയാളം",
                  "Korean": "한국어"}
EXPLORE_TRACKS = 30
QUICK_PICKS = 12           # Home's Quick picks grid (3 columns of 4)
RELATED_SONGS = 12         # the now-playing view's Related tab
# The player bar with nothing to play yet (before a mood is read); songs from
# Explore or the Library can still be played through it.
IDLE_QUEUE = {"id": "idle", "tracks": [], "loop_from": 0, "mode": "Resonance"}

st.set_page_config(page_title=ui.APP_NAME, page_icon="🎧", layout="wide")

# The cached loaders never show a spinner: on a cold start Streamlit would
# otherwise print "Running <function name>()" on the page.
@st.cache_data(show_spinner=False)
def load_songs():
    """Songs with one emotion bucket each, built by scripts/prepare_songs.py (see data/README.md)."""
    return pd.read_csv(SONGS_FILE, keep_default_na=False)


@st.cache_resource(show_spinner=False)
def load_face_model():
    """The face-expression model (face_models.py: HSEmotion by default;
    FACE_MODEL=cnn or ferplus to compare; the original CNN if the model asked
    for can't be loaded). It is loaded after the page appears; TensorFlow is
    only imported for the CNN."""
    return face_models.load()


def load_text_model():
    """DistilBERT text-emotion model, or None if it hasn't been trained on this machine.

    Loaded once per server process by textmodel.load(), which the background
    warm-up (started after the first page is shown) may already have done. If
    the warm-up is still running, this waits for it instead of loading again.
    """
    return textmodel.load(TEXT_MODEL_DIR)


def face_view(face_probs):
    """Face probabilities as the fusion sees them: canonical order, calibrated."""
    return calibrate(align_face(face_probs), FUSION_CONFIG["face_temperature"])


def text_view(text_probs):
    return calibrate(align_text(text_probs), FUSION_CONFIG["text_temperature"])


def scan_picture(frame, box, label):
    """The scan frame as an in-memory JPEG data URI, with the face box and label drawn on it.

    Nothing is written to disk: the picture only lives in the page and the session.
    """
    frame = frame.copy()
    if box is not None:
        x, y, w, h = box
        scale = max(frame.shape[1] / 640, 0.5)
        thick = max(2, round(2 * scale))
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_GREEN, thick)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, thick)
        top = max(y - th - base - 8, 0)
        cv2.rectangle(frame, (x, top), (x + tw + 12, top + th + base + 8), BOX_GREEN, -1)
        cv2.putText(frame, label, (x + 6, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * scale, (0, 0, 0), thick, cv2.LINE_AA)
    if frame.shape[1] > SCAN_IMAGE_WIDTH:
        height = round(frame.shape[0] * SCAN_IMAGE_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (SCAN_IMAGE_WIDTH, height), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode() if ok else None


def summarise_scan(frames):
    """Average the face model over all faces in `frames` and keep one frame to show.

    Returns {"face_probs": averaged softmax or None, "image": data URI or None,
    "face_found": bool, "frames": n, "faces": n}. The frame shown is the one with
    the largest face, or the last frame if no face was seen.
    """
    model = load_face_model()
    face_detector = face.detector()
    crops, shown, shown_box, last = [], None, None, None
    for frame in frames:
        last = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for box in face.find_faces(gray, face_detector):
            # each model builds its own crop of the same box, from the original frame
            crops.append(model.prepare(frame, gray, box))
            if shown_box is None or box[2] * box[3] > shown_box[2] * shown_box[3]:
                shown, shown_box = frame, box
    if not crops:
        return {"face_probs": None, "face_found": False, "frames": len(frames), "faces": 0,
                "image": scan_picture(last, None, "") if last is not None else None}
    face_probs = model.predict(crops).mean(axis=0)
    view = face_view(face_probs)
    label = f"{EMOTIONS[int(view.argmax())]} {view.max():.0%}"
    return {"face_probs": face_probs, "face_found": True, "frames": len(frames), "faces": len(crops),
            "image": scan_picture(shown, shown_box, label)}


def guided_webcam_scan(screen):
    """A short guided scan: "Get ready..." for GET_READY_SECONDS, then capture for CAPTURE_SECONDS.

    The first WARM_UP_FRAMES captured frames are dropped, and at most SCAN_FRAMES of
    the rest, spread evenly, are returned (an empty list if there is no webcam). `screen` is an st.empty() that shows
    the instructions.
    """
    cv2.ocl.setUseOpenCL(False)
    cap = open_camera()
    try:
        if not cap.isOpened():
            return []
        # Countdown. Reading frames meanwhile lets the camera adjust to the light;
        # these frames are not used.
        end, shown = time.time() + GET_READY_SECONDS, None
        while (left := end - time.time()) > 0:
            if math.ceil(left) != shown:
                shown = math.ceil(left)
                screen.markdown(ui.scan_screen("Get ready...", f"Look at the camera &middot; {shown}"), unsafe_allow_html=True)
            if not cap.read()[0]:
                return []
        screen.markdown(ui.scan_screen("Hold your expression...", "Reading your face"),
                        unsafe_allow_html=True)
        frames, end = [], time.time() + CAPTURE_SECONDS
        while time.time() < end and len(frames) < WARM_UP_FRAMES + MAX_CAPTURE_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        kept = frames[WARM_UP_FRAMES:]
        if len(kept) > SCAN_FRAMES:
            kept = [kept[i] for i in np.linspace(0, len(kept) - 1, SCAN_FRAMES).round().astype(int)]
        return kept
    finally:
        cap.release()


def open_camera():
    """The webcam. On Windows DirectShow is tried first: the default backend (Media
    Foundation) took about 6 s to open the camera, DirectShow about 1 s. If DirectShow
    cannot open it, the default backend is used."""
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(0)


def photo_frame(data):
    """An uploaded photo as an OpenCV frame, decoded in memory (or None if it isn't a picture)."""
    return face.load_photo(data)


def text_probabilities(text):
    """Text model softmax output (TEXT_MODEL_LABELS order), or None if unavailable.

    Empty text returns straight away, so the text model (and torch) are only
    loaded once someone actually types something.
    """
    if not text.strip():
        return None
    loaded = load_text_model()
    if loaded is None:
        return None
    import torch
    tokenizer, text_model = loaded
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        return torch.softmax(text_model(**enc).logits, -1)[0].numpy()


def combine_emotions(face_probs, text_probs):
    """Fuse face and text predictions; returns (emotions ranked for the playlist, fused probs).

    If only one of them is available, it is used on its own.
    """
    if face_probs is None and text_probs is None:
        return [], None
    face_weight = FUSION_CONFIG["face_weight"]
    uniform = np.full(len(EMOTIONS), 1 / len(EMOTIONS))
    if text_probs is None:
        face_weight, text_probs = 1.0, uniform
    elif face_probs is None:
        face_weight, face_probs = 0.0, uniform
    _, fused = fuse(face_probs, text_probs, face_weight=face_weight,
                    method=FUSION_CONFIG["method"],
                    face_temperature=FUSION_CONFIG["face_temperature"],
                    text_temperature=FUSION_CONFIG["text_temperature"])
    ranked = sorted(fused, key=fused.get, reverse=True)
    chosen = [e for e in ranked if fused[e] >= MIN_EMOTION_SHARE][:5] or ranked[:1]
    return chosen, fused


def build_queue(songs, emotions, fused, mode_choice, language):
    """Pick the mode and build the play queue for a detected mood.

    Songs come only from `language`; a queue is never padded with another
    language, so it may be shorter than usual.
    """
    note = None
    in_language = (songs.language == language).values
    in_language_text = "" if language == "English" else f" in {language}"
    wants_uplift = mode_choice == "Uplift" or (mode_choice == "Auto" and uplift.should_uplift(fused))
    if wants_uplift and language not in UPLIFT_LANGUAGES:
        note = f"Uplift needs more songs than we have in {language}, so these songs match your mood instead."
    elif wants_uplift:
        tracks, journey = player.uplift_queue(songs, fused, emotions, pool=in_language)
        if tracks:
            steps = sum(track["section"] == "journey" for track in tracks)
            return {"mode": "Uplift", "tracks": tracks, "journey": player.journey_summary(journey),
                    "note": None, "loop_from": steps,
                    "subtitle": f"{steps} songs in order, then more {journey['targets'][-1]} songs{in_language_text}"}
        if tracks is None:
            note = (f"Uplift is for low moods. Sad, angry, fearful and disgusted feelings make up only "
                    f"{uplift.negative_share(fused):.0%} of this result, so these songs match your mood instead.")
        else:
            note = "No playable songs were found along the Uplift path, so these songs match your mood instead."
    buckets = {bucket: rows for bucket, rows in songs[in_language].groupby("bucket")}
    tracks = player.resonance_queue(buckets, emotions)
    return {"mode": "Resonance", "tracks": tracks, "journey": None, "note": note, "loop_from": 0,
            "subtitle": f"{len(tracks)} {' and '.join(emotions)} songs{in_language_text}"}


def requested_mode(mode_choice, fused):
    """The mode the listener asked for: Auto resolves to Uplift for a low mood, otherwise Resonance."""
    if mode_choice != "Auto":
        return mode_choice
    return "Uplift" if uplift.should_uplift(fused) else "Resonance"


def build_view(face_probs, text_probs, fused, queue):
    """What the mood card (gauge, headline, how low each source leans) and the "See all 7 emotions" bars show."""
    order = mood.ranked(fused)
    groups = []
    if face_probs is not None:
        face = dict(zip(EMOTIONS, face_view(face_probs)))
        groups.append({"key": "face", "values": [round(float(face[e]), 4) for e in order]})
    if text_probs is not None:
        text = dict(zip(EMOTIONS, text_view(text_probs)))
        groups.append({"key": "text", "values": [round(float(text[e]), 4) for e in order]})
    groups.append({"key": "combined", "values": [round(float(fused[e]), 4) for e in order]})
    if len(groups) == 3:
        caption = "Combined = face and text fused"
    elif face_probs is not None:
        caption = "Only your face was read, so Combined is the same as Face."
    else:
        caption = "Only your text was read, so Combined is the same as Text."
    # How low each source leans: the share of sad + angry + fear + disgust in its
    # own calibrated probabilities. "combined" is the number Auto mode uses.
    low = {"combined": uplift.negative_share(fused),
           "face": uplift.negative_share(dict(zip(EMOTIONS, face_view(face_probs)))) if face_probs is not None else None,
           "text": uplift.negative_share(dict(zip(EMOTIONS, text_view(text_probs)))) if text_probs is not None else None}
    return {"bars": {"emotions": order, "groups": groups, "caption": caption}, "low": low,
            "card": mood.card(fused, queue["mode"], queue["journey"])}


def scan_note(scan):
    if not scan["face_found"]:
        return ("No face was found in this picture, so your mood comes from your words."
                if scan["image"] else "The webcam gave no picture, so your mood comes from your words.")
    return "From your uploaded photo." if scan["source"] == "photo" else None


NO_FACE_TIP = "Tip: face the light, with the window in front of you, not behind."


def scan_tip(scan):
    """A tip under the note when a picture had no face (not when there was no picture at all)."""
    return NO_FACE_TIP if not scan["face_found"] and scan["image"] else None


def choose(mode):
    st.session_state["mode_choice"] = mode


def want_scan():
    """ "Tap to scan" / "Tap to scan again": open the webcam on this run (and only then)."""
    state["scan_wanted"] = True


def new_scan(scan):
    """Store a new scan (webcam or photo). A new face reading goes back to the
    default mode (Auto unless Settings say otherwise), so a card chosen for the
    old reading doesn't stick; a change of text or language alone keeps the
    chosen card."""
    state["scan"] = scan
    state["scan_id"] = state.get("scan_id", 0) + 1
    state["mode_choice"] = state.get("prefs", {}).get("mode", "Auto")


# Settings > Theme, Default language and Default mode are kept in the browser
# and in this cookie (written by shell.render_profile), so the app can open
# with them. Nothing else from the browser reaches the server.
PREFS_COOKIE = "moodtune_prefs"
DEFAULT_MODES = {"auto": "Auto", "resonance": "Resonance", "uplift": "Uplift"}


def cookie_prefs():
    """{"theme": "light" | "dark", "language": a LANGUAGES name or None, "mode": a MODES name}
    from the prefs cookie (defaults if it is missing or not understood)."""
    try:
        raw = st.context.cookies.get(PREFS_COOKIE) or ""
    except Exception:
        raw = ""
    if not isinstance(raw, str):   # e.g. no browser behind the session (Streamlit's AppTest)
        raw = ""
    values = parse_qs(unquote(raw))
    get = lambda key: (values.get(key) or [""])[0]
    return {"theme": "dark" if get("theme") == "dark" else "light",
            "language": get("lang") if get("lang") in LANGUAGES else None,
            "mode": DEFAULT_MODES.get(get("mode"), "Auto")}


def read_photo(photo):
    """Use a newly uploaded photo as the scan; False if it isn't a picture.

    It is read from memory with the same face detection as the webcam frames,
    and the webcam is not opened for it.
    """
    photo_id = hashlib.md5(photo.getvalue()).hexdigest()
    if state.get("photo_id") == photo_id:
        return True
    state["photo_id"] = photo_id
    frame = photo_frame(photo.getvalue())
    if frame is None:
        return False
    new_scan({**summarise_scan([frame]), "source": "photo"})
    return True


def go(page):
    state["page"] = page


def pick_mood(mood):
    """Explore: pick a mood, or drop it if it's already picked."""
    state["explore_mood"] = None if state.get("explore_mood") == mood else mood


def pick_language(language):
    state["explore_language"] = language


def current_queue(emotions, fused, language, text, face_probs, text_probs):
    """The play queue and mood view for this reading, rebuilt only when the scan,
    the text, the requested mode or the language changes, so other reruns (and
    moving between pages) leave the queue, and the music, alone. Picking the
    card Auto had already chosen doesn't rebuild it."""
    mode_choice = state["mode_choice"]
    wanted = requested_mode(mode_choice, fused)
    queue_key = (state.get("scan_id", 0), text.strip(), wanted, language)
    if state.get("queue_key") != queue_key:
        state["queue"] = {**build_queue(songs, emotions, fused, mode_choice, language), "id": uuid.uuid4().hex,
                          "language": language, "emotion": emotions[0]}
        state["queue"]["related"] = related_songs(language, emotions[0], state["queue"]["tracks"])
        state["view"] = build_view(face_probs, text_probs, fused, state["queue"])
        state["queue_key"] = queue_key
    return state["queue"], state["view"], wanted


def related_songs(language, emotion, tracks):
    """The now-playing view's Related tab: up to RELATED_SONGS playable songs of the queue's
    language and the reading's top emotion that are not in the queue (picked once, with it)."""
    rows = explore_rows(language, emotion)
    rows = rows[~rows.lastfm_url.isin({t["lastfm"] for t in tracks}).values]
    return [player._track(song, preview, section="related", number=n)
            for n, (song, preview) in enumerate(player.playable_songs(rows, RELATED_SONGS, np.random.default_rng()), 1)]


def now_playing_mood(queue):
    """The now-playing view's Mood tab, for the mood queue: the reading's headline, its
    "X% low" band, and the mood card's sentence about the songs (the same words as Home)."""
    view = state.get("view")
    if not queue or queue is not state.get("queue") or not view:
        return None
    share = view["low"]["combined"]
    band, sentence = ui.band(share)
    return {"emoji": view["card"]["emoji"], "headline": view["card"]["headline"], "low": ui.percent(share),
            "band": band, "band_sentence": sentence, "message": view["card"]["message"],
            "emotion": queue.get("emotion", ""), "language": queue.get("language", "")}


def background_queue(language):
    """The queue on Explore and the Library: the last reading (no camera, no
    text model), so the player keeps the same songs; "Songs in" still applies."""
    scan = state.get("scan")
    face_probs = scan["face_probs"] if scan else None
    text = state.get("text", "")
    text_probs = state.get("text_results", {}).get(text.strip())
    emotions, fused = combine_emotions(face_probs, text_probs)
    if fused is None:
        return None
    return current_queue(emotions, fused, language, text, face_probs, text_probs)[0]


def play_chip(mood, language):
    """A mood chip: play that mood's songs in that language (sent once by chips_row). The
    mood queue it replaces is noted, so the chip stops looking selected once that queue
    is rebuilt (a new reading, or another "Songs in")."""
    state["chip_play"] = {"mood": mood, "language": language, "nonce": uuid.uuid4().hex,
                          "queue_key": state.get("queue_key")}


def chips_row(slot, language):
    """Home's row of mood chips, drawn into `slot` (an st.empty at the top of Home). A
    chip replaces the player's queue with up to EXPLORE_TRACKS songs of that mood in the
    "Songs in" language (the same pick as Explore's) and plays the first; picking a song
    of the mood list, or a new reading, brings the mood queue back. A chip looks selected
    only while its songs are the ones it put in the player: nothing is selected on a fresh
    visit. home_page draws the row once per run, after the reading is known, so a new
    reading clears the selection straight away."""
    played = state.get("chip_play")
    selected = (played["mood"] if played and played["language"] == language
                and played["queue_key"] == state.get("queue_key") else None)
    with slot.container():
        with st.container(key="chips", horizontal=True, gap="small"):
            for emotion in EXPLORE_MOODS:
                st.button(f"{mood.EMOJI[emotion]} {emotion.capitalize()}", key=f"chip_mood_{emotion}", on_click=play_chip,
                          args=(emotion, language), type="primary" if emotion == selected else "secondary")
        # The order to play, sent once to the player bar (always in the same place, so nothing after it moves).
        with st.container(key="playcmd"):
            if played:
                picked = explore_list(played["language"], played["mood"])["tracks"]
                in_lang = "" if played["language"] == "English" else f" in {played['language']}"
                label = f"{played['mood'].capitalize()} songs{in_lang}"
                message = (f"Playing {played['mood']} songs{in_lang}" if picked
                           else f"No {played['mood']} songs{in_lang} have a saved preview yet")
                components.html(shell.render_play_command(played["nonce"], picked, message, label), height=0)


def quick_pick_list(language, emotion):
    """QUICK_PICKS songs picked at random from the playable songs of that language (and
    emotion), kept for the session so reruns show the same ones. They avoid Explore's
    first QUICK_PICKS when the pool has enough other songs, so the two differ."""
    lists = state.setdefault("quick_picks", {})
    if (emotion, language) not in lists:
        rows = explore_rows(language, emotion)
        shown = {t["lastfm"] for t in explore_list(language, emotion)["tracks"][:QUICK_PICKS]}
        others = rows[~rows.lastfm_url.isin(shown).values]
        pool = others if len(others) >= QUICK_PICKS else rows
        picked = player.playable_songs(pool, QUICK_PICKS, np.random.default_rng())
        lists[(emotion, language)] = {"id": uuid.uuid4().hex, "tracks": [
            player._track(song, preview, section="mood", number=n) for n, (song, preview) in enumerate(picked, 1)]}
    return lists[(emotion, language)]


def quick_picks(slot, language, emotion=None):
    """Quick picks: QUICK_PICKS songs of the reading's top emotion in "Songs in", or a
    mix of that language before there is a reading."""
    picked = quick_pick_list(language, emotion)
    tracks = picked["tracks"]
    in_lang = "" if language == "English" else f" in {language}"
    if emotion:
        title, subtitle = f"{emotion.capitalize()} songs{in_lang}", "Picked for your mood reading."
    else:
        title, subtitle = f"A mix of songs{in_lang}", "Your picks follow your mood once it has been read."
    with slot.container():
        if tracks:
            components.html(shell.render_quickpicks(picked["id"] + ":qp", title, subtitle, tracks), height=QUICKPICKS_HEIGHT)
        else:
            st.info(f"No songs{in_lang} have a saved preview yet.")


def home_page(language):
    """The chips, the greeting, the "Detect my mood" card (the scan, your words and
    the reading), Quick picks, the choice cards and the songs. Returns the queue,
    or None if there is no reading yet."""
    # The chips are drawn once per run, at the end, once this run's reading is known.
    chips_slot = st.empty()
    local = ui.local_now(st.context.timezone, st.context.timezone_offset)
    st.markdown(ui.greeting_html(local), unsafe_allow_html=True)

    # The "Detect my mood" card: the face scan, the text box, and the reading under them.
    with st.container(key="hero"):
        st.markdown(ui.HERO_HTML, unsafe_allow_html=True)
        face_column, text_column = st.columns(2, gap="medium")
        with face_column, st.container(key="facetile"):
            face_slot = st.empty()
            with st.container(horizontal=True, gap="small"):
                # "Tap to scan" until the webcam has been used this visit (an uploaded photo
                # doesn't open it), "Tap to scan again" after that.
                scanned = state.get("webcam_scans", 0) or state.get("scan_wanted")
                st.button("Tap to scan again" if scanned else "Tap to scan", key="rescan", on_click=want_scan,
                          help="Look at the webcam again" if scanned else "Opens your webcam for a quick scan")
                with st.popover("Upload a photo instead"):
                    tile_photo = st.file_uploader("Upload a photo instead", type=PHOTO_TYPES, key="photo",
                                                  help="A backup if the webcam doesn't work.")
                    st.caption("A photo is read once, in memory, to find a face. It is never saved.")
        with text_column, st.container(key="texttile"):
            st.markdown('<p class="tile-label">Your words</p>', unsafe_allow_html=True)
            # Streamlit forgets a widget's value on a run where it isn't drawn (another
            # page), so the text is kept in state["text"] and put back when Home returns.
            if "user_text" not in state:
                state["user_text"] = state.get("text", "")
            # A form, so the text is read on "Check my mood" (or Enter), not on every keystroke.
            with st.form("textform", border=False):
                user_text = st.text_input("How was your day?", key="user_text", max_chars=180,
                                          placeholder="e.g. Finally finished my exams!")
                st.form_submit_button("Check my mood", type="primary")
            state["text"] = user_text
            reading_slot = st.empty()
            st.markdown('<p class="tile-foot">Up to 180 characters. Your words are read on this computer and kept only '
                        'for this visit.</p>', unsafe_allow_html=True)
        result_slot = st.empty()
        result_slot.markdown(ui.RESULT_WAIT_HTML, unsafe_allow_html=True)
    notices = st.container(key="notices")
    with st.container(key="quickpicks"):
        picks_slot = st.empty()
    if "scan" not in state:
        quick_picks(picks_slot, language)   # a default set until there is a reading

    if tile_photo is not None and not read_photo(tile_photo):
        notices.warning("That file couldn't be read as a picture. Try a JPEG or PNG.")

    # The webcam is opened only on "Tap to scan" / "Tap to scan again", never on its
    # own; anything else reuses the cached scan (face result and picture, both kept in
    # memory only). Before the first scan the tile is idle and the camera is off.
    if state.get("scan_wanted"):
        # The face model loads first, so the countdown and capture are not held up by it.
        face_slot.markdown(ui.scan_screen("Listening to your mood...", "Starting the camera"), unsafe_allow_html=True)
        load_face_model()
        frames = guided_webcam_scan(face_slot)
        face_slot.markdown(ui.scan_screen("Listening to your mood...", "Reading your expression"), unsafe_allow_html=True)
        new_scan({**summarise_scan(frames), "source": "webcam"})
        state["webcam_scans"] = state.get("webcam_scans", 0) + 1
        # Cleared only now: if this run is interrupted mid-scan (say Enter pressed in the
        # text box), the next run scans again instead of losing the tap.
        state["scan_wanted"] = False
    scan = state.get("scan")
    if scan is None:
        face_slot.markdown(ui.IDLE_TILE_HTML, unsafe_allow_html=True)
    else:
        face_slot.markdown(ui.face_tile_html(scan["image"], scan_note(scan), scan_tip(scan), model_name=load_face_model().name),
                           unsafe_allow_html=True)
    # No scan, or a scan that found no face: the reading comes from the words alone
    # (combine_emotions with face_probs None, the same path in both cases).
    face_probs = scan["face_probs"] if scan else None

    # Each distinct text is read once per session. The first read also loads the
    # text model (unless the background warm-up already has), so a friendly
    # message stands in while it happens.
    text_results = state.setdefault("text_results", {})
    if user_text.strip() and user_text.strip() not in text_results:
        reading_slot.markdown(ui.READING_HTML, unsafe_allow_html=True)
        text_results[user_text.strip()] = text_probabilities(user_text)
    text_probs = text_results.get(user_text.strip())
    if user_text.strip() and text_probs is None:
        reading_slot.caption("The text model isn't available on this machine, so only your face was read.")
    else:
        reading_slot.empty()
    emotions, fused = combine_emotions(face_probs, text_probs)

    if fused is None:
        if scan is not None:   # a scan that found no face; before any scan the card's placeholder says what to do
            notices.warning("I couldn't see a face. Tap to scan again, upload a photo, or tell me how your day was.")
        quick_picks(picks_slot, language)
        chips_row(chips_slot, language)
        return None

    queue, view, wanted = current_queue(emotions, fused, language, user_text, face_probs, text_probs)
    mode_choice = state["mode_choice"]

    if previews.rate_limited():
        notices.caption("The preview service is limiting requests right now, so some songs could not be checked. "
                        "Scan again in a minute for a fuller list.")

    # The mood reading, in the card under the scan and the words: "X% low" (the
    # share Auto mode uses), and the same for face and text.
    with result_slot.container(key="moodcard"):
        st.markdown(ui.mood_card_html(view["card"], view["low"]), unsafe_allow_html=True)
        with st.expander("See all 7 emotions"):
            st.markdown(ui.bars_html(view["bars"]), unsafe_allow_html=True)
    quick_picks(picks_slot, language, emotions[0])

    # What do you need from music? Auto stays the default, and the card it
    # would choose is tagged "Recommended".
    recommended = requested_mode("Auto", fused)
    with st.container(key="choice"):
        st.markdown(ui.CHOICE_HTML, unsafe_allow_html=True)
        for column, mode in zip(st.columns(2, gap="medium"), ["Uplift", "Resonance"]):
            fallback = None
            if mode == "Uplift" and language not in UPLIFT_LANGUAGES:
                fallback = f"Uplift needs more songs than we have in {language}, so these songs match your mood instead."
            with column, st.container(key=f"choice_{mode.lower()}"):
                st.markdown(ui.choice_card_html(mode, wanted == mode, recommended == mode, fallback),
                            unsafe_allow_html=True)
                st.button(ui.CHOICES[mode][2], key=f"choose_{mode.lower()}", on_click=choose, args=(mode,),
                          use_container_width=True)
        if mode_choice != "Auto":
            st.button("Let MOOD TUNE choose", key="choose_auto", on_click=choose, args=("Auto",))

    # The path card (Uplift) or the songs for this moment (Resonance).
    with st.container(key="path"):
        if queue["tracks"]:
            components.html(player.render_path(queue, view["card"]["message"]), height=PATH_HEIGHT)
        else:
            # Cache-only by default: only songs already in data/songs/previews.csv can be played.
            st.info(f"No {language} songs with a saved preview match this mood yet. Try another language.")
    chips_row(chips_slot, language)
    return queue


def explore_rows(language, mood=None):
    """Explore: the songs of one language (and mood), only those with a saved preview
    unless live lookups are on."""
    rows = songs[(songs.language == language).values & ((songs.bucket == mood).values if mood else True)]
    if not previews.LIVE_LOOKUPS:
        rows = rows[rows.lastfm_url.isin(previews.playable_urls()).values]
    return rows


def explore_list(language, mood):
    """A random pick of EXPLORE_TRACKS playable songs, kept for the session so reruns don't reshuffle it."""
    lists = state.setdefault("explore_lists", {})
    if (mood, language) not in lists:
        picked = player.playable_songs(explore_rows(language, mood), EXPLORE_TRACKS, np.random.default_rng())
        lists[(mood, language)] = {"id": uuid.uuid4().hex, "tracks": [
            player._track(song, preview, section="mood", number=n) for n, (song, preview) in enumerate(picked, 1)]}
    return lists[(mood, language)]


def explore_page(language):
    """Browse by mood (the seven song buckets) and by language; the songs picked play through the player bar."""
    st.markdown(ui.page_head_html("Explore", "Browse the songs by mood or by language. A song you play here "
                                  "starts straight away, and the queue carries on after it."), unsafe_allow_html=True)
    # The language follows "Songs in" until one is picked here.
    lang = state.get("explore_language") or language
    mood = state.get("explore_mood")
    with st.container(key="explore"):
        st.markdown(ui.section_head_html("Moods", f"Songs with a preview in {lang}"), unsafe_allow_html=True)
        with st.container(key="moodcards"):
            for emotion in EXPLORE_MOODS:
                with st.container(key=f"xcard_mood_{emotion}"):
                    st.markdown(ui.mood_tile_html(emotion, len(explore_rows(lang, emotion)), mood == emotion),
                                unsafe_allow_html=True)
                    st.button(f"{emotion.capitalize()} songs", key=f"xmood_{emotion}", on_click=pick_mood, args=(emotion,))
        st.markdown(ui.section_head_html("Languages", "“Lift my mood” needs more songs than Telugu and Malayalam "
                                         "have, so those two offer “Meet me here” only."), unsafe_allow_html=True)
        with st.container(key="langcards"):
            for name in LANGUAGES:
                with st.container(key=f"xcard_lang_{name.lower()}"):
                    st.markdown(ui.language_tile_html(name, LANGUAGE_NAMES[name], len(explore_rows(name)),
                                                      name in UPLIFT_LANGUAGES, lang == name), unsafe_allow_html=True)
                    st.button(f"Songs in {name}", key=f"xlang_{name.lower()}", on_click=pick_language, args=(name,))
    picked = explore_list(lang, mood)
    in_lang = "" if lang == "English" else f" in {lang}"
    title = f"{mood.capitalize()} songs{in_lang}" if mood else f"Songs in {lang}"
    pool = len(explore_rows(lang, mood))
    shown = len(picked["tracks"])
    subtitle = ((f"All {shown} songs with a preview, in random order." if shown >= pool else
                 f"{shown} of the {pool:,} songs with a preview, picked at random.")
                + ("" if mood else " Pick a mood above to narrow it down."))
    with st.container(key="explorelist"):
        if picked["tracks"]:
            components.html(shell.render_songlist(picked["id"], title, subtitle, picked["tracks"]),
                            height=EXPLORE_LIST_HEIGHT)
        else:
            st.info(f"No {mood + ' ' if mood else ''}songs{in_lang} have a saved preview yet. Try another mood or language.")


def library_page():
    st.markdown(ui.page_head_html("Library", "Your liked songs, your playlists and what you played recently. They "
                                  "are kept in this browser only and never sent anywhere."), unsafe_allow_html=True)
    with st.container(key="library"):
        components.html(shell.render_library(), height=LIBRARY_HEIGHT)


state = st.session_state
# Read once per visit: the theme for the first paint, and the defaults for "Songs in" and the mode.
if "prefs" not in state:
    state["prefs"] = cookie_prefs()
    if state["prefs"]["language"]:
        state["language"] = state["prefs"]["language"]
st.markdown(ui.page_css(dark_first=state["prefs"]["theme"] == "dark"), unsafe_allow_html=True)

if not os.path.exists(SONGS_FILE):
    st.error("Song file not found. Build it once with:  python scripts/prepare_songs.py")
    st.stop()
songs = load_songs()
state.setdefault("mode_choice", state["prefs"]["mode"])
state.setdefault("page", "home")
page = state["page"]

# The page is laid out in keyed containers, created in the same order on every
# run and on every page, so the shell's components and the player bar keep
# their place (and the music keeps playing) across reruns and page changes.

# 1. Top bar: the menu button, logo and search (a component), "Songs in", and
#    the avatar with its menu (a component). Both components never reload.
with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="small"):
    with st.container(key="topleft"):
        components.html(shell.render_topbar(), height=TOP_HEIGHT)
    language = st.selectbox("Songs in", LANGUAGES, key="language")
    with st.container(key="topactions"):
        components.html(shell.render_profile(), height=TOP_HEIGHT)

# 2. Sidebar: Home, Explore and Library, then New playlist and your playlists.
with st.container(key="nav"):
    for key, label, icon in PAGES:
        st.button(label, key=f"nav_{key}", icon=icon, on_click=go, args=(key,),
                  type="primary" if page == key else "tertiary", use_container_width=True)
    with st.container(key="navlists"):
        components.html(shell.render_navlists(), height=NAVLISTS_HEIGHT)

# 3. The player bar's place, pinned to the bottom of the window by the page CSS
#    (filled below, once the queue is known).
bar_slot = st.container(key="playerbar")

# 4. The page.
queue = home_page(language) if page == "home" else background_queue(language)
# With no reading right now (no face and no words, e.g. after "Tap to scan again"
# found no face), the player keeps the last mood queue, and its music, until a
# new reading replaces it.
player_queue = queue if queue is not None else state.get("queue")
with bar_slot:
    bar_queue = player_queue if player_queue and player_queue["tracks"] else IDLE_QUEUE
    components.html(player.render_bar(bar_queue, mood=now_playing_mood(bar_queue), related=bar_queue.get("related")),
                    height=BAR_HEIGHT)
if page == "explore":
    explore_page(language)
elif page == "library":
    library_page()

# Once a page with a reading is up, load the text model in the background
# (once per server process), so it is usually ready before anyone types. The
# thread only imports and loads the model; it never touches Streamlit.
if queue is not None or page != "home":
    textmodel.warm_up(TEXT_MODEL_DIR)
