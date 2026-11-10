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

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from fusion import EMOTIONS, align_face, align_text, calibrate, fuse
import face
import mood
import player
import previews
import textmodel
import ui
import uplift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_emotion_distilbert", "model")
SONGS_FILE = os.path.join(BASE_DIR, "data", "songs", "songs_clean.csv")
with open(os.path.join(BASE_DIR, "models", "fusion", "fusion_config.json")) as f:
    FUSION_CONFIG = json.load(f)
# Emotions whose fused probability is at least this share go into the playlist mix.
MIN_EMOTION_SHARE = 0.15
# Guided webcam scan: a short countdown, then a few seconds of capture.
GET_READY_SECONDS = 2
CAPTURE_SECONDS = 3
WARM_UP_FRAMES = 5         # first frames of the capture are dropped (exposure still settling)
MAX_CAPTURE_FRAMES = 120   # safety cap (about 4 s at 30 frames a second)
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
ACTIONS_HEIGHT = 50
PHOTO_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(page_title=ui.APP_NAME, page_icon="🎧", layout="wide")

# The cached loaders never show a spinner: on a cold start Streamlit would
# otherwise print "Running <function name>()" on the page.
@st.cache_data(show_spinner=False)
def load_songs():
    """Songs with one emotion bucket each, built by scripts/prepare_songs.py (see data/README.md)."""
    return pd.read_csv(SONGS_FILE, keep_default_na=False)


@st.cache_resource(show_spinner=False)
def load_face_model():
    # TensorFlow is only imported inside face.build_model(), so the page can
    # appear (with the scan screen) before it has finished loading.
    return face.build_model(os.path.join(BASE_DIR, 'model.h5'))


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
            crops.append(face.crop(gray, box))
            if shown_box is None or box[2] * box[3] > shown_box[2] * shown_box[3]:
                shown, shown_box = frame, box
    if not crops:
        return {"face_probs": None, "face_found": False, "frames": len(frames), "faces": 0,
                "image": scan_picture(last, None, "") if last is not None else None}
    face_probs = face.predict(model, crops).mean(axis=0)
    view = face_view(face_probs)
    label = f"{EMOTIONS[int(view.argmax())]} {view.max():.0%}"
    return {"face_probs": face_probs, "face_found": True, "frames": len(frames), "faces": len(crops),
            "image": scan_picture(shown, shown_box, label)}


def guided_webcam_scan(screen):
    """A short guided scan: "Get ready..." for GET_READY_SECONDS, then capture for CAPTURE_SECONDS.

    The first WARM_UP_FRAMES captured frames are dropped. Returns the frames kept
    (an empty list if there is no webcam). `screen` is an st.empty() that shows
    the instructions.
    """
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
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
        screen.markdown(ui.scan_screen("Hold your expression...", "Reading your face for a few seconds"),
                        unsafe_allow_html=True)
        frames, end = [], time.time() + CAPTURE_SECONDS
        while time.time() < end and len(frames) < WARM_UP_FRAMES + MAX_CAPTURE_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames[WARM_UP_FRAMES:]
    finally:
        cap.release()


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


def choose(mode):
    st.session_state["mode_choice"] = mode


def new_scan(scan):
    """Store a new scan (webcam or photo). A new face reading goes back to Auto,
    so a card chosen for the old reading doesn't stick; a change of text or
    language alone keeps the chosen card."""
    state["scan"] = scan
    state["scan_id"] = state.get("scan_id", 0) + 1
    state["mode_choice"] = "Auto"


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


st.markdown(ui.PAGE_CSS, unsafe_allow_html=True)

if not os.path.exists(SONGS_FILE):
    st.error("Song file not found. Build it once with:  python scripts/prepare_songs.py")
    st.stop()
songs = load_songs()
state = st.session_state
state.setdefault("mode_choice", "Auto")

# The page is laid out in keyed containers, created in the same order on every
# run, so the two player components keep their place (and the music keeps
# playing) across reruns that don't change the queue.

# 1. Top bar and greeting.
with st.container(key="topbar"):
    brand, controls = st.columns([1, 1], vertical_alignment="center")
    brand.markdown(ui.topbar_html(), unsafe_allow_html=True)
    with controls, st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                                wrap=False, key="topcontrols"):
        language = st.selectbox("Songs in", LANGUAGES, key="language", width=250)
        # My playlist and Settings (and their drawer): a browser-only component that never reloads.
        with st.container(key="topactions"):
            components.html(player.render_actions(), height=ACTIONS_HEIGHT)
local = ui.local_now(st.context.timezone, st.context.timezone_offset)
st.markdown(ui.greeting_html(local), unsafe_allow_html=True)

# 2. Hero card: the face scan and the text box.
with st.container(key="hero"):
    st.markdown(ui.HERO_HTML, unsafe_allow_html=True)
    face_column, text_column = st.columns(2, gap="medium")
    with face_column, st.container(key="facetile"):
        face_slot = st.empty()
        with st.container(horizontal=True, gap="small"):
            if st.button("Tap to scan again", key="rescan", help="Look at the webcam again"):
                state.pop("scan", None)
            with st.popover("Upload a photo instead"):
                tile_photo = st.file_uploader("Upload a photo instead", type=PHOTO_TYPES, key="photo",
                                              help="A backup if the webcam doesn't work.")
                st.caption("A photo is read once, in memory, to find a face. It is never saved.")
    with text_column, st.container(key="texttile"):
        st.markdown('<p class="tile-label">Your words</p>', unsafe_allow_html=True)
        # A form, so the text is read on "Check my mood" (or Enter), not on every keystroke.
        with st.form("textform", border=False):
            user_text = st.text_input("How was your day?", key="user_text", max_chars=180,
                                      placeholder="e.g. Finally finished my exams!")
            st.form_submit_button("Check my mood", type="primary")
        reading_slot = st.empty()
        st.markdown('<p class="tile-foot">Up to 180 characters. Your words are read on this computer and kept only '
                    'for this visit.</p>', unsafe_allow_html=True)
notices = st.container(key="notices")

if tile_photo is not None and not read_photo(tile_photo):
    notices.warning("That file couldn't be read as a picture. Try a JPEG or PNG.")

# The webcam is only opened on first load and on "Tap to scan again"; anything
# else reuses the cached scan (face result and picture, both kept in memory only).
if "scan" not in state:
    # The face model loads first, so the countdown and capture are not held up by it.
    face_slot.markdown(ui.scan_screen("Listening to your mood...", "Starting the camera"), unsafe_allow_html=True)
    load_face_model()
    frames = guided_webcam_scan(face_slot)
    face_slot.markdown(ui.scan_screen("Listening to your mood...", "Reading your expression"), unsafe_allow_html=True)
    new_scan({**summarise_scan(frames), "source": "webcam"})
scan = state["scan"]
face_slot.markdown(ui.face_tile_html(scan["image"], scan_note(scan)), unsafe_allow_html=True)
face_probs = scan["face_probs"]

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
    notices.warning("I couldn't see a face. Tap to scan again, upload a photo, or tell me how your day was.")
    st.stop()

# The queue is rebuilt only when the scan, the text, the requested mode or the
# language changes, so other reruns leave the page (and the music) alone.
# Picking the card Auto had already chosen doesn't rebuild it.
mode_choice = state["mode_choice"]
wanted = requested_mode(mode_choice, fused)
queue_key = (state["scan_id"], user_text.strip(), wanted, language)
if state.get("queue_key") != queue_key:
    state["queue"] = {**build_queue(songs, emotions, fused, mode_choice, language), "id": uuid.uuid4().hex}
    state["view"] = build_view(face_probs, text_probs, fused, state["queue"])
    state["queue_key"] = queue_key
queue, view = state["queue"], state["view"]

if previews.rate_limited():
    notices.caption("The preview service is limiting requests right now, so some songs could not be checked. "
                    "Scan again in a minute for a fuller list.")

# 3. Mood reading: "X% low" (the share Auto mode uses), and the same for face and text.
with st.container(key="moodcard"):
    st.markdown(ui.mood_card_html(view["card"], view["low"]), unsafe_allow_html=True)
    with st.expander("See all 7 emotions"):
        st.markdown(ui.bars_html(view["bars"]), unsafe_allow_html=True)

# 4. What do you need from music? The cards replace the old mode radio: Auto
# stays the default, and the card it would choose is tagged "Recommended".
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

# 5. The path card (Uplift) or the songs for this moment (Resonance), and
# 6. the player bar, pinned to the bottom of the window by the page CSS.
with st.container(key="path"):
    if queue["tracks"]:
        components.html(player.render_path(queue, view["card"]["message"]), height=PATH_HEIGHT)
    else:
        # Cache-only by default: only songs already in data/songs/previews.csv can be played.
        st.info(f"No {language} songs with a saved preview match this mood yet. Try another language.")
if queue["tracks"]:
    with st.container(key="playerbar"):
        components.html(player.render_bar(queue), height=BAR_HEIGHT)

# Now that the page is up, load the text model in the background (once per
# server process), so it is usually ready before anyone types. The thread
# only imports and loads the model; it never touches Streamlit.
textmodel.warm_up(TEXT_MODEL_DIR)
