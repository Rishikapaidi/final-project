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
import uplift

APP_NAME = "MOOD TUNE"
TAGLINE = "Music that meets your mood"
# The app's own mark: four sound-wave bars in the accent green.
LOGO_SVG = ('<svg class="logo" viewBox="0 0 40 40" width="40" height="40" role="img" aria-label="logo">'
            '<rect width="40" height="40" rx="10" fill="#1f1f1f"/>'
            '<rect x="8" y="16" width="4.5" height="8" rx="2.25" fill="#1db954"/>'
            '<rect x="15" y="9" width="4.5" height="22" rx="2.25" fill="#1db954"/>'
            '<rect x="22" y="13" width="4.5" height="14" rx="2.25" fill="#1db954"/>'
            '<rect x="29" y="18" width="4.5" height="4" rx="2.25" fill="#1db954"/></svg>')

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
MODES = ["Auto", "Resonance", "Uplift"]
PLAYER_HEIGHT = 680

st.set_page_config(page_title=APP_NAME, page_icon="🎧", layout="wide")

PAGE_CSS = """
<style>
  #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
  .stApp { background: #121212; }
  .block-container { padding: 1.1rem 1.6rem 0 1.6rem; max-width: 1500px; }
  iframe { height: calc(100vh - 104px) !important; min-height: 420px; border-radius: 10px; }
  .app-head { display: flex; align-items: center; gap: 13px; }
  .app-head .logo { flex: none; }
  .app-name { font-size: 23px; font-weight: 800; color: #fff; letter-spacing: 0.08em; line-height: 1.1; }
  .app-tagline { font-size: 13px; color: #9a9a9a; margin-top: 2px; }
  .listening { display: flex; flex-direction: column; align-items: center; justify-content: center;
               gap: 22px; height: 60vh; color: #b3b3b3; font-size: 18px; }
  .pulse { width: 74px; height: 74px; border-radius: 50%; background: #1db954; animation: pulse 1.3s ease-in-out infinite; }
  .listening-detail { font-size: 14px; color: #8a8a8a; margin-top: -12px; }
  @keyframes pulse { 0%, 100% { transform: scale(0.75); opacity: 0.55; } 50% { transform: scale(1.1); opacity: 1; } }
  .reading { position: fixed; top: 18px; left: 50%; transform: translateX(-50%); z-index: 1000;
             display: inline-flex; align-items: center; gap: 10px; color: #b3b3b3; font-size: 14px;
             background: #1b1b1b; border: 1px solid #2c2c2c; border-radius: 16px; padding: 7px 16px; box-shadow: 0 4px 14px rgba(0,0,0,.5); }
  .reading .dot { width: 10px; height: 10px; border-radius: 50%; background: #1db954; animation: pulse 1.1s ease-in-out infinite; }
</style>
"""
HEADER_HTML = (f'<div class="app-head">{LOGO_SVG}<div><div class="app-name">{APP_NAME}</div>'
               f'<div class="app-tagline">{TAGLINE}</div></div></div>')
READING_HTML = '<div class="reading"><span class="dot"></span>Reading your text...</div>'


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


def scan_screen(title, detail=""):
    return (f'<div class="listening"><div class="pulse"></div><div>{title}</div>'
            f'<div class="listening-detail">{detail}</div></div>')


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
                screen.markdown(scan_screen("Get ready...", f"Look at the camera &middot; {shown}"), unsafe_allow_html=True)
            if not cap.read()[0]:
                return []
        screen.markdown(scan_screen("Hold your expression...", "Reading your face for a few seconds"),
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


def build_queue(songs, emotions, fused, mode_choice):
    """Pick the mode and build the play queue for a detected mood."""
    note = None
    if mode_choice == "Uplift" or (mode_choice == "Auto" and uplift.should_uplift(fused)):
        tracks, journey = player.uplift_queue(songs, fused, emotions)
        if tracks:
            steps = sum(track["section"] == "journey" for track in tracks)
            return {"mode": "Uplift", "tracks": tracks, "journey": player.journey_summary(journey),
                    "note": None, "loop_from": steps,
                    "subtitle": f"{steps} songs in order, then more {journey['targets'][-1]} songs"}
        if tracks is None:
            note = (f"Uplift is for low moods. Sad, angry, fearful and disgusted feelings make up only "
                    f"{uplift.negative_share(fused):.0%} of this result, so these songs match your mood instead.")
        else:
            note = "No playable songs were found along the Uplift path, so these songs match your mood instead."
    buckets = {bucket: rows for bucket, rows in songs.groupby("bucket")}
    tracks = player.resonance_queue(buckets, emotions)
    return {"mode": "Resonance", "tracks": tracks, "journey": None, "note": note, "loop_from": 0,
            "subtitle": f"{len(tracks)} {' and '.join(emotions)} songs"}


def build_view(scan, face_probs, text_probs, fused, queue, user_text):
    """What the page's panels show: the scan picture, the emotion bars and the mood card."""
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

    scan_note = "From your uploaded photo." if scan["source"] == "photo" else None
    if not scan["face_found"]:
        scan_note = ("No face was found in this picture, so your mood came from the text."
                     if scan["image"] else "The webcam gave no picture, so your mood came from the text.")
    elif user_text.strip() and text_probs is None:
        scan_note = "The text model isn't available on this machine, so only your face was read."
    return {"scan": {"image": scan["image"], "note": scan_note},
            "bars": {"emotions": order, "groups": groups, "caption": caption},
            "card": mood.card(fused, queue["mode"], queue["journey"]),
            "note": queue["note"]}


st.markdown(PAGE_CSS, unsafe_allow_html=True)

if not os.path.exists(SONGS_FILE):
    st.error("Song file not found. Build it once with:  python scripts/prepare_songs.py")
    st.stop()
songs = load_songs()
state = st.session_state

head, rescan_slot, settings_slot = st.columns([8, 1.1, 0.7], vertical_alignment="center")
head.markdown(HEADER_HTML, unsafe_allow_html=True)
with settings_slot:
    with st.popover("⚙", help="Settings"):
        mode_choice = st.radio("Mode", MODES, key="mode_choice",
                               captions=["Uplift when the mood is low, otherwise Resonance",
                                         "Songs that match my mood", "Gently lift my mood"])
        user_text = st.text_input("Tell me how you feel (optional)", key="user_text",
                                  placeholder="e.g. Finally finished my exams!")
        photo = st.file_uploader("Upload a photo instead", type=["jpg", "jpeg", "png"], key="photo",
                                 help="A backup if the webcam doesn't work. The photo is read once and not saved.")
with rescan_slot:
    if st.button("Re-scan", help="Look at the webcam again"):
        state.pop("scan", None)

# A newly uploaded photo replaces the scan. It is read from memory with the same
# face detection as the webcam frames, and the webcam is not opened for it.
if photo is not None:
    photo_id = hashlib.md5(photo.getvalue()).hexdigest()
    if state.get("photo_id") != photo_id:
        state["photo_id"] = photo_id
        frame = photo_frame(photo.getvalue())
        if frame is None:
            st.warning("That file couldn't be read as a picture. Try a JPEG or PNG.")
        else:
            state["scan"] = {**summarise_scan([frame]), "source": "photo"}
            state["scan_id"] = state.get("scan_id", 0) + 1

# The webcam is only opened on first load and on Re-scan; changing a setting
# reuses the cached scan (face result and picture, both kept in memory only).
if "scan" not in state:
    listening = st.empty()
    # The face model loads first, so the countdown and capture are not held up by it.
    listening.markdown(scan_screen("Listening to your mood...", "Starting the camera"), unsafe_allow_html=True)
    load_face_model()
    frames = guided_webcam_scan(listening)
    listening.markdown(scan_screen("Listening to your mood...", "Reading your expression"), unsafe_allow_html=True)
    state["scan"] = {**summarise_scan(frames), "source": "webcam"}
    state["scan_id"] = state.get("scan_id", 0) + 1
    listening.empty()

scan = state["scan"]
face_probs = scan["face_probs"]
# Each distinct text is read once per session. The first read also loads the
# text model, so a friendly message stands in while it happens.
text_results = state.setdefault("text_results", {})
if user_text.strip() and user_text.strip() not in text_results:
    reading = st.empty()
    reading.markdown(READING_HTML, unsafe_allow_html=True)
    text_results[user_text.strip()] = text_probabilities(user_text)
    reading.empty()
text_probs = text_results.get(user_text.strip())
emotions, fused = combine_emotions(face_probs, text_probs)

if fused is None:
    st.warning("I couldn't see a face. Press Re-scan, or open the ⚙ settings to tell me how you feel "
               "or upload a photo.")
    st.stop()

# The queue is rebuilt only when the scan, the text or the mode changes, so
# other reruns leave the page (and the music) alone.
queue_key = (state["scan_id"], user_text.strip(), mode_choice)
if state.get("queue_key") != queue_key:
    state["queue"] = build_queue(songs, emotions, fused, mode_choice)
    state["view"] = build_view(scan, face_probs, text_probs, fused, state["queue"], user_text)
    state["queue_key"] = queue_key
queue, view = state["queue"], state["view"]

if previews.rate_limited():
    st.caption("The preview service is limiting requests right now, so some songs could not be checked. "
               "Re-scan in a minute for a fuller list.")
if not queue["tracks"]:
    # Cache-only by default: only songs already in data/songs/previews.csv can be played.
    st.info("No songs with a saved preview match this mood yet. Fill the preview cache with "
            "`python scripts/fetch_previews.py`, or start the app with the environment variable "
            "`LIVE_LOOKUPS=1` to look songs up as you go.")
    st.stop()

# When the queue ends it loops; after an Uplift journey it loops over the
# "keep listening" songs rather than replaying the journey.
components.html(player.render(queue["tracks"], queue["subtitle"], queue["journey"],
                              loop_from=queue["loop_from"], view=view), height=PLAYER_HEIGHT)

# Now that the page is up, load the text model in the background (once per
# server process), so it is usually ready before anyone types. The thread
# only imports and loads the model; it never touches Streamlit.
textmodel.warm_up(TEXT_MODEL_DIR)
