"""Emotion-based music player.

Detects the listener's mood from the webcam (and optional text), fuses the two
predictions, and plays matching songs (Resonance) or a short playlist that
moves towards a better mood (Uplift). See UPLIFT.md, models/README.md and
data/README.md for how each part works.
"""
import html
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from fusion import EMOTIONS, fuse
import player
import previews
import uplift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_emotion_distilbert", "model")
SONGS_FILE = os.path.join(BASE_DIR, "data", "songs", "songs_clean.csv")
with open(os.path.join(BASE_DIR, "models", "fusion", "fusion_config.json")) as f:
    FUSION_CONFIG = json.load(f)
# Emotions whose fused probability is at least this share go into the playlist mix.
MIN_EMOTION_SHARE = 0.15
FRAMES_TO_SCAN = 20
MODES = ["Auto", "Resonance", "Uplift"]
PLAYER_HEIGHT = 680

st.set_page_config(page_title="Emotion Music Player", page_icon="🎧", layout="wide")

PAGE_CSS = """
<style>
  #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
  .stApp { background: #121212; }
  .block-container { padding: 1.1rem 1.6rem 0 1.6rem; max-width: 1200px; }
  iframe { height: calc(100vh - 104px) !important; min-height: 420px; border-radius: 10px; }
  .app-head { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .app-name { font-size: 24px; font-weight: 700; color: #fff; }
  .chip { background: #1f1f1f; border: 1px solid #333; border-radius: 16px; padding: 5px 13px;
          font-size: 13.5px; color: #e6e6e6; white-space: nowrap; }
  .chip b { color: #1db954; font-weight: 600; }
  .chip-source { color: #8a8a8a; font-size: 12px; }
  .listening { display: flex; flex-direction: column; align-items: center; justify-content: center;
               gap: 22px; height: 60vh; color: #b3b3b3; font-size: 18px; }
  .pulse { width: 74px; height: 74px; border-radius: 50%; background: #1db954; animation: pulse 1.3s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { transform: scale(0.75); opacity: 0.55; } 50% { transform: scale(1.1); opacity: 1; } }
  .notice { color: #b3b3b3; font-size: 13px; margin: 2px 0 8px; }
</style>
"""
LISTENING_HTML = '<div class="listening"><div class="pulse"></div><div>Listening to your mood...</div></div>'


@st.cache_data
def load_songs():
    """Songs with one emotion bucket each, built by scripts/prepare_songs.py (see data/README.md)."""
    return pd.read_csv(SONGS_FILE, keep_default_na=False)


@st.cache_resource
def load_face_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights(os.path.join(BASE_DIR, 'model.h5'))
    return model


@st.cache_resource
def load_text_model():
    """DistilBERT text-emotion model, or None if it hasn't been trained on this machine."""
    if not os.path.isdir(TEXT_MODEL_DIR):
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR).eval()
    return tokenizer, text_model


def scan_face():
    """Face model softmax output averaged over webcam frames with a face, or None if no face was seen."""
    model = load_face_model()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(0)
    frame_probs = []
    for _ in range(FRAMES_TO_SCAN):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5):
            # The model was trained on pixels scaled to 0-1, not raw 0-255.
            crop = cv2.resize(gray[y:y + h, x:x + w], (48, 48))[None, :, :, None].astype('float32') / 255.0
            frame_probs.append(model.predict(crop, verbose=0)[0])
    cap.release()
    return np.mean(frame_probs, axis=0) if frame_probs else None


def text_probabilities(text):
    """Text model softmax output (TEXT_MODEL_LABELS order), or None if unavailable."""
    loaded = load_text_model()
    if loaded is None or not text.strip():
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
            path = " → ".join([journey["start_emotion"]] + journey["targets"])
            return {"mode": "Uplift", "tracks": tracks, "journey": player.journey_summary(journey),
                    "title": "Lifting your mood", "subtitle": f"{path} · {len(tracks)} songs, in order", "note": None}
        if tracks is None:
            note = (f"Uplift is for low moods. Sad, angry, fearful and disgusted feelings make up only "
                    f"{uplift.negative_share(fused):.0%} of this result, so here are songs that match your mood instead.")
        else:
            note = "No playable songs were found along the Uplift path right now, so here are songs that match your mood instead."
    buckets = {bucket: rows for bucket, rows in songs.groupby("bucket")}
    tracks = player.resonance_queue(buckets, emotions)
    return {"mode": "Resonance", "tracks": tracks, "journey": None, "title": "Songs for your mood",
            "subtitle": f"{', '.join(emotions)} · {len(tracks)} songs", "note": note}


st.markdown(PAGE_CSS, unsafe_allow_html=True)

if not os.path.exists(SONGS_FILE):
    st.error("Song file not found. Build it once with:  python scripts/prepare_songs.py")
    st.stop()
songs = load_songs()
state = st.session_state

head, rescan_slot, settings_slot = st.columns([8, 1.1, 0.7], vertical_alignment="center")
with settings_slot:
    with st.popover("⚙", help="Settings"):
        mode_choice = st.radio("Mode", MODES, key="mode_choice",
                               captions=["Uplift when the mood is low, otherwise Resonance",
                                         "Songs that match my mood", "Gently lift my mood"])
        user_text = st.text_input("Tell me how you feel (optional)", key="user_text",
                                  placeholder="e.g. Finally finished my exams!")
with rescan_slot:
    if st.button("Re-scan", help="Look at the webcam again"):
        state.pop("face_probs", None)

# The webcam is only opened on first load and on Re-scan; changing a setting
# reuses the cached face result.
if "face_probs" not in state:
    listening = st.empty()
    listening.markdown(LISTENING_HTML, unsafe_allow_html=True)
    state["face_probs"] = scan_face()
    state["scan_id"] = state.get("scan_id", 0) + 1
    listening.empty()

face_probs = state["face_probs"]
text_probs = text_probabilities(user_text)
emotions, fused = combine_emotions(face_probs, text_probs)

if fused is None:
    head.markdown('<div class="app-head"><span class="app-name">Emotion Music Player</span>'
                  '<span class="chip">No mood detected</span></div>', unsafe_allow_html=True)
    st.warning("I couldn't see a face. Press Re-scan, or open the ⚙ settings and tell me how you feel.")
    st.stop()

# The queue is rebuilt only when the scan, the text or the mode changes, so
# other reruns leave the player (and the music) alone.
queue_key = (state["scan_id"], user_text.strip(), mode_choice)
if state.get("queue_key") != queue_key:
    state["queue"] = build_queue(songs, emotions, fused, mode_choice)
    state["queue_key"] = queue_key
queue = state["queue"]

if face_probs is not None and text_probs is not None:
    source = "face + text"
elif face_probs is not None:
    source = "face" + (" (text model not found)" if user_text.strip() and load_text_model() is None else "")
else:
    source = "text (no face detected)"
feeling = ", ".join(f"{e} {fused[e]:.0%}" for e in emotions)
head.markdown(
    '<div class="app-head"><span class="app-name">Emotion Music Player</span>'
    f'<span class="chip">Feeling: {html.escape(feeling)} → <b>{queue["mode"]}</b></span>'
    f'<span class="chip-source">from {html.escape(source)}</span></div>', unsafe_allow_html=True)

if queue["note"]:
    st.markdown(f'<div class="notice">{html.escape(queue["note"])}</div>', unsafe_allow_html=True)
if previews.rate_limited():
    st.markdown('<div class="notice">The preview service is limiting requests right now, so some songs could not be '
                'checked. Re-scan in a minute for a fuller list.</div>', unsafe_allow_html=True)

components.html(player.render(queue["tracks"], queue["title"], queue["subtitle"], queue["journey"]),
                height=PLAYER_HEIGHT)
