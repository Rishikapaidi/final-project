# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64
import json
import os

from fusion import EMOTIONS, fuse
import uplift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_emotion_distilbert", "model")
with open(os.path.join(BASE_DIR, "models", "fusion", "fusion_config.json")) as f:
    FUSION_CONFIG = json.load(f)
# Emotions whose fused probability is at least this share go into the playlist mix.
MIN_EMOTION_SHARE = 0.15
RESONANCE_MODE = "Resonance (match my mood)"
UPLIFT_MODE = "Uplift (gently lift my mood)"

# Songs with one emotion bucket each, built from muse_v3.csv by
# scripts/prepare_songs.py (see data/README.md for how and why).
SONGS_FILE = os.path.join(BASE_DIR, "data", "songs", "songs_clean.csv")
if not os.path.exists(SONGS_FILE):
    st.error("Song file not found. Build it once with:  python scripts/prepare_songs.py")
    st.stop()

df = pd.read_csv(SONGS_FILE)
df['link'] = df['lastfm_url']
df['name'] = df['title']

SONG_BUCKETS = {bucket: songs for bucket, songs in df.groupby('bucket')}

# Map each detected emotion (case-insensitive) to its song bucket. The face
# model says Disgusted/Fearful/Surprised; the text model says disgust/fear/surprise.
EMOTION_BUCKETS = {
    'angry': SONG_BUCKETS['angry'],
    'disgusted': SONG_BUCKETS['disgust'],
    'disgust': SONG_BUCKETS['disgust'],
    'fearful': SONG_BUCKETS['fear'],
    'fear': SONG_BUCKETS['fear'],
    'happy': SONG_BUCKETS['happy'],
    'surprised': SONG_BUCKETS['surprise'],
    'surprise': SONG_BUCKETS['surprise'],
    'neutral': SONG_BUCKETS['neutral'],
    'sad': SONG_BUCKETS['sad'],
}

# How many songs to sample for each emotion, keyed by how many distinct
# emotions were detected (most frequent emotion first).
SAMPLE_SIZES = {
    1: [30],
    2: [30, 20],
    3: [55, 20, 15],
    4: [30, 29, 18, 9],
}

def fun(list):

    data = pd.DataFrame()
    times = SAMPLE_SIZES.get(len(list), [10, 7, 6, 5, 2])

    for v, t in zip(list, times):
        bucket = EMOTION_BUCKETS.get(v.strip().lower(), SONG_BUCKETS['neutral'])
        data = pd.concat([data, bucket.sample(n=t)], ignore_index=True)

    # Songs are added one emotion at a time, and only the first 30 are shown,
    # so shuffle to make the displayed songs reflect the whole mix.
    data = data.sample(frac=1).reset_index(drop=True)

    print("data of list func... :",data)
    return data

    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
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

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


@st.cache_resource
def load_text_model():
    """DistilBERT text-emotion model, or None if it hasn't been trained on this machine."""
    if not os.path.isdir(TEXT_MODEL_DIR):
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR).eval()
    return tokenizer, text_model


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


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>"
            , unsafe_allow_html=True)

mode = st.radio("Mode", [RESONANCE_MODE, UPLIFT_MODE], horizontal=True)

user_text = st.text_area("How are you feeling? (optional, combined with your face)",
                         placeholder="e.g. Finally finished my exams, can't wait for the weekend!")

col1,col2,col3 = st.columns(3)

list = []
uplift_playlist, uplift_journey = None, None
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click here)'):

        count = 0
        list.clear()
        face_frame_probs = []

        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")          
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)          
            count = count + 1

            for (x, y, w, h) in faces:               
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)               
                roi_gray = gray[y:y + h, x:x + w]               
                # The model was trained on pixels scaled to 0-1, not raw 0-255.
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0).astype('float32') / 255.0
                prediction = model.predict(cropped_img)             
                max_index = int(np.argmax(prediction))

                face_frame_probs.append(prediction[0])

                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)              
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            if count >= 20:
                break
        cap.release()
        cv2.destroyAllWindows()

        face_probs = np.mean(face_frame_probs, axis=0) if face_frame_probs else None
        text_probs = text_probabilities(user_text)
        list, fused = combine_emotions(face_probs, text_probs)

        if fused is None:
            st.warning("No face detected and no text entered, so there's no emotion to recommend for.")
        else:
            if face_probs is not None and text_probs is not None:
                source = "face + text"
            elif face_probs is not None:
                source = "face only" + (" (text model not found)" if user_text.strip() and load_text_model() is None else "")
            else:
                source = "text only (no face detected)"
            shares = ", ".join(f"{e} {fused[e]:.0%}" for e in list)
            st.success(f"Detected ({source}): {shares}")

            if mode == UPLIFT_MODE:
                uplift_playlist, uplift_journey = uplift.build_playlist(df, fused, list)
                if uplift_playlist is None:
                    st.info(f"Uplift is for low moods. Sad, angry, fearful and disgusted feelings make up only "
                            f"{uplift.negative_share(fused):.0%} of this result, so here are songs that match your mood instead.")
        

with col3:
    pass

if uplift_playlist is not None:
    # Five songs in play order, from the detected mood towards the target.
    new_df = uplift_playlist
    heading = "Uplift playlist: from {} towards {}, play in this order".format(
        uplift_journey["start_emotion"], " then ".join(uplift_journey["targets"]))
else:
    new_df = fun(list)
    heading = "Recommended song's with artist names"
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>{}</b></h5>".format(heading)
            , unsafe_allow_html=True)

if uplift_playlist is not None:
    st.pyplot(uplift.trajectory_figure(uplift_playlist, uplift_journey), use_container_width=False)

st.write("---------------------------------------------------------------------------------------------------------------------")

try:
  
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):

        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>"""
                    .format(l,i+1,n),unsafe_allow_html=True)
        if uplift_playlist is not None:
            a = "{} · valence {:.1f}, arousal {:.1f}".format(a, new_df["valence_tags"][i], new_df["arousal_tags"][i])
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>" 
                    .format(a), unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except:
    pass