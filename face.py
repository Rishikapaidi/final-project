"""Face detection, cropping and the face-emotion model.

Shared by the app and scripts/diagnose_face.py so both read faces the same
way. TensorFlow is only imported when the model is built.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from fusion import EMOTIONS, align_face

HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
PHOTO_MAX_WIDTH = 1280  # photos are shrunk to this before face detection


def load_photo(data):
    """A photo (bytes) as an OpenCV BGR frame, decoded in memory, or None if it isn't a picture.

    The EXIF rotation is applied, so phone pictures taken sideways come out
    upright, and photos wider than PHOTO_MAX_WIDTH are shrunk.
    """
    try:
        picture = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    if picture.width > PHOTO_MAX_WIDTH:
        picture = picture.resize((PHOTO_MAX_WIDTH, round(picture.height * PHOTO_MAX_WIDTH / picture.width)))
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def build_model(weights_path):
    """The FER2013 CNN with its trained weights (outputs in FACE_MODEL_LABELS order)."""
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

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
    model.load_weights(weights_path)
    return model


def detector():
    return cv2.CascadeClassifier(HAAR_CASCADE)


def find_faces(gray, face_detector):
    """Haar boxes (x, y, w, h) for every face found in a grayscale image."""
    return [tuple(int(v) for v in box)
            for box in face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)]


def crop(gray, box, pad=0.0, equalize=False):
    """The 48x48 model input for one face: pixels scaled to 0-1, shape (48, 48, 1).

    pad: extra margin on each side, as a share of the box size (clipped at the
    image edge). equalize: histogram-equalize the crop before resizing.
    With the defaults this is exactly the crop the app has always used.
    """
    x, y, w, h = box
    px, py = round(w * pad), round(h * pad)
    x0, y0 = max(x - px, 0), max(y - py, 0)
    x1, y1 = min(x + w + px, gray.shape[1]), min(y + h + py, gray.shape[0])
    face = gray[y0:y1, x0:x1]
    if equalize:
        face = cv2.equalizeHist(face)
    # The model was trained on pixels scaled to 0-1, not raw 0-255.
    return cv2.resize(face, (48, 48))[:, :, None].astype('float32') / 255.0


def predict(model, crops):
    """Face-model softmax outputs for a list of crops, in one batch."""
    if not crops:
        return np.empty((0, len(EMOTIONS)))
    return model.predict(np.stack(crops), verbose=0)


def as_emotions(probs):
    """Model output reordered to fusion.EMOTIONS."""
    return align_face(probs)
