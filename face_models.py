"""The face-expression models the app can use, behind one interface.

FACE_MODEL (an environment variable: cnn, ferplus or hsemotion) picks one,
for example to compare them live on a webcam; FACE_MODEL_DEFAULT (hsemotion) is used
otherwise. models/face/report.txt compares them on FER2013 PrivateTest:

  ferplus    FER+ (emotion-ferplus-8 from the ONNX model zoo, MIT licence).
             A CNN trained by Microsoft on the FER2013 training images with
             the FER+ crowd-sourced labels (10 taggers per image). Input: the
             face box, grayscale, 64x64, raw 0-255 pixels.
  hsemotion  HSEmotion enet_b0_8_best_afew (Andrey Savchenko; code Apache-2.0).
             EfficientNet-B0 pretrained on VGGFace2 and fine-tuned on
             AffectNet, whose licence allows non-commercial research and
             education only, so these weights carry that limit. Input: the
             face box in colour (RGB), 224x224, ImageNet normalisation.
  cnn        The CNN (model.h5) inherited from the project this one started
             from (see CREDITS.md). Its training is not documented; its 7
             classes and 48x48 grayscale input match FER2013. Input: the
             face box, grayscale, 48x48, pixels / 255. It is also the
             fallback when an ONNX model's file or onnxruntime is missing.

The ONNX files are not in the repository: scripts/fetch_face_models.py
downloads them into models/face/onnx/ (and checks their SHA-256).

Every model is given the same face box (face.find_faces) and builds its own
crop from the original frame. Every model returns probabilities in
fusion.FACE_MODEL_LABELS order, so fusion and everything after it work
unchanged. Classes outside our seven (FER+ and HSEmotion's "contempt") are
added to "disgust". Nothing here is trained or tuned on anyone's face.
"""
import importlib.util
import json
import os

import cv2
import numpy as np

import face
from fusion import FACE_MODEL_LABELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(BASE_DIR, "models", "face", "onnx")
CNN_WEIGHTS = os.path.join(BASE_DIR, "model.h5")
# Chosen after an informal live webcam comparison by the developer, and because it was trained on natural
# colour photos. On the public benchmark no model was clearly better, and webcam accuracy was not measured.
FACE_MODEL_DEFAULT = "hsemotion"
NAMES = {"ferplus": "FER+", "hsemotion": "HSEmotion", "cnn": "the original CNN"}

# Each model's own classes -> ours (FACE_MODEL_LABELS); "contempt" goes into disgust.
FERPLUS_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
HSE8_LABELS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
TO_OURS = {"neutral": "Neutral", "happiness": "Happy", "surprise": "Surprised", "sadness": "Sad", "anger": "Angry",
           "disgust": "Disgusted", "fear": "Fearful", "contempt": "Disgusted"}


def to_face_labels(probs, labels):
    """(n, k) probabilities over a model's own labels -> (n, 7) in FACE_MODEL_LABELS order."""
    out = np.zeros((len(probs), len(FACE_MODEL_LABELS)))
    for j, label in enumerate(labels):
        out[:, FACE_MODEL_LABELS.index(TO_OURS[label])] += probs[:, j]
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cut(image, box):
    x, y, w, h = box
    return image[max(y, 0):y + h, max(x, 0):x + w]


class CNNModel:
    """The original FER2013 CNN (TensorFlow is imported only when this model is loaded).
    A crop and its mirror image are averaged (on FER2013 PrivateTest this adds about
    half a point on the original labels and 1.6 points on the FER+ labels)."""
    key, name = "cnn", NAMES["cnn"]

    def __init__(self):
        self.model = face.build_model(CNN_WEIGHTS)

    def prepare(self, frame, gray, box):
        return face.crop(gray, box)

    def predict(self, inputs):
        if not inputs:
            return np.empty((0, len(FACE_MODEL_LABELS)))
        x = np.stack(inputs)
        return (self.model.predict(x, verbose=0) + self.model.predict(x[:, :, ::-1], verbose=0)) / 2


class FERPlusModel:
    """FER+ (emotion-ferplus-8). It takes one face at a time."""
    key, name = "ferplus", NAMES["ferplus"]
    FILE = "emotion-ferplus-8.onnx"

    def __init__(self, session):
        self.session, self.input = session, session.get_inputs()[0].name

    def prepare(self, frame, gray, box):
        return cv2.resize(_cut(gray, box), (64, 64))[None].astype("float32")

    def predict(self, inputs):
        if not inputs:
            return np.empty((0, len(FACE_MODEL_LABELS)))
        logits = np.concatenate([self.session.run(None, {self.input: x[None]})[0] for x in inputs])
        return to_face_labels(softmax(logits), FERPLUS_LABELS)


class HSEmotionModel:
    """HSEmotion enet_b0_8_best_afew, on the face box in colour."""
    key, name = "hsemotion", NAMES["hsemotion"]
    FILE = "enet_b0_8_best_afew.onnx"
    MEAN, STD = np.array([0.485, 0.456, 0.406], "float32"), np.array([0.229, 0.224, 0.225], "float32")

    def __init__(self, session):
        self.session, self.input = session, session.get_inputs()[0].name

    def prepare(self, frame, gray, box):
        rgb = cv2.cvtColor(_cut(frame, box), cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(_cut(gray, box), cv2.COLOR_GRAY2RGB)
        return ((cv2.resize(rgb, (224, 224)).astype("float32") / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)

    def predict(self, inputs):
        if not inputs:
            return np.empty((0, len(FACE_MODEL_LABELS)))
        logits = np.concatenate([self.session.run(None, {self.input: np.stack(inputs[i:i + 32])})[0][:, :8]
                                 for i in range(0, len(inputs), 32)])
        return to_face_labels(softmax(logits), HSE8_LABELS)


ONNX_MODELS = {"ferplus": FERPlusModel, "hsemotion": HSEmotionModel}


def wanted():
    """The model asked for: FACE_MODEL if it names one, otherwise FACE_MODEL_DEFAULT."""
    name = os.environ.get("FACE_MODEL", "").strip().lower()
    return name if name in NAMES else FACE_MODEL_DEFAULT


def resolved(name=None):
    """The model load(name) would give, without loading it: the one asked for if its file
    and onnxruntime are there, otherwise the CNN."""
    name = name or wanted()
    if name in ONNX_MODELS and not (os.path.exists(os.path.join(ONNX_DIR, ONNX_MODELS[name].FILE))
                                    and importlib.util.find_spec("onnxruntime")):
        return "cnn"
    return name


def fusion_config(key):
    """The face + text fusion settings for a face model: models/fusion/<key>/fusion_config.json
    when scripts/evaluate_fusion.py --face <key> found a better fit for it on the validation
    pairs, otherwise the CNN's (models/fusion/fusion_config.json)."""
    own = os.path.join(BASE_DIR, "models", "fusion", key, "fusion_config.json")
    path = own if key != "cnn" and os.path.exists(own) else os.path.join(BASE_DIR, "models", "fusion", "fusion_config.json")
    with open(path) as f:
        return json.load(f)


def load(name=None):
    """The face model `name` (default: wanted()), or the CNN if it can't be loaded.
    The model returned has .fallback set to the reason when it isn't the one asked for."""
    name = name or wanted()
    reason = None
    if name in ONNX_MODELS:
        cls = ONNX_MODELS[name]
        path = os.path.join(ONNX_DIR, cls.FILE)
        if not os.path.exists(path):
            reason = f"{cls.FILE} is missing (run scripts/fetch_face_models.py)"
        else:
            try:
                import onnxruntime
                model = cls(onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"]))
                model.fallback = None
                return model
            except Exception as error:   # onnxruntime missing or the file unreadable
                reason = f"{cls.name} could not be loaded ({type(error).__name__})"
    model = CNNModel()
    model.fallback = reason
    return model
