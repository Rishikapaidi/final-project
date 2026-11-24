"""Combine face and text emotion predictions into one emotion.

The face model (HSEmotion by default; or the CNN model.h5, inherited, training
not documented) and the text model (DistilBERT, GoEmotions mapped to the
FER2013 categories) predict the same 7 emotions, but list them in different
orders and spellings. Everything here works on the canonical order below, and
`align_face` / `align_text` reorder each model's raw probability vector by name.

Two fusion rules, both weighted by face_weight (text_weight = 1 - face_weight):
- "weighted_average": p = w * p_face + (1 - w) * p_text
- "log_linear":       log p = w * log p_face + (1 - w) * log p_text (renormalised),
                      i.e. a weighted product of the two distributions

Apply `calibrate` with each model's temperature before fusing; otherwise the
weights mostly reflect which model is more overconfident.
"""
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Output order of each model, from app.py's emotion_dict and the DistilBERT
# model's id2label respectively.
FACE_MODEL_LABELS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
TEXT_MODEL_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

_ALIASES = {"disgusted": "disgust", "fearful": "fear", "surprised": "surprise"}
METHODS = ("weighted_average", "log_linear")
_EPS = 1e-12


def _canonical(label):
    label = label.strip().lower()
    return _ALIASES.get(label, label)


def _reorder(probs, model_labels):
    probs = np.asarray(probs, dtype=float)
    order = [[_canonical(l) for l in model_labels].index(e) for e in EMOTIONS]
    return probs[..., order]


def align_face(probs):
    """Reorder face-model probabilities (FACE_MODEL_LABELS order) to EMOTIONS order."""
    return _reorder(probs, FACE_MODEL_LABELS)


def align_text(probs):
    """Reorder text-model probabilities (TEXT_MODEL_LABELS order) to EMOTIONS order."""
    return _reorder(probs, TEXT_MODEL_LABELS)


def calibrate(probs, temperature):
    """Temperature-scale probabilities: softmax(log(p) / T). T > 1 softens, T < 1 sharpens."""
    logits = np.log(np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def fuse_probs(face_probs, text_probs, face_weight=0.5, method="weighted_average"):
    """Fuse two probability arrays already in EMOTIONS order.

    Works on a single vector (shape (7,)) or a batch (shape (n, 7)).
    Returns fused probabilities in EMOTIONS order.
    """
    if not 0.0 <= face_weight <= 1.0:
        raise ValueError(f"face_weight must be between 0 and 1, got {face_weight}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    face = np.asarray(face_probs, dtype=float)
    text = np.asarray(text_probs, dtype=float)
    if face.shape != text.shape or face.shape[-1] != len(EMOTIONS):
        raise ValueError(f"expected matching (..., {len(EMOTIONS)}) arrays, got {face.shape} and {text.shape}")
    text_weight = 1.0 - face_weight

    if method == "weighted_average":
        fused = face_weight * face + text_weight * text
    else:
        log_fused = (face_weight * np.log(np.clip(face, _EPS, 1.0))
                     + text_weight * np.log(np.clip(text, _EPS, 1.0)))
        log_fused -= log_fused.max(axis=-1, keepdims=True)
        fused = np.exp(log_fused)
    return fused / fused.sum(axis=-1, keepdims=True)


def fuse(face_probs, text_probs, face_weight=0.5, method="weighted_average",
         face_temperature=1.0, text_temperature=1.0):
    """Combine raw outputs of the two models into one emotion.

    face_probs: face model softmax output, in FACE_MODEL_LABELS order
    text_probs: text model softmax output, in TEXT_MODEL_LABELS order
    Returns (emotion, {emotion: probability}) for a single prediction.
    """
    face = calibrate(align_face(face_probs), face_temperature)
    text = calibrate(align_text(text_probs), text_temperature)
    fused = fuse_probs(face, text, face_weight, method)
    if fused.ndim != 1:
        raise ValueError("fuse() takes one prediction; use fuse_probs() for batches")
    return EMOTIONS[int(fused.argmax())], dict(zip(EMOTIONS, fused.tolist()))
