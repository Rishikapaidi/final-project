"""Shared by the offline evaluations: the cached model outputs, the app's fusion path, the test pairs.

Model outputs come from the caches scripts/evaluate_fusion.py writes in
data/fusion/ (so no model is loaded here):
  face_test.npz             the CNN (model.h5) on FER2013 PrivateTest
  face_test_hsemotion.npz   HSEmotion on FER2013 PrivateTest
  text_test.npz             DistilBERT on GoEmotions test (single-label comments)
Each face is the whole 48x48 image. The caches hold probabilities in fusion.EMOTIONS
order; the app's combine_emotions takes the face model's own output order
(fusion.FACE_MODEL_LABELS), so face rows are put back into that order first.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import app_code  # noqa: E402
import face_models  # noqa: E402
from fusion import EMOTIONS, FACE_MODEL_LABELS, _canonical, align_face, align_text  # noqa: E402

PRIMARY = "hsemotion"                    # the app's default face model
MODELS = [("hsemotion", "HSEmotion (default)"), ("cnn", "CNN (secondary)")]
CACHE = {"cnn": "face_test.npz", "hsemotion": "face_test_hsemotion.npz"}
NEGATIVE = ["sad", "angry", "fear", "disgust"]       # uplift.NEGATIVE_EMOTIONS
POSITIVE_FACE = ["happy", "neutral"]


def _load(name):
    z = np.load(os.path.join(ROOT, "data", "fusion", name), allow_pickle=False)
    return z["probs"], z["labels"].astype(str)


def to_face_order(aligned):
    """EMOTIONS order -> FACE_MODEL_LABELS order (the inverse of fusion.align_face)."""
    order = [[_canonical(l) for l in FACE_MODEL_LABELS].index(e) for e in EMOTIONS]
    raw = np.zeros_like(aligned)
    raw[:, order] = aligned
    assert np.allclose(align_face(raw), aligned)
    return raw


def text_order(aligned):
    """EMOTIONS order -> the text model's output order (the same order; checked)."""
    assert np.allclose(align_text(aligned), aligned)
    return aligned


class Setup:
    """One face model: its raw test outputs, the app's fusion function with its settings."""

    def __init__(self, key, **config_changes):
        self.key = key
        self.name = dict(MODELS)[key]
        self.config = {**face_models.fusion_config(key), **config_changes}
        aligned, self.face_y = _load(CACHE[key])
        self.face_raw = to_face_order(aligned)
        self.text_raw, self.text_y = _load("text_test.npz")
        self.text_raw = text_order(self.text_raw)
        ns = app_code.load(["combine_emotions", "face_view", "text_view"], FUSION_CONFIG=self.config)
        self.combine, self.face_view, self.text_view = ns["combine_emotions"], ns["face_view"], ns["text_view"]

    def fused(self, f, t, face=True, text=True):
        """The app's reading for face index f and text index t: (emotions for the playlist, fused probs as an array)."""
        emotions, fused = self.combine(self.face_raw[f] if face else None, self.text_raw[t] if text else None)
        return emotions, np.array([fused[e] for e in EMOTIONS])


def test_pairs(face_y, text_y):
    """The test pairings of scripts/evaluate_fusion.py, rebuilt exactly: the same seed, and the
    validation pairings drawn first from the same generator (their labels are needed for that)."""
    import evaluate_fusion as ef
    _, val_face_y = _load("face_val.npz")
    _, val_text_y = _load("text_val.npz")
    rng = np.random.default_rng(ef.SEED)
    ef.make_pairings(val_face_y, val_text_y, rng)
    return ef.make_pairings(face_y, text_y, rng)


def band(share):
    import ui
    return ui.band(share)[0]


def headline(fused_array):
    import mood
    return mood.headline(dict(zip(EMOTIONS, fused_array)))
