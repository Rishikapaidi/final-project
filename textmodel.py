"""Load the DistilBERT text-emotion model once per server process, optionally in the background.

Streamlit reruns app.py on every interaction, so state that must survive
reruns lives here, in an imported module. The model is loaded at most once:
`load()` holds a lock while loading, so a request that arrives while the
background warm-up is still running waits for it and reuses the same model.

Nothing here calls Streamlit, so it is safe to run in a background thread.
"""
import os
import threading

_lock = threading.Lock()
# warm_up() has its own lock: _lock is held for the whole load, and a rerun
# calling warm_up() must not wait for that (the page would stay "running" and
# a replaced player would keep playing until the model had loaded).
_start_lock = threading.Lock()
_model = None          # (tokenizer, model) once loaded; False if the model folder is missing
_warm_up_started = False
loads = 0              # how many times the model was actually loaded (for tests)


def load(model_dir):
    """(tokenizer, model), loading them on first use; None if the model folder is missing."""
    global _model, loads
    with _lock:
        if _model is None:
            if not os.path.isdir(model_dir):
                _model = False
            else:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
                _model = (tokenizer, model)
                loads += 1
    return _model or None


def is_loaded():
    return bool(_model)


def warm_up(model_dir):
    """Start loading the model in a background thread, once per process.

    Does nothing if it has already been started or the model folder is missing.
    Returns True if a thread was started.
    """
    global _warm_up_started
    with _start_lock:
        if _warm_up_started or _model is not None or not os.path.isdir(model_dir):
            return False
        _warm_up_started = True
    threading.Thread(target=_warm, args=(model_dir,), name="text-model-warm-up", daemon=True).start()
    return True


def _warm(model_dir):
    try:
        load(model_dir)
    except Exception:
        # A failed warm-up is not fatal: the next real request tries again.
        pass
