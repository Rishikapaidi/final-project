"""Use app.py's own functions in the evaluation scripts, without running the page.

app.py builds the Streamlit page as soon as it is imported, so it cannot be
imported from a script. This reads app.py's source, keeps its imports
(except Streamlit's), the settings and the functions asked for, and runs
only those, so the scripts call exactly the code the app runs (for example
combine_emotions and build_queue), not a copy of it.

    ns = app_code.load(["combine_emotions"], FUSION_CONFIG=face_models.fusion_config("cnn"))
    emotions, fused = ns["combine_emotions"](face_probs, text_probs)

Overrides replace settings after they are loaded: the evaluations pass the
fusion settings of the face model being evaluated, which is what the app
itself uses when FACE_MODEL names that model.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
# Settings the functions below read; loaded whenever they are defined in app.py.
SETTINGS = {"BASE_DIR", "FUSION_CONFIG", "MIN_EMOTION_SHARE", "LANGUAGES", "UPLIFT_LANGUAGES", "MODES"}


def load(functions, **overrides):
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    tree = ast.parse(open(APP, encoding="utf-8").read(), APP)
    keep, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.Import) and not any(a.name.startswith("streamlit") for a in node.names):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and not (node.module or "").startswith("streamlit"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in SETTINGS for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            keep.append(node)
            found.add(node.name)
    missing = set(functions) - found
    if missing:
        raise KeyError(f"not defined at the top level of app.py: {sorted(missing)}")
    ns = {"__file__": APP, "__name__": "app_code_from_app"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), APP, "exec"), ns)
    ns.update(overrides)
    return ns
