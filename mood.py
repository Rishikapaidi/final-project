"""What the mood card says: an emoji, a colour, a headline and a short message.

The messages describe what the app noticed and what the playlist will do. They
are deliberately not advice and make no claim that music changes how anyone
feels.
"""

# The top emotion needs at least this share of the fused probability for the
# headline to say "Mostly <emotion>"; below it, the top two are named.
MOSTLY_SHARE = 0.5

EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
         "disgust": "😖", "surprise": "😮", "neutral": "😐"}

COLOURS = {"happy": "#f5c518", "sad": "#4a90d9", "angry": "#e5484d", "fear": "#9b6dd6",
           "disgust": "#8a9a5b", "surprise": "#ff9f43", "neutral": "#9aa0a6"}

MESSAGES = {
    "Resonance": {
        "happy": "You seem to be in a bright mood. Here are some upbeat songs to keep you company.",
        "sad": "It looks like a heavy moment. These songs sit with that feeling instead of rushing past it.",
        "angry": "You seem fired up. Here are songs with some edge, to match where you are right now.",
        "fear": "You seem a bit on edge. These songs share that uneasy mood, so you're not sitting in it alone.",
        "disgust": "Something seems to have left a bad taste. Here are some sharp, sardonic songs for the mood.",
        "surprise": "Something seems to have caught you off guard. Here are some quirky, unexpected songs to go with it.",
        "neutral": "You seem calm and steady. Here are some easy, unhurried songs for the background.",
    },
    "Uplift": {
        "sad": "It looks like a heavy moment. The playlist starts with songs that match it, eases into "
               "something calmer and ends a little brighter. Skip ahead whenever you like.",
        "angry": "You seem fired up. The playlist starts with songs that have some edge and moves step by "
                 "step towards calmer ones.",
        "fear": "You seem a bit on edge. The playlist starts close to that feeling and moves step by step "
                "towards calmer songs.",
        "disgust": "Something seems to have left a bad taste. The playlist starts in that mood and drifts "
                   "towards calmer songs.",
        "happy": "There's some brightness here, with heavier feelings alongside it. The playlist starts "
                 "with the heavier part and moves gently on from there.",
        "surprise": "You seem caught off guard, with some heavier feelings underneath. The playlist starts "
                    "there and moves gently towards steadier songs.",
        "neutral": "You seem fairly steady, with some heavier feelings underneath. The playlist starts "
                   "there and moves gently on.",
    },
}

# Resonance, when no single emotion reaches MOSTLY_SHARE: name both.
MIXED_RESONANCE = ("It looks like a mix of {first} and {second} right now. These songs follow that blend, "
                   "leaning towards the {first} side.")


def ranked(fused):
    """Emotions from most to least likely."""
    return sorted(fused, key=fused.get, reverse=True)


def headline(fused):
    first, second = ranked(fused)[:2]
    if fused[first] >= MOSTLY_SHARE:
        return f"Mostly {first}"
    return f"Mixed: {first} and {second}"


def card(fused, mode, journey=None):
    """Everything the mood card shows.

    mode: "Resonance" or "Uplift"; journey: the Uplift journey summary
    ({"start": ..., "targets": [...]}) when mode is "Uplift".
    """
    top, second = ranked(fused)[:2]
    mode_line = mode
    if mode == "Uplift" and journey:
        mode_line = "Uplift: " + " → ".join([journey["start"]] + journey["targets"])
    message = MESSAGES[mode][top]
    if mode == "Resonance" and fused[top] < MOSTLY_SHARE:
        message = MIXED_RESONANCE.format(first=top, second=second)
    return {"top": top, "emoji": EMOJI[top], "colour": COLOURS[top], "second_colour": COLOURS[second],
            "headline": headline(fused), "mode_line": mode_line, "message": message}
