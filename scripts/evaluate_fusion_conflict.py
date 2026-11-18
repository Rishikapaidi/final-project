"""Fusion when the face and the text disagree: what the app's reading does.

scripts/evaluate_fusion.py only pairs a face and a comment with the SAME true
emotion. Here every pair has a face whose FER2013 label is A and a GoEmotions
comment whose label is B, with A != B: all 42 ordered pairs of the 7
emotions, PAIRS faces and PAIRS comments per ordered pair (drawn without
replacement within one draw), REPEATS draws, one generator seeded with SEED,
in a fixed order (repeat, then A, then B). Sources: FER2013 PrivateTest
(whole 48x48 faces) and GoEmotions test (single-label comments). The same
face or comment can appear in several pairs.

There is no right answer for such a pair, so this reports behaviour only,
through the app's own code (app.py combine_emotions via scripts/app_code.py,
mood.headline, ui.band, uplift.negative_share):
- how often the fused top emotion is the face's label A, the text's label B,
  or neither; the mean fused probability of A and of B;
- the same split against what each model actually said (its top emotion),
  where the two models' top emotions differ;
- how that changes with each signal's confidence (its calibrated top
  probability, as the app's Face / Text bars show it), in bins;
- how often the headline is "Mostly X" or "Mixed: X and Y";
- polarity conflicts (face happy/neutral + text sad/angry/fear/disgust, and
  the reverse): the "X% low" band (Light < 25% <= In between < 50% <= Low)
  and which side wins. Low is also when Auto mode picks Uplift.

HSEmotion (the app's default) is the primary model; the CNN is a second row.
Each uses its own fusion settings (face_models.fusion_config).

Usage (from the project root):
    python scripts/evaluate_fusion_conflict.py
Writes  models/fusion/conflict/report.txt, by_pair.csv, by_confidence.csv,
        polarity.csv, conflict_heatmap.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_common as ec
import uplift
from fusion import EMOTIONS

OUT = os.path.join(ec.ROOT, "models", "fusion", "conflict")
PAIRS = 50
REPEATS = 20
SEED = 2026
BINS = [0.0, 0.4, 0.7, 1.0001]
BIN_NAMES = ["low (<0.40)", "medium (0.40-0.70)", "high (>=0.70)"]
BANDS = ["Light", "In between", "Low"]


def conflict_pairs(face_y, text_y):
    rng = np.random.default_rng(SEED)
    rows = []
    for r in range(REPEATS):
        for a in EMOTIONS:
            for b in EMOTIONS:
                if a == b:
                    continue
                f = rng.choice(np.flatnonzero(face_y == a), PAIRS, replace=False)
                t = rng.choice(np.flatnonzero(text_y == b), PAIRS, replace=False)
                rows += [(r, a, b, fi, ti) for fi, ti in zip(f, t)]
    return pd.DataFrame(rows, columns=["repeat", "face_label", "text_label", "face_idx", "text_idx"])


def readings(setup, pairs):
    face_cal = np.array([setup.face_view(x) for x in setup.face_raw])
    text_cal = np.array([setup.text_view(x) for x in setup.text_raw])
    fused = np.array([setup.fused(f, t)[1] for f, t in zip(pairs.face_idx, pairs.text_idx)])
    d = pairs.copy()
    E = np.array(EMOTIONS)
    a_i = d.face_label.map(EMOTIONS.index).values
    b_i = d.text_label.map(EMOTIONS.index).values
    fc, tc = face_cal[d.face_idx], text_cal[d.text_idx]
    d["face_vote"], d["text_vote"] = E[fc.argmax(1)], E[tc.argmax(1)]
    d["face_conf"], d["text_conf"] = fc.max(1), tc.max(1)
    d["fused_top"] = E[fused.argmax(1)]
    d["p_A"], d["p_B"] = fused[np.arange(len(d)), a_i], fused[np.arange(len(d)), b_i]
    d["headline"] = [ec.headline(p) for p in fused]
    d["mixed"] = d.headline.str.startswith("Mixed")
    two = np.argsort(-fused, axis=1)[:, :2]
    d["mixed_names_A_and_B"] = d.mixed.values & np.array([set(E[x]) == {a, b} for x, a, b in zip(two, d.face_label, d.text_label)])
    neg = [EMOTIONS.index(e) for e in uplift.NEGATIVE_EMOTIONS]
    d["low"] = fused[:, neg].sum(1)
    d["face_low"], d["text_low"] = fc[:, neg].sum(1), tc[:, neg].sum(1)
    for col in ["low", "face_low", "text_low"]:
        d[col.replace("low", "band")] = [ec.band(s) for s in d[col]]
    d["outcome"] = np.where(d.fused_top == d.face_label, "face label",
                            np.where(d.fused_top == d.text_label, "text label", "neither"))
    return d


def pct(x):
    return f"{x:.1%}"


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("FUSION WHEN THE FACE AND THE TEXT DISAGREE (behaviour only: these pairs have no right answer)")
    log(f"Pairs: face from FER2013 PrivateTest with label A, comment from GoEmotions test with label B, A != B;")
    log(f"all 42 ordered (A, B), {PAIRS} pairs each per draw, {REPEATS} draws (seed {SEED}) = {42 * PAIRS * REPEATS:,} pairs per face model.")
    log("Readings come from app.py's combine_emotions; headline from mood.headline; band from ui.band.\n")

    by_pair, by_conf, polarity = [], [], []
    heat = {}
    for key, name in ec.MODELS:
        setup = ec.Setup(key)
        cfg = setup.config
        pairs = conflict_pairs(setup.face_y, setup.text_y)
        d = readings(setup, pairs)
        log(f"=================== {name}: {cfg['method']}, face weight {cfg['face_weight']:.2f}, "
            f"T_face {cfg['face_temperature']:.3f}, T_text {cfg['text_temperature']:.3f}")

        # (a) overall
        o = d.outcome.value_counts(normalize=True)
        log(f"\n(a) All {len(d):,} conflict pairs. Fused top emotion is the face's label A: {pct(o.get('face label', 0))} | "
            f"the text's label B: {pct(o.get('text label', 0))} | neither: {pct(o.get('neither', 0))}")
        log(f"    mean fused probability of A {d.p_A.mean():.3f} (std {d.p_A.std():.3f}), of B {d.p_B.mean():.3f} (std {d.p_B.std():.3f})")
        log(f"    headline 'Mostly X': {pct(1 - d.mixed.mean())} | 'Mixed: X and Y': {pct(d.mixed.mean())} "
            f"(of the Mixed ones, naming exactly A and B: {pct(d.mixed_names_A_and_B[d.mixed].mean())})")
        mostly = d[~d.mixed]
        log(f"    'Mostly X' names A {pct((mostly.fused_top == mostly.face_label).mean())}, "
            f"B {pct((mostly.fused_top == mostly.text_label).mean())}, another emotion {pct(((mostly.fused_top != mostly.face_label) & (mostly.fused_top != mostly.text_label)).mean())}")
        per_repeat = d.groupby("repeat").outcome.value_counts(normalize=True).unstack().fillna(0)
        log("    spread over the 20 draws (std of the share): "
            + ", ".join(f"{c} {per_repeat[c].std():.1%}" for c in ["face label", "text label", "neither"] if c in per_repeat))

        # against what the models said
        dis = d[d.face_vote != d.text_vote]
        log(f"\n    Against the models' own top emotions: they differ in {pct(len(dis) / len(d))} of the pairs "
            f"(the face model said A in {pct((d.face_vote == d.face_label).mean())}, the text model said B in {pct((d.text_vote == d.text_label).mean())}).")
        log(f"    Where they differ, the fused top is the face model's emotion {pct((dis.fused_top == dis.face_vote).mean())}, "
            f"the text model's {pct((dis.fused_top == dis.text_vote).mean())}, another {pct(((dis.fused_top != dis.face_vote) & (dis.fused_top != dis.text_vote)).mean())}.")

        # (b) per ordered pair
        g = d.groupby(["face_label", "text_label"])
        t = pd.DataFrame({"pairs": g.size(), "top_is_face_label": g.outcome.apply(lambda s: (s == "face label").mean()),
                          "top_is_text_label": g.outcome.apply(lambda s: (s == "text label").mean()),
                          "top_is_neither": g.outcome.apply(lambda s: (s == "neither").mean()),
                          "mean_p_A": g.p_A.mean(), "mean_p_B": g.p_B.mean(), "mixed_headline": g.mixed.mean(),
                          "band_light": g.band.apply(lambda s: (s == "Light").mean()),
                          "band_in_between": g.band.apply(lambda s: (s == "In between").mean()),
                          "band_low": g.band.apply(lambda s: (s == "Low").mean())}).reset_index()
        t.insert(0, "face_model", key)
        by_pair.append(t)
        mA = t.pivot(index="face_label", columns="text_label", values="top_is_face_label").reindex(index=EMOTIONS, columns=EMOTIONS)
        mB = t.pivot(index="face_label", columns="text_label", values="top_is_text_label").reindex(index=EMOTIONS, columns=EMOTIONS)
        heat[key] = (name, mA, mB)
        fmt = lambda m: m.map(lambda x: "  -" if pd.isna(x) else f"{x:.0%}").to_string()
        log("\n(b) Share of pairs whose fused top emotion is the FACE's label A (rows: face label A, columns: text label B)")
        log(fmt(mA))
        log("    ... is the TEXT's label B")
        log(fmt(mB))
        log("    By face label A (rows averaged): " + ", ".join(f"{e} {mA.loc[e].mean():.0%}" for e in EMOTIONS))
        log("    By text label B (columns averaged), share = B: " + ", ".join(f"{e} {mB[e].mean():.0%}" for e in EMOTIONS))

        # (c) confidence bins
        d["face_bin"] = pd.cut(d.face_conf, BINS, labels=BIN_NAMES, right=False)
        d["text_bin"] = pd.cut(d.text_conf, BINS, labels=BIN_NAMES, right=False)
        log("\n(c) By each signal's confidence (its calibrated top probability). Cells: n | fused top = A / = B | mean p(A) / p(B)")
        log(f"    face confidence: " + ", ".join(f"{b} {pct((d.face_bin == b).mean())}" for b in BIN_NAMES)
            + " | text confidence: " + ", ".join(f"{b} {pct((d.text_bin == b).mean())}" for b in BIN_NAMES))
        grid = []
        for fb in BIN_NAMES:
            for tb in BIN_NAMES:
                c = d[(d.face_bin == fb) & (d.text_bin == tb)]
                cd = c[c.face_vote != c.text_vote]
                row = {"face_model": key, "face_confidence": fb, "text_confidence": tb, "pairs": len(c),
                       "top_is_face_label": (c.outcome == "face label").mean() if len(c) else np.nan,
                       "top_is_text_label": (c.outcome == "text label").mean() if len(c) else np.nan,
                       "mean_p_A": c.p_A.mean(), "mean_p_B": c.p_B.mean(),
                       "pairs_models_differ": len(cd),
                       "top_is_face_vote": (cd.fused_top == cd.face_vote).mean() if len(cd) else np.nan,
                       "top_is_text_vote": (cd.fused_top == cd.text_vote).mean() if len(cd) else np.nan,
                       "mixed_headline": c.mixed.mean() if len(c) else np.nan}
                grid.append(row)
        gdf = pd.DataFrame(grid)
        by_conf.append(gdf)
        log(f"    {'face \\ text':20}" + "".join(f"{b:>34}" for b in BIN_NAMES))
        for fb in BIN_NAMES:
            cells = []
            for tb in BIN_NAMES:
                r = gdf[(gdf.face_confidence == fb) & (gdf.text_confidence == tb)].iloc[0]
                cells.append(f"{r.pairs:>5} | {r.top_is_face_label:4.0%} / {r.top_is_text_label:4.0%} | {r.mean_p_A:.2f} / {r.mean_p_B:.2f}"
                             if r.pairs else f"{0:>5} | -")
            log(f"    {fb:20}" + "".join(f"{c:>34}" for c in cells))
        log("    Where the two models' top emotions differ, share won by the face model's / the text model's emotion:")
        log(f"    {'face \\ text':20}" + "".join(f"{b:>22}" for b in BIN_NAMES))
        for fb in BIN_NAMES:
            cells = []
            for tb in BIN_NAMES:
                r = gdf[(gdf.face_confidence == fb) & (gdf.text_confidence == tb)].iloc[0]
                cells.append(f"{r.pairs_models_differ:>5}: {r.top_is_face_vote:4.0%} / {r.top_is_text_vote:4.0%}" if r.pairs_models_differ else "-")
            log(f"    {fb:20}" + "".join(f"{c:>22}" for c in cells))

        # (d) polarity conflicts
        log("\n(d) Polarity conflicts. Band = the 'X% low' gauge (share of sad + angry + fear + disgust). Low also means Auto picks Uplift.")
        cases = [("face happy/neutral + text sad/angry/fear/disgust", d.face_label.isin(ec.POSITIVE_FACE) & d.text_label.isin(ec.NEGATIVE)),
                 ("face sad/angry/fear/disgust + text happy/neutral", d.face_label.isin(ec.NEGATIVE) & d.text_label.isin(ec.POSITIVE_FACE))]
        for case, mask in cases:
            c = d[mask]
            def bands(col):
                v = c[col].value_counts(normalize=True)
                return " / ".join(f"{v.get(b, 0):.1%}" for b in BANDS)
            neg_top = c.fused_top.isin(ec.NEGATIVE).mean()
            log(f"-- {case}: {len(c):,} pairs")
            log(f"   band Light / In between / Low: fused {bands('band')} | face alone {bands('face_band')} | text alone {bands('text_band')}")
            log(f"   mean 'low' share: fused {c.low.mean():.2f} (std {c.low.std():.2f}) | face alone {c.face_low.mean():.2f} | text alone {c.text_low.mean():.2f}")
            log(f"   fused top emotion: the face's label {pct((c.outcome == 'face label').mean())}, the text's label {pct((c.outcome == 'text label').mean())}, "
                f"neither {pct((c.outcome == 'neither').mean())}; a heavy emotion (sad/angry/fear/disgust) {pct(neg_top)}")
            log(f"   {'face A':9}{'text B':9}{'pairs':>6}{'Light':>8}{'In betw.':>9}{'Low':>7}{'top=A':>8}{'top=B':>8}{'neither':>9}")
            for (a, b), s in c.groupby(["face_label", "text_label"]):
                v = s.band.value_counts(normalize=True)
                o = s.outcome.value_counts(normalize=True)
                log(f"   {a:9}{b:9}{len(s):>6}" + "".join(f"{v.get(x, 0):>8.0%}" if i != 1 else f"{v.get(x, 0):>9.0%}" for i, x in enumerate(BANDS))
                    + f"{o.get('face label', 0):>8.0%}{o.get('text label', 0):>8.0%}{o.get('neither', 0):>9.0%}")
                polarity.append({"face_model": key, "case": case, "face_label": a, "text_label": b, "pairs": len(s),
                                 **{f"band_{x.lower().replace(' ', '_')}": v.get(x, 0) for x in BANDS},
                                 "top_is_face_label": o.get("face label", 0), "top_is_text_label": o.get("text label", 0),
                                 "top_is_neither": o.get("neither", 0), "mean_low": s.low.mean()})
        log("")

    log("What this does not show: which reading is right (no pair has a true answer); real people whose face and words")
    log("disagree (these are unrelated faces and comments put together); webcam faces (whole 48x48 FER2013 images, no detector).")
    log("The CNN's cached outputs are of one crop, not averaged with its mirror image as in the app.")

    pd.concat(by_pair).to_csv(os.path.join(OUT, "by_pair.csv"), index=False, float_format="%.4f")
    pd.concat(by_conf).to_csv(os.path.join(OUT, "by_confidence.csv"), index=False, float_format="%.4f")
    pd.DataFrame(polarity).to_csv(os.path.join(OUT, "polarity.csv"), index=False, float_format="%.4f")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6))
    for row, key in enumerate(k for k, _ in ec.MODELS):
        name, mA, mB = heat[key]
        for col, (m, what) in enumerate([(mA, "the face's label A"), (mB, "the text's label B")]):
            ax = axes[row, col]
            ax.imshow(m.values.astype(float), cmap="Blues", vmin=0, vmax=1)
            for i in range(7):
                for j in range(7):
                    if i != j:
                        v = m.values[i, j]
                        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "black")
            ax.set_xticks(range(7), EMOTIONS, rotation=35, ha="right"); ax.set_yticks(range(7), EMOTIONS)
            ax.set_xlabel("text label B"); ax.set_ylabel("face label A")
            ax.set_title(f"{name}: fused top = {what}", fontsize=10)
    fig.suptitle(f"Conflict pairs (A != B), {PAIRS * REPEATS} pairs per cell; diagonal not used")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "conflict_heatmap.png"), dpi=120)
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
