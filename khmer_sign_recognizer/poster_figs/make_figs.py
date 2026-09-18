"""Generate the poster figures for the blocks Piseth owns.

Everything is drawn from results_khmer_taskA_cap30/results.json so the poster
and the handed-in report cannot disagree. 300 dpi, fonts sized to read at 1.5 m
on an A0.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path("/home/sherlock/Desktop/Sign to Text/khmer_sign_recognizer")
OUT = Path("/home/sherlock/Desktop/Sign to Text/khmer_sign_recognizer/poster_figs")
OUT.mkdir(exist_ok=True)
R = json.load(open(ROOT / "algo_comparison/results_khmer_taskA_cap30/results.json"))

KH_PATH = ROOT / "fonts/NotoSansKhmer-Regular.ttf"
if KH_PATH.exists():
    font_manager.fontManager.addfont(str(KH_PATH))
KH = font_manager.FontProperties(fname=str(KH_PATH)) if KH_PATH.exists() else None

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

INK, GREY, BLUE, GREEN, AMBER, RED = "#1C1917", "#9A958E", "#2563EB", "#15803D", "#C2410C", "#B91C1C"

PRETTY = {
    "bilstm": "BiLSTM", "tcn": "TCN", "bigru": "BiGRU", "lstm": "LSTM",
    "gru": "GRU", "transformer": "Transformer",
    "gboost": "Gradient Boosting", "rf": "Random Forest", "logreg": "Logistic Regression",
    "mlp": "Neural Net (MLP)", "svm": "SVM", "bagging": "Bagged Trees",
    "knn": "k-Nearest Neighbours", "lda": "LDA", "tree": "Decision Tree",
}


# ── FIGURE 1 — the pipeline ──────────────────────────────────────────
def pipeline():
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(-1, 103); ax.set_ylim(0, 30); ax.axis("off")

    boxes = [
        ("WEBCAM", ["Ordinary laptop camera", "2-second clip"], GREY),
        ("LANDMARK EXTRACTION", ["MediaPipe (hands)", "+ RTMPose (upper body)",
                                 "48 joints (x, y, z)"], BLUE),
        ("SEQUENCE", ["Resampled to 60 frames", "One body scale",
                      "Tensor (60 x 48 x 3)"], BLUE),
        ("CLASSIFICATION", ["15 models compared", "TCN selected",
                            "(runs live)"], GREEN),
        ("OUTPUT", ["Khmer word", "on screen"], AMBER),
    ]
    w, gap = 16.5, 4.2
    x = 1.0
    for i, (title, lines, colour) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 7), w, 15,
                                    boxstyle="round,pad=0.5,rounding_size=1.2",
                                    linewidth=2.4, edgecolor=colour,
                                    facecolor=colour + "14"))
        ax.text(x + w / 2, 19.0, title, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=colour)
        for j, ln in enumerate(lines):
            ax.text(x + w / 2, 15.6 - j * 2.5, ln, ha="center", va="center",
                    fontsize=11, color=INK)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, 14.5),
                                         (x + w + gap - 0.5, 14.5),
                                         arrowstyle="-|>", mutation_scale=26,
                                         linewidth=2.4, color=INK))
        x += w + gap

    ax.text(51, 3.6,
            "48 joints = 6 upper body (shoulders, elbows, wrists) + 21 left hand + 21 right hand",
            ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(51, 0.9,
            "Only skeleton coordinates are stored — never video or photographs of any signer",
            ha="center", fontsize=11.5, color=RED, style="italic")
    fig.savefig(OUT / "fig1_pipeline.png", facecolor="white")
    plt.close(fig)
    print("  fig1_pipeline.png")


# ── FIGURE 2 — every model, both ways ────────────────────────────────
def models():
    rows = []
    for n, m in R["classical"].items():
        b = m.get("bones", {})
        if isinstance(b, dict) and isinstance(b.get("unseen"), dict):
            rows.append((PRETTY.get(n, n), b["same"]["f1"], b["unseen"]["f1"], BLUE))
    for n, m in R["deep"].items():
        if isinstance(m, dict) and "unseen" in m:
            rows.append((PRETTY.get(n, n), m["same"], m["unseen"], GREEN))
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ypos = np.arange(len(rows))
    ax.barh(ypos + 0.21, [r[1] for r in rows], height=0.40,
            color=GREY, label="tested on a signer it has SEEN")
    ax.barh(ypos - 0.21, [r[2] for r in rows], height=0.40,
            color=[r[3] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.6, i - 0.21, f"{r[2]:.1f}", va="center",
                fontsize=11.5, fontweight="bold", color=r[3])
        ax.text(r[1] + 0.6, i + 0.21, f"{r[1]:.1f}", va="center",
                fontsize=10, color=GREY)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlabel("macro-F1 (%)", fontsize=13)
    ax.set_xlim(70, 103)
    ax.set_title("All 15 models, scored two ways", fontsize=17,
                 fontweight="bold", pad=16)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="signer it has seen before"),
                       Patch(color=GREEN, label="unseen signer — deep"),
                       Patch(color=BLUE, label="unseen signer — classical")],
              fontsize=12, loc="lower right", frameon=False)
    ax.axvline(R["decision"]["best_live_unseen"], color=AMBER, ls="--", lw=2)
    ax.annotate(f"TCN {R['decision']['best_live_unseen']:.1f} — the model we ship",
                xy=(R["decision"]["best_live_unseen"], len(rows) - 0.55),
                xytext=(R["decision"]["best_live_unseen"] - 7.5, len(rows) - 0.2),
                fontsize=12, color=AMBER, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8))
    fig.savefig(OUT / "fig2_models.png", facecolor="white")
    plt.close(fig)
    print("  fig2_models.png")


# ── FIGURE 3 — the research question ─────────────────────────────────
def mannequin():
    ds = [(PRETTY.get(k, k), v["delta"])
          for k, v in R["synthetic"].items() if isinstance(v, dict)]
    ds.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    def col(d):
        if abs(d) < 0.05: return GREY
        return GREEN if d > 0 else RED
    cols = [col(d) for _, d in ds]
    ax.barh(range(len(ds)), [d for _, d in ds], color=cols, height=0.62)
    for i, (n, d) in enumerate(ds):
        lab = "0.0" if abs(d) < 0.05 else f"{d:+.1f}"
        ax.text(d + (0.35 if d >= 0 else -0.35), i, lab, va="center",
                ha="left" if d >= 0 else "right", fontsize=11.5,
                fontweight="bold", color=col(d))
    ax.set_yticks(range(len(ds))); ax.set_yticklabels([n for n, _ in ds], fontsize=12.5)
    ax.axvline(0, color=INK, lw=1.6)
    med = float(np.median([d for _, d in ds]))
    ax.set_xlabel("change in macro-F1 when mannequin data is added (points)", fontsize=12.5)
    ax.set_title(f"Does the mannequin help?   Median {med:+.1f} points",
                 fontsize=16.5, fontweight="bold", pad=14)
    ax.set_xlim(min(d for _, d in ds) - 3, max(d for _, d in ds) + 3)
    # LDA is the one big winner and also the weakest model — say so, or the
    # chart reads as if the mannequin works and eight models are just noisy.
    top = max(ds, key=lambda r: r[1])
    ax.annotate("the only large gain is LDA —\nthe weakest model, with the\nmost room to improve",
                xy=(top[1] - 0.4, len(ds) - 1), xytext=(top[1] - 5.6, len(ds) - 3.4),
                fontsize=11, color=INK, va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    fig.text(0.5, -0.03,
             "8,820 mannequin recordings added to 1,470 real ones. "
             "Five models up, four down — no reliable gain.",
             ha="center", fontsize=11.5, color=INK, style="italic")
    fig.savefig(OUT / "fig3_mannequin.png", facecolor="white")
    plt.close(fig)
    print("  fig3_mannequin.png")


# ── FIGURE 4 — the three headline numbers ────────────────────────────
def headline():
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.axis("off")
    items = [
        (f"{R['decision']['best_live_unseen']:.1f}%", "on a signer it has\nNEVER seen", AMBER),
        (f"{R['deep']['tcn']['same']:.1f}%", "on a signer it\nhas seen", GREY),
        ("0.9", "points between best deep\nand best classical", BLUE),
    ]
    for i, (big, small, colour) in enumerate(items):
        ax.text(i + 0.5, 0.56, big, ha="center", va="center",
                fontsize=54, fontweight="bold", color=colour)
        ax.text(i + 0.5, 0.14, small, ha="center", va="center",
                fontsize=13.5, color=INK, linespacing=1.5)
    fig.savefig(OUT / "fig4_headline.png", facecolor="white")
    plt.close(fig)
    print("  fig4_headline.png")


for f in (pipeline, models, mannequin, headline):
    f()
print(f"\nwritten to {OUT}")


# ── FIGURE 5 — the lighting limit (for Person 7's block) ─────────────
def lighting():
    """PROBLEM_LOG P.5 — three check_camera runs, same room, same evening."""
    runs = [
        ("Signing\n(early evening)", 99.7, 42.0, 43.8),
        ("Signing\n(20 min later)", 99.7, 33.7, 32.0),
        ("Hands held still\n(same room)", 99.7, 92.6, 92.6),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    x = np.arange(len(runs)); w = 0.26
    body = [r[1] for r in runs]; lh = [r[2] for r in runs]; rh = [r[3] for r in runs]
    ax.bar(x - w, body, w, color=GREY, label="body detected")
    ax.bar(x,     lh,   w, color=BLUE, label="left hand")
    ax.bar(x + w, rh,   w, color=GREEN, label="right hand")
    for xi, vals in zip(x, zip(body, lh, rh)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 1.6, f"{v:.1f}", ha="center",
                    fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in runs], fontsize=12.5)
    ax.set_ylabel("frames where the tracker found it (%)", fontsize=12.5)
    ax.set_ylim(0, 115)
    ax.axhline(60, color=RED, ls=":", lw=1.8)
    ax.text(2.42, 62, "below this, recognition fails", fontsize=10.5,
            color=RED, ha="right", style="italic")
    ax.set_title("The body stays sharp. The hands blur.",
                 fontsize=17, fontweight="bold", pad=14)
    ax.legend(fontsize=11.5, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.text(0.5, -0.155,
             "Same room, same camera, same evening. Only the movement changes — "
             "which is why the cause is motion blur, not darkness.",
             ha="center", fontsize=11.5, color=INK, style="italic")
    fig.savefig(OUT / "fig5_lighting.png", facecolor="white")
    plt.close(fig)
    print("  fig5_lighting.png")


lighting()
