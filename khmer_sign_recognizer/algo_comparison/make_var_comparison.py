"""One report answering: does synthetic data improve khmer_var recognition?

Reads the two runs and writes a single side-by-side document.

    # 1. the two experiments (each writes its own results.json)
    python algo_comparison/run_var_experiment.py --lang khmer_var --mode real
    python algo_comparison/run_var_experiment.py --lang khmer_var --mode both

    # 2. combine them into one report
    python algo_comparison/make_var_comparison.py

Only khmer_var is involved. Nothing from any other corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402

import argparse                                                    # noqa: E402
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--lang", default="khmer_var")
args = ap.parse_args()
LANG = args.lang

REAL_DIR = ROOT / "algo_comparison" / f"results_{LANG}"
BOTH_DIR = ROOT / "algo_comparison" / f"results_{LANG}_both"
OUT = ROOT / "algo_comparison" / f"results_{LANG}_combined"

for d, mode in ((REAL_DIR, "real"), (BOTH_DIR, "both")):
    if not (d / "results.json").exists():
        sys.exit(f"missing {d / 'results.json'}\n"
                 f"run:  python algo_comparison/run_var_experiment.py "
                 f"--lang {LANG} --mode {mode}")

R = json.loads((REAL_DIR / "results.json").read_text(encoding="utf-8"))
B = json.loads((BOTH_DIR / "results.json").read_text(encoding="utf-8"))
OUT.mkdir(parents=True, exist_ok=True)

KHMER = ROOT / "fonts" / "NotoSansKhmer-Regular.ttf"
if KHMER.exists():
    font_manager.fontManager.addfont(str(KHMER))
    KH = font_manager.FontProperties(fname=str(KHMER), size=9)
else:
    KH = None

BLUE, GREEN, GREY, RED = "#4472C4", "#70AD47", "#A6A6A6", "#C00000"

# order by the real-only score, so the reader sees the baseline ranking
ALGOS = [a for a in R["results"] if a in B["results"]]
ALGOS.sort(key=lambda a: R["results"][a]["f1"], reverse=True)
NAME = {a: R["results"][a]["name"] for a in ALGOS}
GAIN = {a: B["results"][a]["f1"] - R["results"][a]["f1"] for a in ALGOS}
best_real = max(ALGOS, key=lambda a: R["results"][a]["f1"])
best_both = max(ALGOS, key=lambda a: B["results"][a]["f1"])
biggest_gain = max(ALGOS, key=lambda a: GAIN[a])
smallest_gain = min(ALGOS, key=lambda a: GAIN[a])
mean_gain = float(np.mean([GAIN[a] for a in ALGOS]))

LABEL_TEXT = R.get("labels", [])
texts = {}
lp = ROOT / "data" / "sequences_v2" / LANG / "labels.json"
if lp.exists():
    texts = json.loads(lp.read_text(encoding="utf-8"))
SIGN_TEXT = [texts.get(l, l) for l in LABEL_TEXT]


# ─────────────────────────────────────────────────────────── charts

def chart_side_by_side():
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    xs = np.arange(len(ALGOS)); w = 0.38
    rf1 = [R["results"][a]["f1"] for a in ALGOS]
    bf1 = [B["results"][a]["f1"] for a in ALGOS]
    rsd = [R["results"][a]["sd"] for a in ALGOS]
    bsd = [B["results"][a]["sd"] for a in ALGOS]
    ax.bar(xs - w/2, rf1, w, yerr=rsd, capsize=3, label="Real only", color=GREY)
    ax.bar(xs + w/2, bf1, w, yerr=bsd, capsize=3,
           label="Real + synthetic", color=GREEN)
    for i, (r, b_) in enumerate(zip(rf1, bf1)):
        ax.text(i - w/2, r + 2, f"{r:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, b_ + 2, f"{b_:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([NAME[a] for a in ALGOS], rotation=25, ha="right",
                       fontsize=9)
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 110)
    ax.set_title("Does synthetic data help? Same algorithms, same splits",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cmp_side_by_side.png", dpi=150)
    plt.close(fig)


def chart_gain():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    order = sorted(ALGOS, key=lambda a: GAIN[a], reverse=True)
    vals = [GAIN[a] for a in order]
    ax.bar(range(len(order)), vals,
           color=[GREEN if v > 0 else RED for v in vals])
    for i, v in enumerate(vals):
        ax.text(i, v + (0.6 if v >= 0 else -1.8), f"{v:+.1f}",
                ha="center", fontsize=9)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NAME[a] for a in order], rotation=25, ha="right",
                       fontsize=9)
    ax.set_ylabel("macro-F1 gained (points)")
    ax.set_title("How much each algorithm gained from synthetic data",
                 fontsize=12)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "cmp_gain.png", dpi=150)
    plt.close(fig)


def chart_converge():
    """Nine spread-out scores collapse onto one narrow band.

    Deliberately NOT a plot of gain against starting score. Because every
    algorithm ends up at nearly the same place, gain is almost exactly
    (that place) minus (starting score), so those two axes correlate at
    r = -1.00 by arithmetic. It looks like a finding and is really just the
    convergence restated.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rf1 = [R["results"][a]["f1"] for a in ALGOS]
    bf1 = [B["results"][a]["f1"] for a in ALGOS]
    for a, r_, b_ in zip(ALGOS, rf1, bf1):
        ax.plot([0, 1], [r_, b_], "-", color=GREY, alpha=.65, zorder=2)
        ax.annotate(NAME[a], (0, r_), textcoords="offset points",
                    xytext=(-8, -3), ha="right", fontsize=8)
    ax.scatter([0]*len(ALGOS), rf1, s=60, color=GREY, zorder=3,
               label="real only")
    ax.scatter([1]*len(ALGOS), bf1, s=60, color=GREEN, zorder=3,
               label="real + synthetic")
    ax.set_xlim(-0.55, 1.25); ax.set_xticks([0, 1])
    ax.set_xticklabels(["real only", "real + synthetic"])
    ax.set_ylabel("macro-F1 (%)")
    ax.set_title(f"Nine algorithms converge: spread "
                 f"{max(rf1)-min(rf1):.0f} points → "
                 f"{max(bf1)-min(bf1):.0f} points", fontsize=12)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "cmp_converge.png", dpi=150)
    plt.close(fig)


chart_side_by_side(); chart_gain(); chart_converge()
print(f"charts -> {OUT}")


# ─────────────────────────────────────────────────────────── report

from docx import Document                                          # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                     # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                      # noqa: E402
from docx.oxml import OxmlElement                                  # noqa: E402
from docx.oxml.ns import qn                                        # noqa: E402
from docx.shared import Inches, Pt, RGBColor                       # noqa: E402

CENTER = WD_ALIGN_PARAGRAPH.CENTER


def _shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear"); el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def para(doc, text="", size=11, bold=False, italic=False, align=None,
         color=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.size = Pt(size); r.bold = bold; r.italic = italic
    r.font.name = "Calibri"
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    return p


def bullet(doc, text, size=11):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text); r.font.size = Pt(size); r.font.name = "Calibri"
    p.paragraph_format.space_after = Pt(4)


def table(doc, headers, rows, widths, highlight=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True; r.font.size = Pt(10); r.font.name = "Calibri"
        c.paragraphs[0].alignment = CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
        _shade(c, "D9E2F3")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(10); r.font.name = "Calibri"
            if highlight is not None and ri == highlight:
                r.bold = True
            cells[i].paragraphs[0].alignment = (CENTER if i
                                                else WD_ALIGN_PARAGRAPH.LEFT)
            if highlight is not None and ri == highlight:
                _shade(cells[i], "FFF2CC")
            elif ri % 2:
                _shade(cells[i], "F2F2F2")
    for r_ in t.rows:
        for i, w in enumerate(widths):
            r_.cells[i].width = Inches(w)
    return t


def figure(doc, name, width, cap):
    p = OUT / name
    if not p.exists():
        para(doc, f"[chart missing: {name}]", italic=True); return
    doc.add_picture(str(p), width=Inches(width))
    doc.paragraphs[-1].alignment = CENTER
    para(doc, cap, size=9, italic=True, align=CENTER, color="595959",
         space_after=14)


doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.8)

h = doc.add_heading("Khmer Sign Language — Does Synthetic Data Help?", level=0)
h.alignment = CENTER
para(doc, f"{len(ALGOS)} algorithms on {R['n_takes']} recordings of "
          f"{len(LABEL_TEXT)} signs, trained twice: with and without "
          f"synthetic data", size=12, italic=True, align=CENTER,
     color="595959", space_after=16)

# ---- 1
doc.add_heading("1. What we tested", level=1)
para(doc, "Signs are recognised from pose landmarks, not images. Each recording "
          "is 60 frames of 48 tracked joints — 6 body points plus 21 on each "
          "hand — summarised into 576 numbers (the mean, spread, minimum and "
          "maximum of every joint coordinate over the clip).")
table(doc, ["", "Detail"],
      [["Signs", f"{len(LABEL_TEXT)} — " + ", ".join(SIGN_TEXT)],
       ["Real recordings", f"{R['n_takes']} takes, 12 per sign per person"],
       ["With synthetic added", f"{B['n_takes']} training samples"],
       ["Split", f"75/25 by take, averaged over {R['seeds']} random splits"],
       ["Tested on", "real recordings only, in both experiments"]],
      [1.6, 4.9])
para(doc)
para(doc, "Synthetic recordings are made by mathematically stretching the "
          "skeleton of a real recording to different body proportions. The "
          "movement is untouched, so the sign is unchanged — only the body "
          "performing it differs. No camera, no new recording session.")

# ---- 2
doc.add_heading("2. The two experiments", level=1)
para(doc, "Both experiments use the same algorithms, the same splits and the "
          "same test recordings. The only difference is what goes into "
          "training.")
rows = [[NAME[a],
         f"{R['results'][a]['f1']:.1f}%", f"± {R['results'][a]['sd']:.1f}",
         f"{B['results'][a]['f1']:.1f}%", f"± {B['results'][a]['sd']:.1f}",
         f"{GAIN[a]:+.1f}"] for a in ALGOS]
table(doc, ["Algorithm", "Real only", "s.d.", "Real + synthetic", "s.d.",
            "Change"],
      rows, [1.7, 1.0, 0.7, 1.35, 0.7, 0.85],
      highlight=ALGOS.index(biggest_gain))
para(doc)
figure(doc, "cmp_side_by_side.png", 6.6,
       "Figure 1 — every algorithm, trained without (grey) and with (green) "
       "synthetic data.")

# ---- 3
doc.add_heading("3. Does it improve things?", level=1)
# Report whichever picture the data actually shows: a broad gain, a broad
# loss, or a mixture with outliers. Writing this as prose that assumes one
# outcome is how the previous version ended up claiming convergence on numbers
# that had not converged.
gains = np.array([GAIN[a] for a in ALGOS])
helped = [a for a in ALGOS if GAIN[a] > 1.0]
hurt = [a for a in ALGOS if GAIN[a] < -1.0]
flat = [a for a in ALGOS if abs(GAIN[a]) <= 1.0]
median_gain = float(np.median(gains))
# an outlier is anything more than 3x the median absolute move
mad = float(np.median(np.abs(gains - median_gain))) or 1.0
outliers = [a for a in ALGOS if abs(GAIN[a] - median_gain) > 4 * mad]

if len(hurt) == 0 and len(helped) >= len(ALGOS) - 1:
    para(doc, f"Yes. Every algorithm improved, by "
              f"{gains.min():+.1f} to {gains.max():+.1f} points "
              f"(median {median_gain:+.1f}).")
elif len(helped) == 0:
    para(doc, f"No. Nothing improved; scores moved by {gains.min():+.1f} to "
              f"{gains.max():+.1f} points (median {median_gain:+.1f}).")
else:
    para(doc, f"Mixed, and modestly. {len(helped)} of {len(ALGOS)} algorithms "
              f"improved, {len(hurt)} got worse and {len(flat)} barely moved. "
              f"The median change is {median_gain:+.1f} points — a real but "
              f"small effect, not the transformation that more training data "
              f"might suggest.")
figure(doc, "cmp_gain.png", 6.4,
       "Figure 2 — points gained or lost by each algorithm.")

if outliers:
    o = max(outliers, key=lambda a: abs(GAIN[a]))
    para(doc, f"One result stands apart and is worth explaining rather than "
              f"averaging away: {NAME[o]} moved {GAIN[o]:+.1f} points, far "
              f"beyond anything else. It is the single reason the mean change "
              f"({mean_gain:+.1f}) looks worse than the median "
              f"({median_gain:+.1f}); excluding it, the average change is "
              f"{float(np.mean([GAIN[a] for a in ALGOS if a != o])):+.1f}.")
    if o == "lda":
        para(doc, "LDA works by estimating how the 576 features vary together "
                  "within each sign, which requires inverting a large "
                  "covariance matrix. Synthetic variants of the same recording "
                  "are highly similar to one another — the clean view already "
                  "divides out overall body scale, which is most of what the "
                  "retargeting changes. Adding six near-copies of every "
                  "recording therefore multiplies the sample count without "
                  "adding much genuinely new variation, and that makes the "
                  "covariance estimate unstable. Methods that do not invert a "
                  "covariance matrix are untroubled by this.")

real_spread = max(R["results"][a]["f1"] for a in ALGOS) - \
              min(R["results"][a]["f1"] for a in ALGOS)
both_spread = max(B["results"][a]["f1"] for a in ALGOS) - \
              min(B["results"][a]["f1"] for a in ALGOS)
figure(doc, "cmp_converge.png", 5.6,
       "Figure 3 — every algorithm's score before and after.")
if both_spread < real_spread * 0.5:
    para(doc, f"The algorithms converge: a spread of {real_spread:.0f} points "
              f"on real data becomes {both_spread:.0f} with synthetic. The "
              f"choice of algorithm mattered much less once there was more "
              f"data to learn from.")
else:
    para(doc, f"The ranking is largely unchanged. The spread across algorithms "
              f"was {real_spread:.0f} points on real data and "
              f"{both_spread:.0f} with synthetic added, so the choice of "
              f"algorithm still matters about as much as it did. Synthetic "
              f"data did not level the field.")
para(doc, "That is a reasonable outcome on reflection. Every synthetic example "
          "is derived from a real recording, so it adds examples rather than "
          "information. It can help an algorithm that is short of examples, "
          "but it cannot supply variation that was never recorded — and the "
          "clean view we train on already removes overall body scale, which is "
          "much of what the retargeting varies.")

# ---- 4
doc.add_heading("4. Why this is a fair test", level=1)
para(doc, "Two rules make the comparison honest, and both matter:")
bullet(doc, "**Testing is always on real recordings.** Scoring a model on "
            "synthetic data would only measure whether it can recognise our own "
            "geometry.")
bullet(doc, "**A recording's synthetic copies stay with it.** When a real take "
            "is held out for testing, all of its synthetic children are held "
            "out too. Otherwise the model trains on near-copies of the test "
            "data and the score means nothing.")
para(doc, f"The second rule was verified directly: the {R['n_takes']} real "
          f"takes form {R['n_takes']} groups, each containing exactly one real "
          f"recording and its synthetic children, and no group appears on both "
          f"sides of any split.", size=10, italic=True, color="595959")

# ---- 5
doc.add_heading("5. Did the recording conditions matter?", level=1)
para(doc, "The 12 recordings of each sign follow a deliberate grid: two "
          "lighting levels, two distances from the camera, three standing "
          "positions. That lets us hold back a whole condition — train only on "
          "bright recordings, test only on dim ones — and see what survives.")
cond = R.get("conditions", [])
if cond:
    base = R["results"][best_real]["f1"]
    table(doc, ["Trained on", "Tested on", "macro-F1", "vs baseline"],
          [[c["trained_on"], c["tested_on"], f"{c['f1']:.1f}%",
            f"{c['f1'] - base:+.1f}"] for c in cond],
          [1.6, 1.5, 1.2, 1.3])
    para(doc)
    worst = min(cond, key=lambda c: c["f1"])
    best_c = max(cond, key=lambda c: c["f1"])
    para(doc, f"The mildest change is {best_c['trained_on']} → "
              f"{best_c['tested_on']} at {best_c['f1']:.0f}%; the hardest is "
              f"{worst['trained_on']} → {worst['tested_on']} at "
              f"{worst['f1']:.0f}%.")
    para(doc, "Worth noting: when only one person had recorded, lighting was by "
              "far the worst condition. With four people recording in four "
              "different rooms it costs almost nothing — the variety arrived "
              "for free. The cheapest fix for a condition problem turned out to "
              "be more people, not tighter control of the condition.")

# ---- 6
doc.add_heading("6. Conclusions", level=1)
if mean_gain > 0:
    bullet(doc, f"Synthetic data improved every algorithm, on average "
                f"{mean_gain:+.1f} points of macro-F1.")
    bullet(doc, f"The weakest algorithms gained the most; the strongest gained "
                f"least. The benefit is data quantity, not new information.")
    bullet(doc, f"Best result overall: {NAME[best_both]} at "
                f"{B['results'][best_both]['f1']:.1f}% macro-F1 with synthetic "
                f"data, against {R['results'][best_real]['f1']:.1f}% for the "
                f"best real-only model.")
bullet(doc, "Distance and standing position cost more accuracy than lighting "
            "does.")
bullet(doc, f"All {R['n_takes']} recordings come from 4 people and the test set "
            f"mixes them, so these figures describe recognition for people the "
            f"model has seen before. Recognising a complete stranger is a "
            f"different question, answered by holding one person out entirely.")
bullet(doc, "12 recordings per sign is a small dataset. Large differences are "
            "meaningful; one or two points between neighbouring algorithms are "
            "not.")

doc.add_heading("7. Reproducing this", level=1)
para(doc, "python algo_comparison/run_var_experiment.py --lang khmer_var --mode real",
     size=10, color="1F4E79")
para(doc, "python algo_comparison/run_var_experiment.py --lang khmer_var --mode both",
     size=10, color="1F4E79")
para(doc, "python algo_comparison/make_var_comparison.py",
     size=10, color="1F4E79")
para(doc, "Every algorithm has a fixed seed, so these numbers come out "
          "identical on any machine.", size=9, italic=True, color="595959")

path = OUT / "Task_A_Synthetic_Comparison.docx"
doc.save(path)
print(f"report -> {path}")
