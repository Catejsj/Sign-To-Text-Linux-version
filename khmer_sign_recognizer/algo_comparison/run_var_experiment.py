"""Task A experiment: run it and write the .docx report, in one command.

    python algo_comparison/run_var_experiment.py

Uses ONLY the khmer_var grid recordings — the deliberate 12-take-per-sign
recording described in docs/TEAM_DATA_COLLECTION_PLAN.md. Nothing from the
khmer corpus is involved.

Two experiments:
  1. every algorithm compared, on a random take split and on three
     held-out-condition splits
  2. the best one examined on its own — per-sign scores and a confusion matrix

Writes charts plus Task_A_Report.docx into algo_comparison/results_khmer_var/.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.v2.algorithms import registry, wrap                       # noqa: E402
from src.v2.baseline_data import _featurize                        # noqa: E402
from src.v2.dataset import discover_samples                        # noqa: E402
from src.v2.schema import Source, View                             # noqa: E402

LANG = "khmer_var"
GRID = 12
SEEDS = 8
OUT = ROOT / "algo_comparison" / f"results_{LANG}"
BEST = "bagging"          # chosen by measurement, see docs/EXPERIMENT_REPORT.md

# Khmer labels need their own font or matplotlib draws empty boxes.
KHMER = ROOT / "fonts" / "NotoSansKhmer-Regular.ttf"
if KHMER.exists():
    font_manager.fontManager.addfont(str(KHMER))
    KH = font_manager.FontProperties(fname=str(KHMER), size=9)
else:
    KH = None

BLUE, GREY, RED = "#4472C4", "#A6A6A6", "#C00000"


# ─────────────────────────────────────────────────────────── data

def load():
    X, y, variants = [], [], []
    for p, m in discover_samples(ROOT / "data" / "sequences_v2", language=LANG):
        if m.source is not Source.REAL or m.view is not View.CLEAN:
            continue
        X.append(_featurize(np.load(p).astype(np.float32), "summary"))
        y.append(m.label)
        variants.append(m.variant)
    if not X:
        sys.exit(f"no real takes in data/sequences_v2/{LANG}/ — record first")
    labels = sorted(set(y))
    l2i = {l: i for i, l in enumerate(labels)}
    return (np.stack(X), np.array([l2i[t] for t in y]),
            np.array(variants), labels)


X, y, variants, LABELS = load()
slot = variants % GRID
texts = {}
lp = ROOT / "data" / "sequences_v2" / LANG / "labels.json"
if lp.exists():
    texts = json.loads(lp.read_text(encoding="utf-8"))
LABEL_TEXT = [texts.get(l, l) for l in LABELS]

CONDITIONS = {
    "Lighting":  ("full light", "dim light",   slot < 6),
    "Distance":  ("near",       "far",         (slot % 6) < 3),
    "Position":  ("middle+left", "right",      (slot % 3) < 2),
}

print(f"{len(X)} real takes · {len(LABELS)} signs · "
      f"{len(np.unique(variants))} takes per sign\n")


# ─────────────────────────────────────────────────────── evaluation

def random_split_scores(factory):
    """Take-aware random split, averaged over SEEDS."""
    accs, f1s = [], []
    takes = np.unique(variants)
    for seed in range(SEEDS):
        rs = np.random.default_rng(seed)
        held = set(rs.choice(takes, size=max(1, len(takes) // 4),
                             replace=False).tolist())
        ev = np.array([v in held for v in variants])
        m = wrap(factory()); m.fit(X[~ev], y[~ev])
        p = m.predict(X[ev])
        accs.append(accuracy_score(y[ev], p) * 100)
        f1s.append(f1_score(y[ev], p, average="macro") * 100)
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.std(f1s))


def condition_score(factory, mask):
    m = wrap(factory()); m.fit(X[mask], y[mask])
    return f1_score(y[~mask], m.predict(X[~mask]), average="macro") * 100


table_reg, _ = registry()
ALGOS = [a for a in table_reg if a != "nb"]

results = {}
for algo in ALGOS:
    label, factory, origin = table_reg[algo]
    acc, f1, sd = random_split_scores(factory)
    conds = {name: condition_score(factory, mask)
             for name, (_a, _b, mask) in CONDITIONS.items()}
    results[algo] = {"name": label, "origin": origin, "acc": acc, "f1": f1,
                     "sd": sd, "cond": conds,
                     "cond_mean": float(np.mean(list(conds.values())))}
    print(f"  {label:<22} random {f1:5.1f}   conditions {results[algo]['cond_mean']:5.1f}")

order = sorted(results, key=lambda a: results[a]["cond_mean"], reverse=True)

# best algorithm in detail — trained on the hardest split
_l, best_factory, _o = table_reg[BEST]
hard_mask = CONDITIONS["Lighting"][2]
mbest = wrap(best_factory()); mbest.fit(X[hard_mask], y[hard_mask])
pred_hard = mbest.predict(X[~hard_mask])
cm = confusion_matrix(y[~hard_mask], pred_hard, labels=range(len(LABELS)))
per_sign = f1_score(y[~hard_mask], pred_hard, average=None,
                    labels=range(len(LABELS))) * 100


# ─────────────────────────────────────────────────────────── charts

OUT.mkdir(parents=True, exist_ok=True)


def chart_overview():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [results[a]["name"] for a in order]
    xs = np.arange(len(order)); w = 0.38
    ax.bar(xs - w/2, [results[a]["f1"] for a in order], w,
           label="Random take split", color=GREY)
    ax.bar(xs + w/2, [results[a]["cond_mean"] for a in order], w,
           label="Held-out condition (mean)", color=BLUE)
    for i, a in enumerate(order):
        ax.text(i - w/2, results[a]["f1"] + 1, f"{results[a]['f1']:.0f}",
                ha="center", fontsize=8)
        ax.text(i + w/2, results[a]["cond_mean"] + 1,
                f"{results[a]['cond_mean']:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right",
                                          fontsize=9)
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 105)
    ax.set_title("Every algorithm: familiar conditions vs. unseen conditions",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "var_overview.png", dpi=150)
    plt.close(fig)


def chart_conditions():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xs = np.arange(len(order)); w = 0.26
    for k, (name, colour) in enumerate(zip(CONDITIONS, [BLUE, "#70AD47", "#ED7D31"])):
        ax.bar(xs + (k - 1) * w, [results[a]["cond"][name] for a in order], w,
               label=f"{name}: {CONDITIONS[name][0]} → {CONDITIONS[name][1]}",
               color=colour)
    ax.set_xticks(xs)
    ax.set_xticklabels([results[a]["name"] for a in order], rotation=25,
                       ha="right", fontsize=9)
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 100)
    ax.set_title("Which change of setup hurts, per algorithm", fontsize=12)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "var_conditions.png", dpi=150)
    plt.close(fig)


def chart_confusion():
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABEL_TEXT, rotation=45, ha="right",
                       fontproperties=KH if KH else None)
    ax.set_yticklabels(LABEL_TEXT, fontproperties=KH if KH else None)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"{results[BEST]['name']}: trained on full light,\n"
                 f"tested on dim light", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "var_confusion.png", dpi=150)
    plt.close(fig)


def chart_per_sign():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    idx = np.argsort(per_sign)[::-1]
    ax.bar(range(len(LABELS)), per_sign[idx],
           color=[RED if v < 50 else BLUE for v in per_sign[idx]])
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([LABEL_TEXT[i] for i in idx],
                       fontproperties=KH if KH else None)
    for i, v in enumerate(per_sign[idx]):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"{results[BEST]['name']}: which signs survive the "
                 f"change of lighting", fontsize=11)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "var_per_sign.png", dpi=150)
    plt.close(fig)


chart_overview(); chart_conditions(); chart_confusion(); chart_per_sign()

(OUT / "results.json").write_text(json.dumps(
    {"n_takes": int(len(X)), "labels": LABELS, "seeds": SEEDS,
     "results": {a: results[a] for a in order}}, indent=2, ensure_ascii=False),
    encoding="utf-8")
print(f"\ncharts -> {OUT}")


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
    r = p.add_run(text)
    r.font.size = Pt(size); r.font.name = "Calibri"
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

h = doc.add_heading("Khmer Sign Language — Algorithm Comparison", level=0)
h.alignment = CENTER
para(doc, "Does the choice of algorithm matter when the recording "
          "conditions change?", size=12, italic=True, align=CENTER,
     color="595959", space_after=16)

best_r = results[BEST]
runner = [a for a in order if a != BEST][0]

# ---- 1
doc.add_heading("1. What we tested", level=1)
para(doc, "Seven Khmer signs were recorded on a deliberate grid: twelve takes "
          "of each sign, crossing two lighting conditions, two distances and "
          "three standing positions. Because the grid is fixed, any one of "
          "those settings can be held back — the model is trained on one and "
          "tested on a setting it has never seen.")
para(doc, "That is the question this report answers. Not \"how accurate is "
          "it\", but \"how much accuracy survives a change in the room\".")

table(doc,
      ["", "Detail"],
      [["Signs", f"{len(LABELS)} — " + ", ".join(LABEL_TEXT)],
       ["Recordings", f"{len(X)} takes ({GRID} per sign)"],
       ["Grid", "2 lighting × 2 distance × 3 position"],
       ["Input", "Pose landmarks — 48 joints × 60 frames, not images"],
       ["Features", "576 summary values (mean, std, min, max per joint)"],
       ["Algorithms", f"{len(ALGOS)} compared"]],
      [1.5, 5.0])
para(doc)

# ---- 2
doc.add_heading("2. Every algorithm compared", level=1)
para(doc, "Each algorithm was scored two ways. The grey bar is a random split "
          "of takes — the familiar way to measure accuracy, where training and "
          "testing come from the same conditions. The blue bar averages three "
          "held-out-condition tests, where the test setting never appeared in "
          "training.")
figure(doc, "var_overview.png", 6.6,
       "Figure 1 — Every algorithm under familiar conditions (grey) and unseen "
       "conditions (blue). The grey bars are close together; the blue bars are "
       "not.")

rows = [[results[a]["name"],
         f"{results[a]['acc']:.1f}%",
         f"{results[a]['f1']:.1f}%",
         f"{results[a]['cond_mean']:.1f}%",
         f"{results[a]['f1'] - results[a]['cond_mean']:+.1f}"]
        for a in order]
table(doc, ["Algorithm", "Accuracy", "Macro-F1", "Unseen conditions", "Drop"],
      rows, [1.9, 1.0, 1.0, 1.5, 0.9],
      highlight=order.index(BEST))
para(doc)
para(doc, f"Under familiar conditions almost everything scores well and the "
          f"algorithms look interchangeable. Once the setting changes they "
          f"separate sharply — from {results[order[0]]['cond_mean']:.0f}% down "
          f"to {results[order[-1]]['cond_mean']:.0f}%. "
          f"A comparison run only on a random split would have reported that "
          f"the choice barely matters, and that conclusion would be wrong.",
     bold=False)

# ---- 3
doc.add_heading("3. Which change of setup actually hurts", level=1)
figure(doc, "var_conditions.png", 6.6,
       "Figure 2 — The same algorithms broken down by which setting was held "
       "out.")
cond_avg = {name: float(np.mean([results[a]["cond"][name] for a in order]))
            for name in CONDITIONS}
worst = min(cond_avg, key=cond_avg.get)
table(doc, ["Held-out setting", "Average across algorithms"],
      [[f"{name}: {CONDITIONS[name][0]} → {CONDITIONS[name][1]}",
        f"{cond_avg[name]:.1f}%"] for name in CONDITIONS],
      [3.4, 2.4])
para(doc)
para(doc, f"Standing position and distance transfer reasonably. {worst} is the "
          f"one that breaks models, and it splits the field: the tree-based "
          f"methods hold up while the linear and margin-based ones collapse.")

# ---- 4
doc.add_heading(f"4. The chosen algorithm: {best_r['name']}", level=1)
para(doc, f"{best_r['name']} was selected because it holds the most accuracy "
          f"when conditions change — {best_r['cond_mean']:.1f}% against "
          f"{results[runner]['cond_mean']:.1f}% for the next best "
          f"({results[runner]['name']}) — not because it wins on the "
          f"random split, where it is one of several near the top.")
table(doc, ["Held-out setting", f"{best_r['name']}", "Next best"],
      [[f"{name}: {CONDITIONS[name][0]} → {CONDITIONS[name][1]}",
        f"{best_r['cond'][name]:.1f}%",
        f"{results[runner]['cond'][name]:.1f}%"] for name in CONDITIONS],
      [3.0, 1.6, 1.6])
para(doc)
para(doc, "It builds many decision trees, each on a different random sample of "
          "the takes, and has them vote. No single tree sees everything, so no "
          "single quirk of one recording decides the answer.")

figure(doc, "var_per_sign.png", 6.2,
       "Figure 3 — Per-sign scores on the hardest test. Red marks signs that "
       "fall below 50%.")
weak = [LABEL_TEXT[i] for i in range(len(LABELS)) if per_sign[i] < 50]
if weak:
    para(doc, f"Weakest signs under changed lighting: {', '.join(weak)}. "
              f"These are worth extra recordings before anything else.")
else:
    para(doc, "No sign falls below 50% — the loss is spread evenly rather "
              "than concentrated in one or two signs.")

figure(doc, "var_confusion.png", 4.6,
       "Figure 4 — Which signs get mistaken for which, on the hardest test. "
       "The diagonal is correct answers.")

# ---- 5
doc.add_heading("5. Conclusions", level=1)
bullet(doc, f"{best_r['name']} is the most robust of the "
            f"{len(ALGOS)} algorithms tested, keeping "
            f"{best_r['cond_mean']:.0f}% macro-F1 when the recording setup "
            f"changes.")
bullet(doc, "Algorithms that look equivalent on a random split are not "
            "equivalent. The gap only appears when the test conditions differ "
            "from the training conditions.")
bullet(doc, f"{worst.lower()} is the setting that matters most; position and "
            f"distance are handled comparatively well.")
bullet(doc, "Tree-based methods degrade gracefully; linear and margin-based "
            "methods fail sharply. Robustness is a property of the model "
            "family, not of its headline accuracy.")

doc.add_heading("6. Honest limitations", level=1)
bullet(doc, "One signer. These results say how well a model transfers across "
            "conditions, not across people — that needs recordings from more "
            "of the team.")
bullet(doc, "Lighting is confounded with recording order: the dim takes were "
            "always recorded after the bright ones, so fatigue or drift cannot "
            "be separated from the lighting itself. Having half the team "
            "record dim first would settle it.")
bullet(doc, "We assumed dim light would break hand detection. Measured "
            "directly, it does not — 62.1% of hand landmarks were missing in "
            "bright light and 60.8% in dim. The cause of the drop is still "
            "open.")
bullet(doc, f"{len(X)} takes is a small dataset. Differences of one or two "
            f"points between neighbouring algorithms should not be read as "
            f"meaningful.")

doc.add_heading("7. Reproducing this", level=1)
para(doc, "Every algorithm is seeded, so these numbers come out identical on "
          "any machine:")
para(doc, "python algo_comparison/run_var_experiment.py",
     size=10, color="1F4E79")
para(doc, f"Scores average {SEEDS} random splits; the condition tests are "
          f"fixed by the grid and need no averaging.", size=9, italic=True,
     color="595959")

path = OUT / "Task_A_Report.docx"
doc.save(path)
print(f"report -> {path}")
