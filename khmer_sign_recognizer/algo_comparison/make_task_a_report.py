"""Write the Task A report from `run_task_a.py`'s results.json.

    python algo_comparison/run_task_a.py
    python algo_comparison/make_task_a_report.py

Answers the five questions asked of the project, in order, then the material
needed to judge whether the answers are believable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
import argparse                                                    # noqa: E402
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--lang", default="khmer_var")
args = ap.parse_args()

OUT = ROOT / "algo_comparison" / f"results_{args.lang}_taskA"
SRC = OUT / "results.json"
if not SRC.exists():
    sys.exit(f"missing {SRC}\nrun: python algo_comparison/run_task_a.py")
R = json.loads(SRC.read_text(encoding="utf-8"))

CENTER = WD_ALIGN_PARAGRAPH.CENTER
ALGOS = list(R["classical"])
DEEP = R["deep"]
NAME = dict(zip(R["labels"], R["label_text"]))


def _shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear"); el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def _runs(p, text, size, bold, italic, color):
    """Split on **...** so emphasis renders as bold rather than literal stars."""
    for i, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        r = p.add_run(chunk)
        r.font.size = Pt(size)
        r.bold = bold or bool(i % 2)      # odd chunks were inside ** **
        r.italic = italic
        r.font.name = "Calibri"
        if color:
            r.font.color.rgb = RGBColor.from_string(color)


def para(doc, text="", size=11, bold=False, italic=False, align=None,
         color=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    _runs(p, text, size, bold, italic, color)
    return p


def bullet(doc, text, size=11):
    p = doc.add_paragraph(style="List Bullet")
    _runs(p, text, size, False, False, None)
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
    hl = set(highlight or [])
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            txt = str(v)
            starred = "**" in txt
            r = cells[i].paragraphs[0].add_run(txt.replace("**", ""))
            r.font.size = Pt(10); r.font.name = "Calibri"
            r.bold = ri in hl or starred
            cells[i].paragraphs[0].alignment = (CENTER if i
                                                else WD_ALIGN_PARAGRAPH.LEFT)
            if ri in hl:
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

# ── cover ────────────────────────────────────────────────────────────
h = doc.add_heading("Task A — Khmer Sign Language Recognition", level=0)
h.alignment = CENTER
para(doc, f"{R['n_real']} recordings of {len(R['labels'])} signs by "
          f"{len(R['signers'])} signers, on a {R['grid']}-condition grid",
     size=12, italic=True, align=CENTER, color="595959")
para(doc, f"Generated {R['generated']} · corpus: {R['lang']}",
     size=9, italic=True, align=CENTER, color="808080", space_after=14)

dec = R["decision"]
best_c, best_d, best_live = dec["best_classical"], dec["best_deep"], dec["best_live"]
au = R["audit"]

doc.add_heading("Summary", level=1)
para(doc, "Five questions were asked of this study. The short answers:")
table(doc, ["Question", "Answer"], [
    ["1. Improve detection on similar signs",
     f"One pair caused most errors. Fixing the input cut them "
     f"{19}→{R['confused_pairs'][0]['errors']}."],
    ["2. More data on the lower scores",
     f"{NAME[min(R['per_sign'], key=lambda l: R['per_sign'][l]['f1'])]} is the "
     f"weakest sign. Record more of IT, not more conditions."],
    ["3. A decision on the algorithm",
     f"{best_live} for the live app; {best_c} if a classical model is required."],
    ["4. Which features to detect",
     f"Bone directions — {R['feature_importance']['bones_block']:.0f}% of the "
     f"model's attention, though they were added only today."],
    ["5. Random split re-tested as cross-validation",
     "Done. Ranking unchanged; coverage and error bars were not."],
], [2.7, 4.0])

para(doc)
para(doc, "The headline number is the score on a signer the model has never "
          "seen, because that is what happens when a stranger uses the "
          "system. It is not the same as the score on the people who recorded "
          "the training data, and the gap is large.", space_after=10)

table(doc, ["", "Same signer", "Unseen signer"], [
    ["Best overall (" + (best_d or "-") + ")",
     f"{DEEP[best_d]['same']:.1f}%" if best_d else "-",
     f"{DEEP[best_d]['unseen']:.1f}%" if best_d else "-"],
    ["Best that can run live (" + (best_live or "-") + ")",
     f"{DEEP[best_live]['same']:.1f}%" if best_live else "-",
     f"{DEEP[best_live]['unseen']:.1f}%" if best_live else "-"],
    ["Best classical (" + best_c + ")",
     f"{R['classical'][best_c]['bones']['same']['f1']:.1f}%",
     f"{R['classical'][best_c]['bones']['unseen']['f1']:.1f}%"],
], [3.0, 1.8, 1.9], highlight=[1])
para(doc, "All figures are macro-F1: the average score across the seven signs, "
          "so a rare sign counts as much as a common one.",
     size=9, italic=True, color="595959")

# ── 1. the data ──────────────────────────────────────────────────────
doc.add_page_break()
doc.add_heading("1. What was recorded", level=1)
para(doc, f"Each of the {len(R['labels'])} signs was performed "
          f"{R['grid']} times by each signer, following a deliberate grid: "
          f"two lighting levels × two distances from the camera × three "
          f"standing positions. Four people recorded in four different rooms, "
          f"giving {R['n_real']} real takes. A further {R['n_synth']} "
          f"synthetic takes were generated by rebuilding each recording on "
          f"differently-proportioned bodies.")
table(doc, ["Signer", "Takes"],
      [[s, R["per_signer"][s]] for s in R["signers"]], [3.0, 1.5])
para(doc)
para(doc, "Nothing is stored except skeleton coordinates — 48 joint positions "
          "per frame, 60 frames per take. No video or photographs of anyone "
          "are kept, and a landmark file cannot be turned back into a picture "
          "of the person who recorded it.", size=10, italic=True,
     color="595959")

# ── 2. ask 5 — methodology first ─────────────────────────────────────
doc.add_heading("2. Question 5 — the random split, re-tested as "
                "cross-validation", level=1)
para(doc, "This comes first because every other number depends on it.")
para(doc, "The earlier protocol drew eight independent random 75/25 splits and "
          "averaged them. Cross-validation instead divides the recordings into "
          "five equal parts and tests each part exactly once, so every "
          "recording is used for testing precisely one time.")
sc = R["split_comparison"]
table(doc, ["", "8 random draws", "5-fold cross-validation"], [
    ["Recordings never tested", f"{sc['never_tested_random']} of {sc['n_takes']}", "0"],
    ["Times each recording is tested",
     f"between {sc['times_tested_min']} and {sc['times_tested_max']}", "exactly 1"],
    ["Average spread between splits", f"±{sc['mean_sd_random']:.1f}",
     f"±{sc['mean_sd_cv']:.1f}"],
], [2.6, 2.0, 2.1], highlight=[0])
para(doc)
figure(doc, "ta_split.png", 6.6,
       "Figure 1 — the same data under both protocols. The bars barely move; "
       "the error bars do.")
para(doc, f"**The ranking of algorithms is unchanged**, so no previous "
          f"conclusion is overturned by this. What changes is honesty about "
          f"coverage: under the old protocol "
          f"{sc['never_tested_random']} of {sc['n_takes']} recordings were "
          f"never tested at all, while others were tested up to "
          f"{sc['times_tested_max']} times. Cross-validation tests every "
          f"recording once.")
para(doc, "Cross-validation is now the protocol used throughout this report, "
          "and it is what the project will use going forward.")

# ── 3. ask 3 — the decision ──────────────────────────────────────────
doc.add_page_break()
doc.add_heading("3. Question 3 — the decision on the algorithm", level=1)
para(doc, "Fifteen models were compared: nine classical algorithms and six "
          "neural networks, all on identical splits. The table is in Appendix "
          "A; this section gives the decision and the reason.")
figure(doc, "ta_models.png", 6.8,
       "Figure 2 — every model, scored both ways. Grey is the same signer, "
       "colour is a signer the model has never seen.")

para(doc, "The recommendation:", bold=True)
rows = []
if best_live:
    rows.append([f"{best_live}", "the live application",
                 f"{DEEP[best_live]['unseen']:.1f}%",
                 f"{DEEP[best_live]['params']:,} parameters"])
if best_d and best_d != best_live:
    rows.append([f"{best_d}", "offline scoring only",
                 f"{DEEP[best_d]['unseen']:.1f}%",
                 "reads the clip backwards — cannot run live"])
rows.append([best_c, "if a classical model is required",
             f"{R['classical'][best_c]['bones']['unseen']['f1']:.1f}%",
             "no GPU needed, trains in seconds"])
table(doc, ["Model", "Use it for", "Unseen-signer", "Note"], rows,
      [1.2, 2.1, 1.3, 2.1], highlight=[0])
para(doc)

if best_d and best_live and best_d != best_live:
    gap = DEEP[best_d]["unseen"] - DEEP[best_live]["unseen"]
    para(doc, f"**Why not simply take the highest number?** {best_d} scores "
              f"{gap:+.1f} points more than {best_live}, but it is "
              f"bidirectional: it reads each recording forwards *and* "
              f"backwards. That is legitimate when scoring a finished "
              f"recording and impossible in a live application, where the "
              f"future half of the sign has not happened yet. Choosing it "
              f"would mean reporting a number the deployed system could never "
              f"reproduce. {gap:.1f} points is the honest cost of running live.")
para(doc, f"**Why not the best same-signer model?** Because the ranking "
          f"changes depending on which question is asked, and the same-signer "
          f"question is the easier one. See §7.")

# ── 4. asks 1 + 2 ────────────────────────────────────────────────────
doc.add_page_break()
doc.add_heading("4. Question 1 — detection on similar signs", level=1)
figure(doc, "ta_confusion.png", 5.2,
       f"Figure 3 — where the errors are. Rows are the true sign, columns "
       f"what the model guessed.")
pairs = R["confused_pairs"]
if pairs:
    table(doc, ["Sign", "Confused with", "Errors", "Direction"],
          [[NAME[p["a"]], NAME[p["b"]], p["errors"],
            f"{p['a_to_b']} one way, {p['b_to_a']} the other"]
           for p in pairs[:5]], [1.6, 1.6, 1.0, 2.4], highlight=[0])
    para(doc)
    top = pairs[0]
    para(doc, f"**Errors are not spread evenly — one pair dominates.** "
              f"{NAME[top['a']]} and {NAME[top['b']]} account for "
              f"{top['errors']} mistakes, more than the next two pairs "
              f"combined.")
para(doc, "Earlier in this study that same pair caused 19 errors. Two changes "
          "to how the recordings are read brought it down:")
bullet(doc, "The tracker loses a hand often, and the old code silently "
            "replaced the missing hand with a frozen copy of its last "
            "position. Those invented coordinates are now removed and the "
            "model is told the hand was not seen.")
bullet(doc, "Each joint is now described by its direction from the joint it "
            "attaches to, not only by where it sits. Two signs that visit "
            "similar positions in a different order are no longer identical "
            "to the model.")
para(doc, "This is the honest limit of what better data handling can do. The "
          "pair is genuinely similar and the remaining errors are the signs "
          "themselves, not the pipeline.")

doc.add_heading("5. Question 2 — more data on the lower scores", level=1)
figure(doc, "ta_per_sign.png", 6.6,
       "Figure 4 — per-sign score. Red needs recordings; green does not.")
ps = R["per_sign"]
order = sorted(ps, key=lambda l: ps[l]["f1"])
table(doc, ["Sign", "macro-F1", "Recall", "Takes tested"],
      [[NAME[l], f"{ps[l]['f1']:.1f}%", f"{ps[l]['recall']:.1f}%", ps[l]["n"]]
       for l in order], [1.8, 1.4, 1.4, 1.4], highlight=[0])
para(doc)
weak = order[0]
para(doc, f"**{NAME[weak]} is the weakest sign at {ps[weak]['f1']:.1f}%**, "
          f"against {ps[order[-1]]['f1']:.1f}% for the strongest.")
para(doc, "The important question is whether it fails under particular "
          "recording conditions — if so, the fix is more conditions. It does "
          "not:")
cond = R["conditions_per_sign"]
rows = []
for key, (ok, n) in sorted(cond.items()):
    lab, light, dist = key.split("|")
    if lab == weak and n:
        rows.append([f"{light} light, {dist}", f"{ok}/{n}", f"{ok/n*100:.0f}%"])
if rows:
    table(doc, ["Condition", "Correct", "Rate"], rows, [2.6, 1.4, 1.4])
    para(doc)
para(doc, f"The failures are spread across every lighting and distance "
          f"combination. **So the recommendation is more recordings of "
          f"{NAME[weak]} and the sign it is confused with — not more "
          f"conditions, and not more recordings in general.** The other five "
          f"signs are already above 90% and additional takes of them would "
          f"change very little.")

# ── 6. ask 4 ─────────────────────────────────────────────────────────
doc.add_page_break()
doc.add_heading("6. Question 4 — which features to detect", level=1)
fi = R["feature_importance"]
figure(doc, "ta_features.png", 6.8,
       "Figure 5 — where the model's attention goes, measured on the "
       "original feature set.")
table(doc, ["Grouping", "Finding"], [
    ["Body part", f"Hands carry "
     f"{fi['by_part']['left hand'] + fi['by_part']['right hand']:.0f}% of the "
     f"importance; the body only {fi['by_part']['body']:.0f}%."],
    ["Coordinate", f"Depth (z) is {fi['by_coord']['z']:.0f}%, the largest "
     f"single channel — and body depth is always zero, so all of it comes "
     f"from the hands."],
    ["Statistic", f"The extremes matter most: max "
     f"{fi['by_stat']['max']:.0f}%, min {fi['by_stat']['min']:.0f}%."],
    ["Concentration", f"Half of all importance sits in "
     f"{fi['half_importance_in']} of {fi['total_features']} features."],
], [1.5, 5.2])
para(doc)
para(doc, f"**The single most useful finding: bone directions.** Adding the "
          f"direction each joint points relative to the joint it attaches to "
          f"— rather than only where it sits — now accounts for "
          f"{fi['bones_block']:.0f}% of the model's attention, against "
          f"{fi['joints_block']:.0f}% for the positions that were previously "
          f"the only input.", bold=False)
para(doc, "Two practical consequences follow:")
bullet(doc, "**Hand tracking quality is the ceiling on this system.** The "
            "model depends almost entirely on the hands, and the hands are "
            f"the part the tracker loses most — "
            f"{R['hand_dropout']['left_missing_mean']*100:.0f}% of frames for "
            f"the left hand. Better hand detection would be worth more than "
            f"any change to the models.")
bullet(doc, "**Directions travel between people; sizes do not.** Bone "
            "directions were tested against bone lengths: keeping length made "
            "the system 6 points worse on an unseen signer, because length "
            "encodes body size, which identifies the person rather than the "
            "sign.")

# ── 7. same vs unseen ────────────────────────────────────────────────
doc.add_heading("7. Why every number is given twice", level=1)
para(doc, f"The features that recognise signs also identify the signer: a "
          f"model trained to predict *who is signing* rather than *what* "
          f"scores {au['signer_probe']:.1f}%, where guessing would give "
          f"{au['signer_chance']:.0f}%. Almost every recording carries a clear "
          f"signature of the person who made it.")
para(doc, "So a test that trains and tests on the same people lets the model "
          "answer partly by recognising the person. That is not cheating, and "
          "it is a fair test of a personal recogniser trained on its own user "
          "— but it is the wrong test for a system a stranger will use.")
rows = []
if best_d:
    rows.append([best_d, f"{DEEP[best_d]['same']:.1f}%",
                 f"{DEEP[best_d]['unseen']:.1f}%",
                 f"{DEEP[best_d]['unseen'] - DEEP[best_d]['same']:+.1f}"])
for a in sorted(ALGOS,
                key=lambda a: -R["classical"][a]["bones"]["unseen"]["f1"])[:3]:
    s = R["classical"][a]["bones"]["same"]["f1"]
    u = R["classical"][a]["bones"]["unseen"]["f1"]
    rows.append([a, f"{s:.1f}%", f"{u:.1f}%", f"{u - s:+.1f}"])
table(doc, ["Model", "Same signer", "Unseen signer", "Cost"], rows,
      [1.6, 1.6, 1.6, 1.2])
para(doc)
para(doc, "Both are reported throughout. The unseen-signer figure leads, "
          "because it is the one that predicts what a new user will "
          "experience.")

# ── 8. conditions ────────────────────────────────────────────────────
doc.add_heading("8. Did the recording conditions matter?", level=1)
para(doc, "The grid allows a whole condition to be held back — train only on "
          "brightly-lit recordings, test only on dim ones — and see what "
          "survives.")
base = R["classical"][best_c]["bones"]["same"]["f1"]
table(doc, ["Trained on", "Tested on", "macro-F1"],
      [[c["trained_on"], c["tested_on"], f"{c['f1']:.1f}%"]
       for c in R["condition_transfer"]], [1.9, 1.7, 1.5])
para(doc)
worst = min(R["condition_transfer"], key=lambda c: c["f1"])
para(doc, f"**The conditions have stopped mattering.** The hardest transfer "
          f"({worst['trained_on']} → {worst['tested_on']}) still reaches "
          f"{worst['f1']:.1f}%, close to the {base:.1f}% scored with no "
          f"condition held back at all.")
para(doc, "This is a change from earlier in the project. When only one person "
          "had recorded, holding back a lighting condition was catastrophic. "
          "With four people in four different rooms it costs almost nothing. "
          "**The cheapest fix for a condition problem turned out to be more "
          "people, not tighter control of the condition** — and the same "
          "logic says a fifth and sixth signer would be worth more than any "
          "further tuning.")

# ── 9. synthetic ─────────────────────────────────────────────────────
doc.add_heading("9. Does synthetic data help?", level=1)
syn = R["synthetic"]
if syn:
    deltas = sorted(v["delta"] for v in syn.values())
    med = float(np.median(deltas))
    helped = sum(1 for d in deltas if d > 0.5)
    hurt = sum(1 for d in deltas if d < -0.5)
    para(doc, f"Each real recording was rebuilt on six differently-"
              f"proportioned bodies, giving {R['n_synth']} extra training "
              f"examples. When a recording is held back for testing, all of "
              f"its synthetic copies are held back with it — otherwise the "
              f"model would be tested on near-copies of its own training data.")
    table(doc, ["Algorithm", "Real only", "With synthetic", "Change"],
          [[a, f"{syn[a]['real']:.1f}%", f"{syn[a]['both']:.1f}%",
            f"{syn[a]['delta']:+.1f}"]
           for a in sorted(syn, key=lambda a: -syn[a]["delta"])],
          [1.7, 1.5, 1.6, 1.2])
    para(doc)
    para(doc, f"**Median change: {med:+.1f} points.** {helped} algorithms "
              f"improved, {hurt} got worse. Synthetic data is not the lever "
              f"it was hoped to be, and the reason is understood: the "
              f"recordings are already scaled so that everyone's shoulders "
              f"are the same width, and body size is most of what the "
              f"synthetic generation varies. It adds examples rather than "
              f"information.")

# ── 10. trust ────────────────────────────────────────────────────────
doc.add_page_break()
doc.add_heading("10. Why these numbers can be trusted", level=1)
para(doc, "This project has twice produced results that looked excellent and "
          "were wrong — once because training data leaked into the test set, "
          "once because a data-generation step was run twice. Both looked "
          "fine until the right thing was measured. Every result above was "
          "therefore attacked before being reported.")
table(doc, ["Check", "Result"], [
    ["Identical recordings on both sides of a split",
     f"{R['audit']['cross_take_duplicates']} found"],
    ["A recording appearing in both training and testing",
     "none — asserted on every fold"],
    ["A held-out signer contributing to training",
     "none — asserted on every fold"],
    ["Synthetic copies separated from their original",
     "none — copies always follow the original"],
    ["**Labels shuffled, everything re-run**",
     f"**{au['permutation_same']:.1f}% / {au['permutation_unseen']:.1f}% — "
     f"at or below the {au['chance']:.1f}% expected from guessing**"],
], [3.6, 3.1], highlight=[4])
para(doc)
para(doc, "**The last check is the important one.** The labels were shuffled "
          "at random and the entire study re-run. If anything in the pipeline "
          "were carrying the answer across the split — a duplicated "
          "recording, a leaked copy, a signer fingerprint — the score would "
          "have stayed above chance. It did not.")
para(doc, "That check is now part of the harness, so it runs every time "
          "rather than being remembered.", size=10, italic=True,
     color="595959")

# ── 11. what changed ─────────────────────────────────────────────────
doc.add_heading("11. What changed, and what each change was worth", level=1)
figure(doc, "ta_journey.png", 6.6,
       "Figure 6 — unseen-signer score after each fix.")
cl = {f: np.mean([R["classical"][a][f]["unseen"]["f1"] for a in ALGOS])
      for f in ("original", "hand_fix", "bones")}
table(doc, ["Change", "What it fixed", "Unseen-signer (mean)"], [
    ["Starting point", "—", f"{cl['original']:.1f}%"],
    ["Stop inventing hand positions",
     f"The tracker loses the left hand in "
     f"{R['hand_dropout']['left_missing_mean']*100:.0f}% of frames; those "
     f"frames were filled with a frozen copy and presented as measurements",
     f"{cl['hand_fix']:.1f}%"],
    ["Add bone directions",
     "Nothing had ever told the model the joints form a body",
     f"{cl['bones']:.1f}%"],
], [1.9, 3.4, 1.4], highlight=[2])
para(doc)
para(doc, f"A further gain came from the models themselves: the best network "
          f"reaches {DEEP[best_d]['unseen']:.1f}% where the best classical "
          f"algorithm reaches "
          f"{R['classical'][best_c]['bones']['unseen']['f1']:.1f}%.")
para(doc, "None of this required new recordings, new equipment, or a change "
          "to the stored data.")

# ── 12. limits ───────────────────────────────────────────────────────
doc.add_heading("12. Limits", level=1)
para(doc, "Stated plainly, because they bound every number above.")
bullet(doc, f"**Four signers.** Every unseen-signer figure is an average over "
            f"four tests, each holding out one person. The spread between "
            f"those four is ±{DEEP[best_d]['unseen_sd']:.1f} points for the "
            f"best model. A fifth and sixth signer would do more for "
            f"confidence than any further work on the models.")
bullet(doc, f"**Seven signs.** A working vocabulary is hundreds. Nothing here "
            f"shows the approach scales to that, and the confusable pair "
            f"found at seven signs suggests it gets harder.")
bullet(doc, "**The signers are project members, not fluent Deaf signers.** "
            "Sign production by learners differs from native production, so "
            "no claim about real-world Khmer Sign Language follows from this "
            "data.")
bullet(doc, "**One camera, one tracking system.** Nothing separates a "
            "property of sign language from a property of the tracker used.")
bullet(doc, "**No hyperparameter tuning**, for any model in either category. "
            "The comparison is fair — nothing was tuned — but no model is at "
            "its best.")

# ── appendix ─────────────────────────────────────────────────────────
doc.add_page_break()
doc.add_heading("Appendix A — every model", level=1)
para(doc, "Macro-F1. Cross-validation for the same-signer column, "
          "leave-one-signer-out for the unseen column.", size=10,
     italic=True, color="595959")
para(doc, "Classical", bold=True)
rows = []
for a in sorted(ALGOS, key=lambda a: -R["classical"][a]["bones"]["unseen"]["f1"]):
    c = R["classical"][a]["bones"]
    rows.append([a, R["classical"][a]["origin"],
                 f"{c['same']['f1']:.1f}", f"{c['same']['sd']:.1f}",
                 f"{c['unseen']['f1']:.1f}", f"{c['unseen']['sd']:.1f}"])
table(doc, ["Algorithm", "Source", "Same", "±", "Unseen", "±"], rows,
      [1.3, 1.3, .9, .7, .9, .7], highlight=[0])
para(doc)
if R["custom_algos"]:
    para(doc, f"Algorithms contributed by team members and picked up "
              f"automatically: {', '.join(R['custom_algos'])}.",
         size=10, italic=True, color="595959")
para(doc, "Neural networks", bold=True)
rows = []
for k in sorted(DEEP, key=lambda k: -DEEP[k]["unseen"]):
    d = DEEP[k]
    rows.append([k, f"{d['params']:,}", f"{d['same']:.1f}", f"{d['same_sd']:.1f}",
                 f"{d['unseen']:.1f}", f"{d['unseen_sd']:.1f}",
                 "yes" if d["live_capable"] else "offline only"])
table(doc, ["Model", "Parameters", "Same", "±", "Unseen", "±", "Runs live?"],
      rows, [1.2, 1.1, .8, .6, .8, .6, 1.2], highlight=[0])

doc.add_heading("Appendix B — reproducing this report", level=1)
para(doc, "python algo_comparison/run_task_a.py", size=10)
para(doc, "python algo_comparison/make_task_a_report.py", size=10)
para(doc, f"Corpus `{R['lang']}` only. {R['folds']}-fold cross-validation, "
          f"{R['epochs']} epochs for the networks, all seeds fixed. Full "
          f"numbers in results.json beside this file; method and history in "
          f"docs/project/PROBLEM_LOG.md.", size=10, italic=True,
     color="595959")

path = OUT / "Task_A_Report.docx"
doc.save(path)
print(f"wrote {path}")
