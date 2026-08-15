# SignLink — Algorithm Comparison and Robustness Report

Two experiments on Khmer Sign Language recognition from pose landmarks, over
7 signs: ជម្រាប់សួរ, អរគុណ, ខុស, ត្រូវ, គ្រួសារ, ប៉ា, ម៉ាក់.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| question | which algorithm generalises to a **new person**? | which survives a **change in recording setup**? |
| data | `khmer` — 2 signers × 7 signs × 30 takes = 420 takes | `khmer_var` — 1 signer × 7 signs × 12 takes = 84 takes |
| split | leave-one-signer-out | held-out condition |
| headline | **bagged trees, 75.5% macro-F1** | **tree-based methods survive; linear ones collapse** |

Every model is trained on the same 576 summary features (per-joint mean, std,
min, max over 60 frames) and wrapped in the same `StandardScaler`, so the
comparison is between algorithms and nothing else. Evaluation is always on real
recordings — never synthetic.

---

## 1. Method

**Landmarks, not pixels.** Each take is a `(60 frames, 48 joints, 3)` array:
6 body joints from RTMPose plus 21 landmarks per hand from MediaPipe. Two views
are stored per take — `clean` (shoulder-anchored, scaled by shoulder width,
rotated so the shoulder line is level) and `noisy` (raw image coordinates). All
results below use `clean`.

**Why the split matters more than the score.** A random split of takes puts
recordings of the same person, in the same room, on both sides. Every algorithm
then scores in the nineties and the comparison is meaningless. Both experiments
therefore split along the axis we actually care about.

---

## 2. Experiment 1 — generalising to an unseen person

Train on one signer, test on the other, both directions, 5 seeds each.

| algorithm | holdout=A | holdout=B | pooled macro-F1 | 95% interval |
|---|---|---|---|---|
| **Bagged Trees** | 58.4 ± 4.1 | 81.6 ± 2.4 | **75.5** | [71.4, 79.3] |
| Ridge Classifier | 74.9 | 45.3 | 61.9 | [57.3, 66.2] |
| LDA | 67.5 | 50.6 | 60.9 | [56.0, 65.5] |
| Random Forest | 49.4 ± 0.7 | 62.1 ± 2.3 | 57.6 | [52.5, 62.1] |

Bagged trees lead random forest by roughly 18 points, and the bootstrap
intervals do not overlap, so the gap is not sampling noise.

**Why bagging beats random forest here.** Both grow many trees on bootstrap
samples; random forest additionally restricts each split to a random subset of
features. With 576 summary features derived from only 48 joints, many features
carry nearly the same information, and that extra randomness discards signal
rather than decorrelating error. Bagging keeps the resampling and drops the
feature restriction.

### The same-signer numbers, for contrast

Standard 5-seed random split on the same data:

| algorithm | accuracy | macro-F1 | unseen signer |
|---|---|---|---|
| Logistic Regression | 97.1% | 97.1% | 43.6% |
| Random Forest | 96.9% | 96.9% | 59.3% |
| Gradient Boosting | 95.0% | 95.0% | 36.2% |
| LDA | 94.8% | 94.7% | 62.4% |
| GRU (recurrent) | 94.3% | 94.3% | 61.4% |
| MLP | 94.0% | 94.0% | 40.2% |
| SVM | 94.0% | 94.0% | 27.1% |
| k-NN | 83.8% | 83.5% | 31.0% |

**This table is the main result of the report.** Logistic regression is top on
the left and second-worst on the right. The ordering barely survives the change
of split. A ~35-point drop separates "works on people it has seen" from "works
on anyone", and no amount of same-signer accuracy predicts the second column.

---

## 3. Experiment 2 — surviving a change of setup

`khmer_var` was recorded on a deliberate grid: 12 takes per sign, crossing
2 lighting conditions × 2 distances × 3 standing positions. The condition is
recoverable from the take number, so each axis can be held out in turn — train
on one setting, test on a setting the model has never seen.

Macro-F1, training on the first setting and testing on the second:

| algorithm | → dim light | → far | → right | mean |
|---|---|---|---|---|
| Random Forest | 63.8 | 85.5 | 81.1 | **76.8** |
| Decision Tree | 65.9 | 81.0 | 72.8 | 73.2 |
| LDA | 23.4 | 77.1 | 82.3 | 60.9 |
| Gradient Boosting | 56.3 | 67.7 | 53.2 | 59.1 |
| Logistic Regression | 13.9 | 75.2 | 78.5 | 55.9 |
| MLP | 16.0 | 70.5 | 59.7 | 48.7 |
| SVM | 3.7 | 64.9 | 66.7 | 45.1 |
| k-NN | 16.1 | 40.5 | 33.6 | 30.0 |

Bagged trees score **77.6** mean across the same three splits, again the best of
the non-built-in candidates.

### The lighting column is the interesting one

Position and distance transfer reasonably for most algorithms. Lighting does
not, and it separates the field sharply:

- tree-based (decision tree 65.9, random forest 63.8, gradient boosting 56.3)
  hold up
- linear and margin-based (LDA 23.4, logistic regression 13.9, SVM 3.7) fall
  apart — SVM at 3.7% is far below the ~14% a random guess would score

A plausible reading: threshold-based models ask "is this coordinate past this
value", which survives a systematic shift in part of the feature vector, while
weighted-sum models have every coefficient shifted at once.

### Two honest caveats

**The cause is not what we assumed.** The obvious explanation was that dim light
breaks hand detection. Measured directly: **62.1% of hand landmarks missing in
full light versus 60.8% in dim** — no meaningful difference, and marginally
*better* in dim. Detection loss does not explain the collapse.

**Lighting is confounded with recording order.** Takes 1–6 were recorded in full
light and 7–12 in dim, always in that sequence, so "dim light" is inseparable
from "recorded later" — fatigue, drift, or simply signing differently after
twenty takes. The → dim column is a real, measurable transfer failure, but it
cannot be attributed to lighting specifically. Counterbalancing (half the
signers recording dim first) would resolve this and is the obvious next step.

The **→ right** column has no such problem: standing position cycles every three
takes, so it is evenly spread through the session.

---

## 4. What this means

1. **Report the split, not just the score.** Any of these algorithms can be
   presented as "94% accurate". The number that predicts real use is the one
   from a split the model has never seen along.

2. **Benchmark rankings do not transfer.** Logistic regression is best
   same-signer and near-worst cross-signer. On an image dataset (Sign Language
   MNIST) the ordering inverts again — LDA is best cross-signer on landmarks and
   last on images. An algorithm choice is only justified on the data and split
   it was measured on.

3. **Prefer ensembles of trees when conditions vary.** Across both experiments
   the tree-based family was consistently the most robust, and the single best
   model overall was bagged trees.

4. **The bottleneck is data, not algorithms.** The gap between 97% same-signer
   and 75% unseen-signer is far larger than the gap between the best and worst
   algorithm on a fixed split. More signers would buy more than more tuning.

---

## 5. Reproducing this

```bash
# Experiment 1 — full comparison, charts, and .docx report
python algo_comparison/run_comparison.py --lang khmer --seeds 5

# a subset, much faster while iterating
python algo_comparison/run_comparison.py --lang khmer --algos rf,lda,bagging

# one algorithm, one split
python scripts/run_baseline.py --algo bagging --lang khmer --mode real \
    --holdout Vichet

# Experiment 2 — same commands against the grid data
python algo_comparison/run_comparison.py --lang khmer_var --seeds 5
python scripts/run_baseline.py --algo bagging --lang khmer_var --mode real \
    --split-random 0.25
```

Every algorithm is seeded (`random_state=0`), so these reproduce exactly.

`bagging` is not one of the built-in nine — it lives in `custom_algos/bagging.py`
and is picked up automatically. See `custom_algos/README.md` to add your own the
same way.
