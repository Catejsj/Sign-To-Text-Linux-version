# Research paper — content draft

*Working draft, 2026-09-02. Every number here traces to a file named beside it.
Sections marked **[GAP]** are things the paper needs that we do not have yet.*

---

## The argument in one paragraph

Khmer Sign Language has almost no labelled data, so any system for it is built
under low-resource conditions where evaluation choices dominate reported
performance. We built a landmark-based KSL recogniser and two corpora, and
found that three separate data-leakage mechanisms — each invisible because
leaks only ever improve results — had inflated our own numbers, one of them
totally. After closing them we show that the conclusions a low-resource sign
recogniser reports are governed less by model choice than by protocol:
algorithm rankings do not transfer across datasets, across recording protocols
on the same language, or between same-signer and unseen-signer evaluation; and
whether deep learning is worth its cost depends entirely on which corpus you
measure it on. We argue that low-resource sign-recognition papers should report
a difficulty-graded corpus and an unseen-signer condition, and we give the
measurements that show why.

**This is a methodology paper with a system attached, not an accuracy paper.**
That is the honest framing: our best number (86.8 macro-F1) is not competitive
with well-resourced sign-language work, and pretending otherwise invites the
one question we cannot answer — why so few signs and signers.

---

## Title candidates

1. *Protocol Before Architecture: Evaluation Leakage and Ranking Instability in
   Low-Resource Sign Language Recognition*
2. *What Your Split Is Hiding: Take-Aware Evaluation for Small Sign Language
   Corpora*
3. *When Is Deep Learning Worth It? A Difficulty-Graded Study on a Low-Resource
   Sign Language*

Prefer 1 or 2 — they name the contribution. 3 promises a broader answer than
n=4 signers can support.

---

## 1. Introduction

**Content to write:**

- KSL is effectively unresourced. Nearest prior work is Ponleur Veng / IDRI
  Cambodia (2024), ~2,645 sign videos. **[GAP: verify this citation, get the
  full reference, and read it — currently second-hand from an internal doc.]**
- Low-resource conditions force small corpora, which makes evaluation protocol
  decisive: with 337 takes, a leak or an optimistic split moves the headline
  number more than any architecture choice does.
- We report the leaks we found in our own pipeline, because the failure modes
  are structural rather than careless, and a paper that only reports the final
  clean number teaches nothing about how to avoid them.

**Claimed contributions:**

1. Two KSL landmark corpora, one recorded on a deliberate variation grid.
2. A take-aware evaluation protocol, and a measurement of what sample-level
   splitting costs: 100% of validation samples compromised.
3. Evidence that algorithm ranking is unstable along three independent axes.
4. A difficulty-conditional answer to whether sequence models earn their cost.
5. Feature attribution showing the model's dominant dependency is its least
   reliable input.

---

## 2. Related work

**[GAP — this is the weakest section and needs real reading.]**
`.icm/refs/` holds stored papers; `python3 .icm/tools/refs.py list`. Needed:

- isolated sign recognition from pose landmarks (the standard pipeline)
- low-resource / few-shot sign language recognition
- **signer-independent evaluation** — this is our closest neighbour and the
  literature almost certainly already argues for it; we must not claim novelty
  for the idea, only for the measurement in this setting
- synthetic / augmented signers via skeletal retargeting
- data leakage in ML evaluation generally (Kapoor & Narayanan's leakage
  taxonomy is the obvious anchor)

---

## 3. Data

### 3.1 Capture

MediaPipe + RTMPose, 48 joints (6 body + 21 left hand + 21 right hand), 60
frames at 2 seconds per take, stored `(60, 48, 3)`. Each take is written in two
paired views: `clean` (shoulder-anchored, scale-normalised, de-rolled) and
`noisy` (raw image space). See `docs/reference/SYSTEM_ARCHITECTURE.md`.

**Body `z` is identically zero** — 121,320 values checked. The body landmarks
are 2D; only hand landmarks carry a depth estimate. This matters in §7.

### 3.2 Two corpora

| | `khmer` | `khmer_var` |
|---|---|---|
| signs | 7 | 7 |
| takes / sign | 30 | 12 |
| real takes | 420 | 337 |
| synthetic | 2,520 | 2,022 |
| signers | 2 | 4 |
| protocol | freeform | **12-cell variation grid** |

The grid: 2 lighting × 2 distances × 3 standing positions, each sign performed
once per cell, by 4 people in 4 rooms. This is what makes `khmer_var` the
harder corpus and lets us test condition transfer directly.

### 3.3 Synthetic signers

Parametric skeletal retargeting: each take is rebuilt on differently
proportioned bodies. Joint angles are invariant under bone-length scaling
(documented max change 0.0004°), so the sign survives the transform — this is
the correctness argument, and it is checkable rather than assumed. See
`docs/reference/SYNTHETIC_RETARGETING.md`.

### 3.4 Ethics and privacy

Only skeleton coordinates are stored. No video or images are retained, and a
landmark array cannot be inverted to an image of the signer. Signers are
project members who consented to the recording. **[GAP: state whether there was
a written consent process; a reviewer will ask.]**

---

## 4. Evaluation protocol — the paper's spine

### 4.1 What a "take" is, and why it is the unit

One recording produces a clean view, a noisy view, and *n* synthetic copies of
each. All of them are the same two seconds of one person signing. Any split
that separates them is scoring a model on near-copies of its training data.

### 4.2 Three leaks, all silent

| # | mechanism | effect |
|---|---|---|
| C2 | random split over samples, classical path | augmented copies straddle the split |
| C3 | deleted take orphaned its synthetic children | ratio stopped dividing; all synthetic for a label silently dropped |
| C4 | `split_random`, deep path | **100% of validation samples had their own take in training** |

Plus a fourth, operational: `generate_synthetic.py` run twice without
`--clean` produced 12 copies numbered as two interleaved runs, so the
`variant // ratio` parent map grouped children under the wrong parent. That one
inflated results by up to 47 points and was caught only because a rerun made
scores *drop*.

**The unifying observation, and the sentence the paper should be built around:
a leak never announces itself, because it only ever makes results better.** A
result that contradicts an established one in the flattering direction deserves
the same scrutiny as one that contradicts it in the damaging direction.

### 4.3 The C4 measurement

Holding out 15% at seed 42, before the fix:

| corpus | val samples | with their own take in train |
|---|---|---|
| `khmer_var` | 707 | **707 (100%)** |
| `khmer` | 882 | **882 (100%)** |

Every validation clip was a different view, or a warped copy, of something the
model had already seen. This is the strongest single number in the paper.

### 4.4 Repeated random splits understate their own uncertainty

Our prior protocol averaged 8 independent take-aware 75/25 draws. Against
5-fold cross-validation on identical data:

- rankings identical
- **30 of 337 takes never tested**; others tested 0–7 times
- reported spread roughly **doubles** (rf ±2.5→±5.8, bagging ±3.1→±6.6,
  knn ±5.1→±7.9)

Overlapping draws are correlated, so their variance is not an honest estimate.
We recommend k-fold with grouping over repeated random subsampling for corpora
this size.

**All results below:** 5-fold `StratifiedGroupKFold` on take id, folds asserted
disjoint, **evaluated on real clean takes only**, identical folds across
categories.

---

## 5. Results — is deep learning worth it?

### 5.1 Main table

| corpus | best classical | best deep | verdict |
|---|---|---|---|
| `khmer_var` real | gboost 80.1 ±5.6 | tcn **85.9 ±2.9** | deep **+5.8** |
| `khmer_var` +synth | gboost 78.6 ±3.1 | transformer **86.8 ±1.0** | deep **+8.2** |
| `khmer` real | logreg 96.4 ±2.9 | tcn 98.3 ±1.8 | **tie** (1.9 < 2.4) |
| `khmer` +synth | logreg 97.1 ±1.0 | tcn 98.3 ±1.8 | **tie** (1.2 < 1.4) |

**The answer is conditional and the condition is the finding.** On the easier
corpus everything saturates at 96–98% and the categories are indistinguishable.
On the harder one, sequence models win by 5.8–8.2. A paper reporting only
`khmer` would have concluded — correctly for that data, wrongly in general —
that classical ML suffices.

### 5.2 Ranking instability, three axes

| axis | evidence |
|---|---|
| across datasets | LDA best cross-signer on landmarks (62.4), **last** on Sign Language MNIST (43.3); SVM 6th on landmarks, **1st** on images (83.7) |
| across protocols, same language | logreg 1st on `khmer`, **6th** on `khmer_var`; gboost 1st on `khmer_var`, 3rd on `khmer` |
| same vs unseen signer | logreg 97.1 → 43.6; bagging 96.0 → 75.0. **The ranking inverts** |

**Therefore: naming a single best algorithm is not defensible at this scale.**
This is a stronger and more useful claim than any winner we could name, and it
is what we should say when asked which algorithm to use.

### 5.3 Unseen signers

4-fold leave-one-signer-out on `khmer_var` (all of a held-out signer's data —
real *and* synthetic — removed from training):

| | real | +synthetic |
|---|---|---|
| gboost | **57.5 ±10.3** | 55.7 ±6.5 |
| bagging | 53.1 ±8.9 | **57.7 ±5.2** |
| rf | 54.2 ±7.2 | 56.0 ±8.2 |
| lda | 43.5 ±6.1 | 24.8 ±8.2 |

Same-signer 80–86 versus unseen-signer 57. **The generalisation gap, not the
headline accuracy, is the honest description of where this technology is.**

### 5.4 Synthetic augmentation helps the highest-capacity model

| model | real | +synth | Δ |
|---|---|---|---|
| transformer | 84.4 ±5.3 | **86.8 ±1.0** | **+2.4** |
| tcn | 85.9 ±2.9 | 85.4 ±2.9 | −0.5 |
| gboost | 80.1 | 78.6 | −1.5 |
| lda | 71.9 | 42.7 | **−29.2** |

The augmentation varies body geometry; the `clean` normalisation already
divides body scale out; so a model reading order-invariant summary statistics
has little left to gain. A high-capacity sequence model on 7× the data does
gain — and its fold-to-fold spread collapses from ±5.3 to ±1.0, **which is the
more important half of that result**.

LDA's collapse is reproducible and mechanistic: near-duplicate rows destabilise
the covariance inversion. **[GAP: test `solver="lsqr", shrinkage="auto"` — a
reviewer will ask, and it is one line.]**

---

## 6. Error analysis

### 6.1 One pair dominates

Confusion is not spread evenly. ជម្រាប់សួរ ↔ អរគុណ accounts for 19 of 337
takes; ជម្រាប់សួរ recall is 54.2% against ≥72.9% for every other sign.
Per-sign F1 ranges 60.5 (ជម្រាប់សួរ) to 92.2 (ត្រូវ).

The failures are **not** concentrated in one recording condition (41.7–66.7%
across all four lighting × distance cells), so this is a property of the sign
pair, not the capture setup. **Implication for data collection: more takes of
the confusable pair, not more conditions.**

### 6.2 Order-invariant features are part of the cause

`summary` is mean/std/min/max over 60 frames — reverse a clip and the vector is
unchanged. Two signs visiting the same positions in a different order are the
same point in feature space.

Adding per-third means and frame-to-frame velocity cut the pair's errors from
**19 to 13** and lifted recall to 60.4% / 81.2% — but is worth only **+0.7**
averaged over nine algorithms. Order-invariance is genuinely part of the
problem and bolting time back on is genuinely a partial fix; the remainder is
what the sequence models recover, which is why they win on this corpus.

**This is the cleanest causal story in the paper — a diagnosed defect, a
partial fix that works exactly as far as predicted, and an architecture that
closes the rest.**

---

## 7. Feature attribution — the dependency worth reporting

Random-forest importance over 576 `summary` features:

| body part | | statistic | | coordinate | |
|---|---|---|---|---|---|
| left hand | 49.9% | max | 36.4% | **z** | **40.6%** |
| right hand | 42.4% | min | 22.4% | x | 36.5% |
| body | **7.6%** | mean | 20.8% | y | 22.9% |
| | | std | 20.3% | | |

Hands carry **92.3%**. Body `z` is identically zero, so the entire 40.6%
z-importance is *hand* depth — a monocular estimate from a single RGB camera,
the least reliable channel in the representation. Half of all importance sits
in 112 of 576 features.

**The model's dominant dependency is its weakest input.** Our own logs already
rated hand detection the weak link; this quantifies how exposed the system is
to it, and makes hand-landmark quality the highest-value engineering target
rather than a known annoyance.

---

## 8. Threats to validity — write this section honestly, it is a strength

- **7 signs, 4 signers, 337 takes.** Every cross-signer claim rests on 4 folds.
  A 12-point gap between first and second place is well within what a fifth
  signer could move.
- `khmer` and `khmer_var` differ in takes/sign, signer count **and** protocol
  at once, so "harder corpus" is not decomposed into which factor causes it.
- Deep hyperparameters were not tuned; neither were classical ones. Neither
  category is at its ceiling, so the 5.8–8.2 gap is a floor for deep, not a
  measured maximum.
- Single camera, single capture stack. Nothing here separates a property of
  sign language from a property of MediaPipe/RTMPose.
- Signers are project members, not fluent Deaf signers. **This is the most
  serious limitation and must be stated plainly.** Sign production by learners
  differs from native production, and no claim about real-world KSL recognition
  follows from this data. **[GAP: is any validation by a fluent signer or a
  KSL teacher possible? It would change what the paper can claim.]**

---

## 9. Conclusion

For low-resource sign language recognition, protocol dominates architecture.
We found three leaks in our own pipeline, one total; we show that algorithm
ranking is unstable along three axes; and we show that the value of sequence
models is conditional on corpus difficulty in a way that a single-corpus study
cannot reveal. We recommend take-aware grouping, k-fold over repeated random
draws, a difficulty-graded corpus, and an unseen-signer condition reported
alongside every same-signer number.

---

## What to do before writing prose

1. **[GAP] Related work.** The single biggest hole. Signer-independent
   evaluation is likely already argued for in the literature — find it and
   position against it rather than claiming it.
2. **[GAP] Verify the IDRI Cambodia citation** first-hand.
3. Test the LDA shrinkage fix — one line, closes an obvious reviewer question.
4. Re-run the four category configs with **3 seeds** rather than 1, and report
   mean ± across seeds *and* folds. Cheap, and it hardens every table.
5. Land these runs in `algo_comparison/` so the numbers are reproducible from
   a repo driver rather than a scratch harness — the EVALUATE contract's
   requirement, and currently unmet.
6. Decide authorship and venue. A workshop on low-resource or sign-language NLP
   fits this better than a main conference: the contribution is methodological
   and the corpus is small, which a workshop audience will read as honest and a
   main-track reviewer may read as thin.

Sources: `docs/project/PROBLEM_LOG.md` §C, §G, §J ·
`.icm/experiments/runs/` · `.icm/notes/2026-09-02-deep-beats-classical-after-leak-fix.md`
