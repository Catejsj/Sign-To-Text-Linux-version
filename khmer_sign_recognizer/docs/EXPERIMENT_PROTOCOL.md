# SignLink — Synthetic-Data Experiment Protocol

**Goal:** prove (or disprove) that our synthetic data generator makes sign
recognition *better*. We test this on the AUTSL Turkish dataset first,
before spending weeks recording Khmer. Each team member runs ONE simple
(non-deep-learning) algorithm so we get an 8-way comparison.

This is a **landmark** experiment — nobody records anything here. The data
is already on disk. You only run training scripts.

---

## The question we're answering

> Does adding synthetic (body-retargeted) samples to the training set
> raise accuracy on **real, unseen** signers?

For every algorithm we run two trainings and compare:

| Run | Trained on | Tested on |
|---|---|---|
| **A — baseline** | real samples only | real held-out signers |
| **B — augmented** | real + synthetic | the **same** real held-out signers |

If **B's macro-F1 > A's macro-F1**, synthetic helped. The test set is
always real and identical between runs, so the comparison is fair.

---

## One-time setup

```powershell
cd khmer_sign_recognizer
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The data is synced via Drive. Make sure you have it:

```powershell
python scripts\drive_sync.py pull-data
# you should now have data\sequences_v2\autsl\ with real + synthetic files
```

---

## Who runs what

Each person owns ONE algorithm (all from scikit-learn, all "simple"):

| Person | `--algo` | Algorithm |
|---|---|---|
| 1 | `knn` | k-Nearest Neighbors |
| 2 | `logreg` | Logistic Regression |
| 3 | `rf` | Random Forest |
| 4 | `svm` | Support Vector Machine |
| 5 | `nb` | Gaussian Naive Bayes |
| 6 | `tree` | Decision Tree |
| 7 | `gboost` | Gradient Boosting |
| 8 | `lda` | Linear Discriminant Analysis |

---

## The two commands you run

Replace `knn` with your assigned algorithm:

```powershell
# Run A — baseline (real only)
python scripts\run_baseline.py --algo knn --lang autsl --mode real

# Run B — augmented (real + synthetic)
python scripts\run_baseline.py --algo knn --lang autsl --mode both
```

Each prints accuracy, macro-F1, per-signer F1, and a confusion matrix,
and appends a row to `data\experiments\baseline_results.csv` (our shared
table).

Optional sanity check (train on synthetic only — expect it LOWER than A):

```powershell
python scripts\run_baseline.py --algo knn --lang autsl --mode synthetic
```

---

## What to report back

Copy these four numbers from each run into the team sheet:

- **accuracy**
- **macro-F1**  ← the main comparison number
- **per-signer F1 mean (+/- std)**
- a screenshot of the **confusion matrix**

Then state: *"For <algo>, macro-F1 went from A=__% to B=__% (Δ = __)."*

---

## Reading the result as a team

When all 8 are done, `baseline_results.csv` gives us:

| algo | A (real) | B (both) | Δ |
|---|---|---|---|
| knn | 0.55 | ? | ? |
| ... | ... | ... | ... |

- **Most algorithms B > A** → synthetic works. Green light to record KSL
  with `--synthetic` on. This is the evidence for Dr. May Thu.
- **B ≈ A** → synthetic neither helps nor hurts; real data already varied
  enough. Report honestly.
- **B < A** → synthetic is hurting. We debug `src/v2/retarget.py` (likely
  the body-jitter range is too wide) before recording KSL.

---

## Notes / FAQ

- **Why macro-F1 not accuracy?** Macro-F1 averages per-class, so it isn't
  fooled by easy classes. It's our headline metric.
- **Why per-signer F1?** Leave-one-signer-out only means something if the
  model does *evenly* across held-out signers. Big variance = it's
  overfitting to signer style, not learning the sign.
- **Why is the test set always real?** Grading on synthetic data would
  measure performance on fake people. We only care about real signers.
  The loader enforces this — you can't accidentally test on synthetic.
- **`--features summary` vs `--features flat`?** Default `summary` (576
  features: per-joint mean/std/min/max over time) works better for most
  classical models. Try `--features flat` (8640 raw numbers) if you want
  to compare — just report which you used.
- **`--eval-on val` vs `test`?** Use `val` for all development. Only run
  `--eval-on test` ONCE at the very end for the final paper number, so we
  don't accidentally tune to the test set.
