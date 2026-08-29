# SignLink — Engineering Findings and Experimental Results

Reference notes for the Khmer Sign Language (KSL) recognition project: the
problems encountered, what fixed them, what was measured, and the conditions
under which the system works well or poorly.

Every number here was measured on this project's own data. Negative results are
included deliberately — several are more informative than the positive ones.

**System under test.** Landmark-based isolated sign recognition. RTMPose
(ONNX Runtime, GPU) provides 6 body keypoints; MediaPipe Holistic provides
2 × 21 hand landmarks. Each take is resampled to a fixed 60-frame clip of 48
joints × 3 coordinates and stored in two views (`clean` = shoulder-normalized,
`noisy` = raw image space). Classification uses summary statistics
(mean/std/min/max per joint-coordinate over time → 576 features) with
scikit-learn models, plus a TCN/Transformer path in PyTorch.

**Dataset.** 7 KSL signs (ជម្រាបសួរ, អរគុណ, ខុស, ត្រូវ, គ្រួសារ, ប៉ា, ម៉ាក់);
2 signers × 30 real takes per sign = 420 real takes, plus 6 synthetic
body-variants per take (2,520 synthetic). Environment: Python 3.11,
torch 2.11+cu128, opencv 4.13, mediapipe 0.10.14, scikit-learn 1.9,
RTX 4060 Laptop GPU.

---

## 1. Environment and platform problems

### 1.1 Silent CPU fallback in ONNX Runtime

**Problem.** RTMPose ran far slower than expected while appearing to work.
`onnxruntime.get_available_providers()` listed `CUDAExecutionProvider`, so the
GPU looked available.

**Cause.** Two independent failures. First, `rtmlib` depends on the **CPU**
`onnxruntime` package, which pip installs over `onnxruntime-gpu` into the same
directory, silently removing CUDA support. Second, the CUDA provider library
links against specific sonames (`libcublas.so.12`, `libcudnn.so.9`); the default
PyTorch wheel now ships CUDA **13**, so the versions did not match.

**Critical detail.** ONNX Runtime does **not** raise when the CUDA provider
cannot load. Even an explicit
`InferenceSession(model, providers=['CUDAExecutionProvider'])` succeeds on CPU
with only a warning, returning correct — but slow — results.

**Solution.** Install torch from the cu12 index; reinstall `onnxruntime-gpu`
after any `pip install -r`; and preload the CUDA libraries bundled with torch
using `ctypes.CDLL(..., RTLD_GLOBAL)` before the first session
(`src/cuda_setup.py`). `LD_LIBRARY_PATH` cannot fix this from inside Python
because glibc reads it once at process start.

**Verification rule.** Never trust `get_available_providers()`. Build a session
and read back `session.get_providers()`. Result after the fix: RTMPose bound to
CUDA at **31.7 FPS**.

### 1.2 Cross-platform capture and rendering

| Problem | Cause | Fix |
|---|---|---|
| Camera failed / slow to open on Linux | `cv2.CAP_DSHOW` (Windows-only) hard-coded | Select `CAP_V4L2` on Linux, `CAP_DSHOW` on Windows |
| 3D mannequin window never appeared | Open3D uses **GLFW**, whose Wayland path fails to initialize GLEW | Steer GLFW to X11/XWayland before the first `create_window()` |
| Camera preview rendered solid black | Open3D (GLFW/OpenGL) and OpenCV (Qt5) windows in one process conflict | Render the mannequin **offscreen** and composite it into a single OpenCV window |

Note: `QT_QPA_PLATFORM=xcb` does **not** fix the Open3D case — Open3D uses GLFW,
not Qt.

### 1.3 CPU throttling on battery power

Running on battery put all cores in `powersave` at 800–1300 MHz. MediaPipe
Holistic is CPU-bound and measured **64.5 ms/frame (15.5 FPS ceiling)** in that
state, which was mistaken for a camera or lighting problem.

**Implication for data collection:** capture frame rate roughly halves on
battery. Since takes are resampled to a fixed 60 frames, low-FPS recordings are
temporally coarser. Record on mains power, and do not mix power states within a
dataset.

---

## 2. Capture conditions

### 2.1 Hand tracking is the weak link

Measured detection rates while signing:

| landmark source | detection rate |
|---|---|
| RTMPose body (pose) | ~100% |
| MediaPipe hands | **~57%** |

**Consequence for sign selection.** Signs distinguished by *arm trajectory* are
reliable; signs distinguished by *fine handshape* are not, because the hand
landmarks are missing in roughly 4 frames out of 10 at webcam distance. Choose
vocabulary accordingly, especially for a small pilot lexicon.

### 2.2 Low light: measurable but insufficient

A low-light enhancement stage (CLAHE on the LAB lightness channel + gray-world
white balance + camera brightness/gamma control) was implemented and measured:

| lighting | no enhancement | best preset |
|---|---|---|
| warm/orange bulb | 33.3% | **57.3%** |
| white bulb | 42.0% | **53.6%** |

(hand-detection rate, 15 s per condition)

Enhancement roughly doubled hand detection, yet ~57% remained too low for
practical use — supplementary physical lighting was still required. **The feature
was subsequently removed** as not worth its complexity.

**Hardware finding worth recording.** The webcam exposes no `gain` control, only
`exposure_time_absolute`. An initial design capped exposure to reduce motion blur
during fast signing; this *reduced* pose detection from 100% to **0%**, because
with no gain to compensate, a shorter shutter only removes light. On such a
sensor, brightness must be added via `brightness`/`gamma`, never by shortening
exposure.

**Practical conclusion:** add real light. No post-processing substitutes for
photons reaching the sensor.

### 2.3 Camera geometry: what is corrected and what is not

`shoulder_normalize` anchors on the mid-shoulder and divides by shoulder width,
which removes **distance/scale** and **position in frame**. It does *not* remove
rotation. Measured camera roll across one session: **−10.5° to +19.7°**.

Effect of camera angle on accuracy (train level, test tilted):

| condition | 0° | 10° | 20° | 30° |
|---|---|---|---|---|
| without correction | 96.3 | 94.7 | 92.2 | **90.0** |
| with de-roll | 96.7 | **96.7** | **96.7** | **96.7** |

`deroll()` rotates each frame so the shoulder line is horizontal. It removes the
nuisance variable entirely and outperformed rotation *augmentation* (94–95%),
which only teaches tolerance. Vertical foreshortening (camera pitch) cost under
1 point and was left uncorrected.

**Framing matters more than either.** After moving to a taller desk, recognition
degraded sharply. Diagnosis: shoulders sat at 0.883 of image height (training
range 0.361–0.719) and **wrists were detected in only 13.5% of frames** — the
hands were largely outside the frame. No model can classify landmarks that were
never captured.

---

## 3. Data and evaluation methodology

### 3.1 Evaluation must split by *take*, not by sample

Each real take generates N synthetic body-variants. A naive random split places
variants of a held-out take into training, leaking the test motion.

The synthetic filenames carry no parent reference, but variants are generated
N-per-take sequentially, so when the counts divide evenly, synthetic variant `v`
belongs to real take `v // N`. Splitting on that group key keeps a take's
variants on one side of the split.

**Related failure:** deleting a real take orphans its synthetic children, the
ratio stops dividing evenly, the parent link becomes underivable, and affected
synthetic data is then **silently dropped from training**. Regenerate synthetic
after deletions.

The deep-learning path had the same leak (`split_random` splits samples, not
takes), which is why it reported a perfect validation score.

### 3.2 Offline accuracy does not predict live behaviour

Models are trained on complete, segmented takes but must classify a
**continuously sliding window** that usually contains a partial gesture — an
input distribution never seen in training.

| model | offline (complete clips) | live (sliding window) | confidence on partial windows |
|---|---|---|---|
| logistic regression | **98.1%** | 90.5% | **0.90** |
| random forest | 96.1% | **100%** | **0.60** |
| TCN (deep) | (leaky) 100% | 88.1% | **0.91** |

Logistic regression and the TCN are **overconfident out of distribution** — they
report ~0.9 confidence while wrong, defeating any confidence threshold. The tree
ensemble cannot extrapolate beyond its training data, so it reports honest
uncertainty (0.60) and a confidence filter works.

**For a continuously-running recognizer, calibration matters as much as peak
accuracy.** Ranking models by offline accuracy alone selects the wrong model.

### 3.3 Deep model did not help at this data scale

The TCN reached 100% training accuracy by epoch 45 (memorization) on 30 takes per
class and performed *worse* live than random forest. Its theoretical advantage is
real — summary statistics discard temporal *order*, which a TCN preserves — but
it cannot express itself at this dataset size.

---

## 4. The central result: single-signer data cannot validate signer-invariance

Synthetic body-variant augmentation was repeatedly measured as **neutral or
harmful** — until a second signer existed.

**Same-signer evaluation (one signer, held-out takes):**

| method | effect |
|---|---|
| synthetic body-variants | mean −7.3 pts; helped 1 of 7 algorithms |
| time-warp/noise/rotation augmentation | helped only k-NN; −1 to −2 for strong models |
| anthropometric + IK retargeting vs random scaling | +0.69 pts, SEM ±0.57 (not significant, 14 paired splits) |

**Cross-signer evaluation (train on one person, test on the other):**

| holdout | algo | real only | + synthetic | effect |
|---|---|---|---|---|
| Vichet | rf | 60.2% | **72.0%** | **+11.9** |
| Vichet | logreg | 37.6% | 38.2% | +0.6 |
| Vichet | svm | 16.6% | 17.1% | +0.5 |
| Vichet | knn | 18.1% | 21.1% | +3.1 |
| Piseth | rf | 48.3% | **54.4%** | +6.2 |
| Piseth | logreg | 37.0% | **48.0%** | +11.0 |
| Piseth | svm | 19.6% | 29.0% | +9.4 |
| Piseth | knn | 33.8% | 34.6% | +0.8 |

**Mean +5.4 points; helped 5 of 8; hurt 0 of 8.**

**Interpretation.** Synthetic body-variation augmentation varies the one factor a
single-signer test holds constant. Such a test is *structurally incapable* of
showing its benefit, and will mildly penalize it for trading specificity for
generalization. Reporting "augmentation did not help" from single-signer data is
a methodological error.

**Corollary observed in both directions:** synthetic helps unknown signers
(+5.4) and slightly hurts known signers (96.6% → 95.6%). Both facts follow from
the same mechanism.

### 4.1 Extreme body warping degrades class separability

Distance from a synthetic variant to its source, expressed as a percentage of the
mean distance between *different signs*:

| jitter | distance vs between-sign distance |
|---|---|
| 0.20 | 32% |
| 0.35 | 57% |
| 0.50 | **91%** |
| 0.65 | **124%** |

Beyond ~0.35 a variant drifts as far from its source as a different sign, and
accuracy falls (rf: 96.1 → 93.5 at jitter 0.50). **Jitter 0.20 is retained.**

Two structural notes: `shoulder_normalize` divides by shoulder width, which
**exactly cancels** any global/shoulder scaling parameter; and independently
sampling each bone produced anatomically impossible identities
(forearm/upper-arm ratio 0.67–1.48, versus 0.84–1.15 when sampling one correlated
build factor).

---

## 5. Multi-signer results

**Cross-signer generalization (unseen person): ~63%**, versus ~96% for a known
signer — a ~33-point drop, consistent with the 10–30% reported in the literature
for better-resourced sign languages.

**Combining signers into one model:**

| trained on | overall | on signer A | on signer B |
|---|---|---|---|
| signer A only | 80.6% | **98.0%** | 60.5% |
| signer B only | 78.1% | 52.0% | **97.2%** |
| **both (real)** | **96.6%** | 96.0% | 97.2% |
| both (real + synthetic) | 95.6% | 96.4% | 94.8% |

**A single model serves multiple signers at almost no cost.** Adding a second
signer cost signer A 2 points (98.0 → 96.0) while gaining signer B 37 points
(60.5 → 97.2). Per-signer models are unnecessary.

---

## 6. Operating conditions

### 6.1 Conditions for full potential

| factor | requirement |
|---|---|
| Signer | present in training data (~96%); otherwise expect ~63–72% |
| Lighting | bright, even, front-facing; avoid single dim warm sources |
| Framing | upper body centred, shoulders near mid-frame, **hands fully in frame** |
| Camera angle | consistent with training; roll auto-corrected, keep pitch stable |
| Power | mains (battery halves capture frame rate) |
| Vocabulary | signs distinguished by arm trajectory, visually distinct |
| Signing style | clear pause between signs — segmentation triggers on motion stop |
| Distance | shoulder width ~0.41–0.58 of frame width |

### 6.2 Conditions that degrade performance

| condition | measured impact |
|---|---|
| Unseen signer | 96% → **63%** |
| Hands outside frame | detection collapses; sign unrecognizable |
| Low light (warm bulb) | hand detection 33–57% |
| Camera roll (uncorrected) | −6.3 points at 30° |
| Battery power | 30 → ~15 FPS |
| Fine-handshape signs | unreliable (~57% hand detection) |
| Continuous signing without pauses | end-of-sign segmentation cannot trigger |
| Mixed camera setups within a dataset | adds nuisance variance |

### 6.3 Known limitations

1. **Body landmarks are 2D.** RTMPose supplies no depth (`z = 0`), so camera
   pitch cannot be distinguished from body proportion. Only hands carry a
   relative z.
2. **Isolated signs only.** No continuous-sentence segmentation or grammar.
3. **Two signers.** Cross-signer numbers rest on a single held-out person per
   direction.
4. **Small lexicon.** 7 signs; confusion grows with vocabulary size.
5. **Non-manual markers ignored.** Facial expression and mouthing, which are
   grammatically significant in sign languages, are not captured.

---

## 7. Recommendations

1. **Recruit more signers before any other optimization.** Going from one to two
   signers converted every negative augmentation result into a positive one and
   lifted unseen-signer accuracy from 60% to 72%. No algorithmic change came
   close to that.
2. **Evaluate with leave-one-signer-out** (`--holdout`). Same-signer scores
   overstate deployment performance by ~33 points.
3. **Prefer calibrated models for live use.** Random forest beat logistic
   regression and a TCN in live operation despite lower offline accuracy.
4. **Keep synthetic augmentation on for deployment to unknown users**; it costs
   ~1 point on known signers and gains ~5–12 on unknown ones.
5. **Standardize capture** (lighting, framing, camera angle, mains power) and
   record the setup alongside the data.
6. **Report both same-signer and cross-signer accuracy.** Publishing only the
   former is the single most misleading choice available in this problem domain.
