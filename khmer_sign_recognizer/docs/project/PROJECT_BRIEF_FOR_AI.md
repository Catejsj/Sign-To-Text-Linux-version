# SignLink — Project Brief (for AI-assisted literature research)

A self-contained description of the SignLink Khmer Sign Language recognition
project: architecture, specifications, measured results, novel aspects, and
open questions. Paste this into an AI assistant to find related papers, position
the work against existing literature, or draft summaries.

**How to use this file:** paste it whole, then ask for what you need — e.g.
*"Find papers related to sections 6 and 7 and tell me whether our findings are
novel"*, or *"Write a 250-word abstract from sections 1-6"*. Suggested prompts
are in section 10.

---

## 1. One-paragraph summary

SignLink is a landmark-based isolated sign-language recognition system for
Khmer Sign Language (KSL), a low-resource sign language with no public dataset.
Rather than classifying raw video, it extracts a skeletal representation
(body + hand keypoints) with two pose estimators running in parallel, normalizes
it to be invariant to signer position, distance and camera roll, and classifies
fixed-length landmark sequences with classical machine-learning models. It
includes a browser-based recording studio for building a dataset from scratch, a
parametric "synthetic signer" augmentation that retargets recorded motion onto
different body proportions, and a live recognition mode. The project's principal
empirical contribution is a demonstration that single-signer evaluation
systematically misjudges signer-invariance techniques.

---

## 2. Problem context

- **Khmer Sign Language (KSL)** is used by Cambodia's Deaf community. It is a
  low-resource language: no public annotated dataset, limited standardization,
  and few speakers relative to ASL/BSL.
- Standardization reference: the official KSL app from Cambodia's Ministry of
  Education, Youth and Sport; the Deaf Development Programme (DDP) has published
  five books of CSL signs and grammar.
- Practical constraint: the system must run on a consumer laptop with a webcam,
  because that is what the target users and researchers actually have.
- Consequence: **the dataset had to be created as part of the project**, which
  makes data-efficiency and signer-generalization central research questions
  rather than side issues.

---

## 3. System architecture

### 3.1 Two-detector capture pipeline

Two pose estimators run in parallel threads on the same camera frames:

| component | role | device | output |
|---|---|---|---|
| RTMPose (`rtmw-dw-x-l_simcc-cocktail14_270e-256x192`, COCO-WholeBody) via ONNX Runtime | body / arm keypoints | GPU (CUDA) | 9 body joints, 2D |
| YOLOX-m (`yolox_m_8xb8-300e_humanart`) | person detection for RTMPose | GPU | bounding box |
| MediaPipe Holistic (`model_complexity=1`) | hand landmarks | CPU (XNNPACK) | 2 × 21 hand landmarks |

**Rationale for the split:** RTMPose is more stable for body/arm tracking, while
MediaPipe provides detailed per-finger landmarks that RTMPose's wholebody hand
points do not match in quality. Measured detection reliability differs sharply
(see §6.2), which drove several design decisions.

Both streams are smoothed with **One-Euro filters** (adaptive low-pass:
low cutoff when still, high cutoff when moving) to suppress jitter without
adding lag during fast motion. MediaPipe is rate-limited to ~20 fps so it does
not starve the GPU pipeline or camera thread.

### 3.2 Representation

- **Per frame:** 48 joints × 3 coordinates
  = 6 body joints (shoulders, elbows, wrists) + 2 × 21 hand landmarks.
- **Per take:** resampled to a fixed **60 frames** → tensor `(60, 48, 3)`.
- **Body z is always 0** — RTMPose provides 2D keypoints only. Hands carry
  MediaPipe's relative z. This is an important limitation (see §8).
- **Two views stored per take:**
  - `clean` — mid-shoulder anchored, scaled by shoulder width, shoulder line
    levelled (signer-invariant)
  - `noisy` — raw image-space coordinates (preserves position/scale variation)
- **Classical feature vector:** per-(joint, coordinate) mean, std, min, max over
  time → **576 features**. Note this *discards temporal order*, which motivated
  testing sequence models.

### 3.3 Normalization chain

1. `resample_time` → fixed 60 frames (speed invariance)
2. `fill_nans` → interpolate missing landmarks
3. `shoulder_normalize` → anchor on mid-shoulder, divide by shoulder width
   (removes position and distance/scale)
4. `deroll` → rotate each frame so the shoulder line is horizontal
   (removes camera roll; added after measurement, see §6.3)

### 3.4 Models evaluated

- **Classical:** LDA, SVM (RBF), logistic regression, k-NN, gradient boosting,
  MLP, random forest, naive Bayes, decision tree (scikit-learn, `StandardScaler`
  pipeline).
- **Deep:** TCN (dilated residual temporal conv, ~4 dilation stages) and a
  Transformer, on the raw `(60, 144)` sequence; a bidirectional GRU for the
  comparison study.

### 3.5 Software / tooling

- Flask control panel (localhost) for recording: language folder management,
  label CRUD, per-take deletion, live configuration, and a Record/Recognize mode
  switch. Native OpenCV window shows camera + 3D mannequin side by side.
- Open3D parametric mannequin renders synthetic body-variants live, offscreen,
  composited into the camera window.
- Environment: Python 3.11, PyTorch 2.11+cu128, ONNX Runtime GPU 1.24,
  OpenCV 4.13, MediaPipe 0.10.14, scikit-learn 1.9; NVIDIA RTX 4060 Laptop GPU.

---

## 4. Dataset specification

| property | value |
|---|---|
| Language | Khmer Sign Language (KSL) |
| Signs | 7: ជម្រាបសួរ (hello), អរគុណ (thank you), ខុស (wrong), ត្រូវ (right/correct), គ្រួសារ (family), ប៉ា (father), ម៉ាក់ (mother) |
| Signers | 2 |
| Real takes | 420 (2 signers × 7 signs × 30 takes) |
| Synthetic takes | 2,520 (6 body-variants per real take) |
| Recording | ~3 s per take, 640×480 webcam, 30 fps nominal |
| Storage | `.npy` (60,48,3) + `.json` metadata sidecar, per view |
| Secondary dataset | AUTSL (Turkish SL) subsets, used for pipeline validation |
| Comparison dataset | Sign Language MNIST (Kaggle), 27,455/7,172 images, 24 static letters |

**Metadata per take:** label, signer id, source (real/synthetic), language, view
(clean/noisy), fps, variant index, notes. Signer id is preserved on synthetic
samples so leave-one-signer-out evaluation stays leak-free.

---

## 5. Synthetic data generation (parametric signer augmentation)

Retargets a recorded motion onto a different body. **Bone directions are
preserved exactly — only bone lengths change** — so every joint angle (i.e. the
sign itself) is mathematically invariant. Verified: maximum elbow-angle change
under extreme scaling was **0.0004°**.

Two variants implemented:

- **`scale`** (default): shoulders, upper arms, forearms and hands scaled by
  independently sampled factors in [1−j, 1+j], j = 0.20.
- **`ik`**: samples one anatomically consistent identity (a single "build"
  factor with ±3% per-bone deviation), then re-solves the elbow by **two-link
  inverse kinematics** so the wrist stays at the same body-relative location —
  motivated by *location being phonemic* in sign languages. Handles both
  reachability bounds (the wrist must lie in the annulus between |L1−L2| and
  L1+L2).

**Measured constraint on augmentation strength.** Distance from a synthetic
variant to its source, as a percentage of the mean distance between *different
signs*:

| jitter | 0.20 | 0.35 | 0.50 | 0.65 | 0.80 |
|---|---|---|---|---|---|
| distance vs between-sign distance | 32% | 57% | **91%** | **124%** | 198% |

Beyond ~0.35 a variant is as far from its own source as a different sign is,
i.e. augmentation starts crossing class boundaries and accuracy falls
(RF: 96.1% → 93.5% at j = 0.50). **This suggests a general principle: augmentation
magnitude should be bounded by a fraction of the inter-class distance.**

---

## 6. Key experimental results

### 6.1 The central finding — single-signer evaluation misjudges signer-invariance

The *same* synthetic augmentation, evaluated two ways:

| evaluation | effect of synthetic augmentation |
|---|---|
| held-out takes, **same signer** | mean **−7.3 points**; helped 1 of 7 algorithms |
| held-out **signer** (leave-one-signer-out) | mean **+5.4 points**; helped 5 of 8, hurt 0 of 8 |

Per-algorithm cross-signer detail:

| holdout | algorithm | real only | + synthetic | Δ |
|---|---|---|---|---|
| signer B | random forest | 60.2% | **72.0%** | **+11.9** |
| signer B | logistic regression | 37.6% | 38.2% | +0.6 |
| signer B | SVM | 16.6% | 17.1% | +0.5 |
| signer B | k-NN | 18.1% | 21.1% | +3.1 |
| signer A | random forest | 48.3% | 54.4% | +6.2 |
| signer A | logistic regression | 37.0% | **48.0%** | **+11.0** |
| signer A | SVM | 19.6% | 29.0% | +9.4 |
| signer A | k-NN | 33.8% | 34.6% | +0.8 |

**Interpretation:** body-variation augmentation varies precisely the factor a
same-signer test holds constant, so such a test is *structurally incapable* of
detecting its benefit and mildly penalizes it for trading specificity for
generalization. The same trade appears in reverse: adding synthetic data costs
~1 point on *known* signers (96.6% → 95.6%).

Corollary methodological claim: **reporting "augmentation did not help" from
single-signer data is an error of experimental design, not a property of the
augmentation.**

### 6.2 Detection reliability asymmetry

| landmark source | detection rate while signing |
|---|---|
| RTMPose body | ~100% |
| MediaPipe hands | **~57%** |

Design consequence: signs distinguished by *arm trajectory* are reliable; signs
distinguished by *fine handshape* are not, at webcam distance. This should
inform vocabulary selection for small pilot lexicons.

### 6.3 Camera roll is a significant, cheaply-removable nuisance variable

Measured camera roll across one recording session: **−10.5° to +19.7°**.

| eval tilt | 0° | 10° | 20° | 30° |
|---|---|---|---|---|
| without correction | 96.3 | 94.7 | 92.2 | **90.0** |
| with `deroll` | 96.7 | **96.7** | **96.7** | **96.7** |

Normalization (removing the variable) beat rotation *augmentation* (teaching
tolerance, 94–95%). Vertical foreshortening (pitch) cost <1 point.

### 6.4 Offline accuracy does not predict live performance

Models train on complete segmented takes but must classify a **continuously
sliding window** that usually contains a partial gesture — an input distribution
absent from training.

| model | offline (complete clips) | live (sliding window) | confidence on partial windows |
|---|---|---|---|
| logistic regression | **98.1%** | 90.5% | **0.90** |
| random forest | 96.1% | **100%** | **0.60** |
| TCN | 100% (leaky split) | 88.1% | **0.91** |

Linear and neural models are **overconfident out-of-distribution** (~0.9
confidence while wrong), defeating confidence thresholding. The tree ensemble
cannot extrapolate beyond its training data and therefore reports honest
uncertainty, so a confidence filter works. **For continuously-running
recognizers, calibration matters as much as peak accuracy.**

### 6.5 Cross-signer generalization gap

| condition | accuracy |
|---|---|
| held-out takes, known signer | ~96% |
| held-out **signer** (unseen person) | **~63%** |

Consistent with the 10–30% drops reported for higher-resource sign languages.

### 6.6 Multi-signer training

| trained on | overall | on signer A | on signer B |
|---|---|---|---|
| signer A only | 80.6% | **98.0%** | 60.5% |
| signer B only | 78.1% | 52.0% | **97.2%** |
| **both** | **96.6%** | 96.0% | 97.2% |

Adding a second signer cost signer A 2 points and gained signer B 37. **A single
model serves multiple signers at negligible cost; per-signer models are
unnecessary.**

### 6.7 Algorithm rankings do not transfer across problems

Same 8 algorithms, two datasets:

| algorithm | KSL landmarks (accuracy) | Sign Language MNIST images (accuracy) |
|---|---|---|
| logistic regression | **97.1%** (1st) | 70.2% (7th) |
| random forest | 96.9% | 82.5% (3rd) |
| gradient boosting | 95.0% | 79.8% |
| LDA | 94.8% | **43.3% (8th)** |
| GRU | 94.3% | 83.0% (2nd) |
| MLP | 94.0% | 74.0% |
| SVM | 94.0% | **83.7% (1st)** |
| k-NN | 83.8% (8th) | 80.6% (4th) |

LDA leads on cross-signer landmark generalization (62.4%) yet collapses on raw
pixels; k-NN is worst on landmarks and mid-table on images. **Benchmarking on one
dataset does not justify an algorithm choice on another**, even within the same
application domain.

### 6.8 Negative results worth reporting

- **Low-light enhancement** (CLAHE + gray-world white balance + camera
  brightness/gamma) roughly doubled hand detection (33.3%→57.3% warm bulb;
  42.0%→53.6% white bulb) but remained insufficient; the feature was removed.
- **Exposure capping to reduce motion blur backfired badly**: pose detection fell
  from 100% to **0%**, because the webcam exposes no `gain` control, so a shorter
  shutter only removes light. On such sensors brightness must be added via
  `brightness`/`gamma`, never by shortening exposure.
- **A "not-a-sign" negative class** (trained on partial windows and static holds)
  cut simulated flicker from 1.15 to 0.04 label changes per sign and held
  accuracy at 96.7% when masked at commit time — but felt *worse* in real use and
  was reverted. Offline simulation of live behaviour proved an unreliable proxy.
- **Deep models did not help at this scale**: the TCN memorized (100% train
  accuracy by epoch 45 on 30 takes/class) and underperformed random forest live.

---

## 7. Novel or unusual aspects (candidates for a contribution claim)

1. **Explicit demonstration that single-signer evaluation inverts the measured
   sign of an augmentation technique** (−7.3 → +5.4 points), with the same code,
   data and models. A concrete, reproducible methodological warning.
2. **Parametric skeletal signer synthesis with provable sign preservation** —
   joint angles invariant to 0.0004°, versus GAN/diffusion appearance-transfer
   approaches that offer no such guarantee.
3. **An augmentation-magnitude bound derived from inter-class distance**
   (synthetic drift must stay well under the between-class distance; measured
   failure above ~50% jitter).
4. **IK-based retargeting motivated by sign-language phonology** — preserving
   *location* (a phonemic parameter) rather than naively scaling limbs.
5. **Calibration-aware model selection for streaming recognition**, with
   measured out-of-distribution confidence per model family.
6. **Take-aware data splitting for augmented datasets**, including the failure
   mode where deleting a source sample orphans its augmented children and
   silently removes them from training.
7. **A complete low-resource pipeline**: dataset creation tooling, augmentation,
   training, evaluation and live inference for a language with no public corpus.

---

## 8. Known limitations

1. **2D body landmarks only** (`z = 0`) — camera pitch is indistinguishable from
   body proportion; only hands carry relative depth.
2. **Two signers** — cross-signer figures rest on one held-out person per
   direction.
3. **7-sign lexicon** — confusion grows with vocabulary size; results are not
   directly comparable to 100+ class benchmarks.
4. **Isolated signs only** — no continuous sentence segmentation or grammar.
5. **Non-manual markers ignored** — facial expression and mouthing are
   grammatically significant in sign languages but not captured.
6. **Same-session recording** — training takes were recorded consecutively, so
   session-specific factors may be partly learned.
7. **Hand detection ~57%** limits any sign relying on fine handshape.

---

## 9. Suggested literature search terms

**Core:** isolated sign language recognition; skeleton-based sign language
recognition; signer-independent sign language recognition; low-resource sign
language; pose-based gesture recognition.

**Augmentation:** skeleton data augmentation; synthetic signer generation;
body-shape retargeting; motion retargeting; SMPL/SMPL-X shape parameters;
adversarial skeleton augmentation; sign language production for augmentation.

**Methodology:** leave-one-subject-out evaluation; subject-independent
evaluation; data leakage in augmented datasets; group-aware cross-validation;
domain shift across subjects.

**Calibration / streaming:** out-of-distribution confidence; neural network
overconfidence; calibration of classifiers; continuous gesture spotting; gesture
segmentation; sliding-window action recognition; open-set recognition.

**Reference points:** AUTSL, WLASL, MS-ASL, LSA64, CSL-500, PHOENIX-2014T;
SAM-SLR (CVPR 2021 winner); Sign Language MNIST.

**Named specifics:** RTMPose; COCO-WholeBody; MediaPipe Holistic; One-Euro
filter; temporal convolutional network; Cambodian/Khmer Sign Language.

---

## 10. Ready-to-use prompts

> **Positioning.** "Here is a project brief for a Khmer Sign Language recognition
> system. Find recent papers (2020+) on skeleton-based signer-independent sign
> language recognition and synthetic skeletal augmentation. For each, state
> whether it evaluates with leave-one-signer-out, and whether our finding in §6.1
> (single-signer evaluation inverting the measured effect of augmentation) has
> been reported before. Cite sources."

> **Novelty check.** "Given §7, assess which claims are genuinely novel versus
> already established. Be skeptical and cite counter-examples where they exist."

> **Related-work section.** "Write a 600-word related-work section covering
> skeleton-based SLR, augmentation for low-resource sign languages, and
> subject-independent evaluation, positioning this project against them."

> **Abstract.** "Write a 250-word conference abstract from sections 1–6,
> emphasising the methodological finding rather than the accuracy numbers."

> **Critique.** "Act as a reviewer. Identify the three weakest claims and what
> additional experiments would be needed to support them."

---

## 11. Headline numbers (quick reference)

| metric | value |
|---|---|
| Signs / signers / real takes | 7 / 2 / 420 |
| Accuracy, known signer | ~96–97% |
| Accuracy, unseen signer | ~63% |
| Best cross-signer algorithm | LDA (62.4%), random forest (59.3%) |
| Synthetic augmentation, cross-signer | **+5.4 points** (helped 5/8, hurt 0/8) |
| Synthetic augmentation, same-signer | −7.3 points |
| Camera-roll correction | +6.3 points at 30° tilt |
| Hand vs body detection | 57% vs ~100% |
| Live inference | RTMPose 31.7 fps (GPU); MediaPipe ~15 fps (CPU-bound) |
| Sign Language MNIST best | SVM 83.7% (24 classes, 27,455 train images) |
