# SignLink — System Architecture

Detailed reference for presentations and code review. This is the document that defends the engineering choices.

---

## 1. Problem Statement

Khmer Sign Language (KSL) has **no automatic recognition system, no public dataset, and no published benchmark**. Deaf KSL users have no real-time text translation tool of any kind. SignLink is a research prototype that addresses this gap for an initial vocabulary of 5 isolated signs.

---

## 2. Research Question

> **Can a Transformer trained on skeleton sequences generalise to KSL signers it has never seen before, using a small set of real recordings from ≤8 signers?**

**Target:** ≥90% leave-one-signer-out accuracy on 5 isolated signs (`Eat`, `Hello`, `No`, `Thank you`, `Yes`).

**Why this is publishable:**
- No prior KSL recognition work exists
- Cross-signer generalisation is the standard SLR research challenge
- Method is reproducible (uses public libraries)
- Result is measurable (LOSO accuracy)

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         END USER (signer)                        │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ webcam feed (1280×720 @ 30fps)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CAPTURE LAYER  (Windows)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  OpenCV     │→ │ MediaPipe    │  │ RTMPose                 │  │
│  │  camera I/O │  │ Hands        │  │ body keypoints (CUDA)   │  │
│  └─────────────┘  │ 21+21 joints │  │ 6 joints (shoulders,    │  │
│                   │              │  │ elbows, wrists)         │  │
│                   └──────────────┘  └─────────────────────────┘  │
│                                 │                                │
│                                 ▼                                │
│           Combined skeleton: 48 joints × 3 coords                │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ (60 frames × 48 joints × 3) per take
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     NORMALISATION LAYER                          │
│                                                                  │
│       ┌────────────────────┐    ┌────────────────────────┐       │
│       │  CLEAN view        │    │  NOISY view            │       │
│       │  shoulder-anchored │    │  raw [0,1] image space │       │
│       │  scale-normalised  │    │  preserves position    │       │
│       │  signer-invariant  │    │  + scale variation     │       │
│       └────────┬───────────┘    └───────────┬────────────┘       │
└────────────────┼────────────────────────────┼────────────────────┘
                 │  paired (clean, noisy) — free 2× augmentation   │
                 ▼                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                              │
│                                                                  │
│   Local disk:  data/sequences_v2/<label>/<signer>__real__       │
│                  <view>__<NNNN>.npy + .json                     │
│                                                                  │
│   Google Drive: SignLink/data/sequences_v2/                     │
│                  (mirrored via rclone sync)                     │
│                                                                  │
│   GitHub:       code only — .npy and .pt are gitignored         │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ Colab pulls data + code
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER  (Google Colab T4 GPU)      │
│                                                                  │
│       data/sequences_v2/  ──▶  PyTorch DataLoader               │
│                                  │  (with augmentation)         │
│                                  ▼                               │
│                          ┌───────────────────┐                   │
│                          │  SignTransformer  │                   │
│                          │  4 layers · 8 heads                   │
│                          │  d_model = 256    │                   │
│                          │  2.16M params     │                   │
│                          └─────────┬─────────┘                   │
│                                    │                             │
│                AdamW + CosineAnnealingLR + CrossEntropyLoss      │
│                       100 epochs · best-checkpoint save          │
│                                    │                             │
│                                    ▼                             │
│                  models/weights_v2/ksl_transformer_latest.pt    │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ pulled to laptop / web app
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     INFERENCE LAYER  (planned, Task 5)           │
│                                                                  │
│   webcam → capture pipeline → 60-frame buffer → SignTransformer  │
│                                                       │          │
│                                                       ▼          │
│                              predicted sign + confidence score   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
┌────────────────────────┐              ┌────────────────────────┐
│  VISUALISATION LAYER   │              │  APPLICATION LAYER     │
│  (browser, working)    │              │  (planned, Tasks 3+4)  │
│                        │              │                        │
│  MediaPipe Tasks API   │              │  React frontend        │
│  → Kalidokit IK        │              │  ⇆ FastAPI backend     │
│  → three-vrm avatar    │              │  ⇆ inference engine    │
│                        │              │                        │
│  mirror feedback for   │              │  user-facing demo:     │
│  the signer            │              │  text + correction UI  │
└────────────────────────┘              └────────────────────────┘
```

---

## 4. Component Detail

### 4.1 Capture layer

**Input:** webcam frames at 1280×720 @ 30 fps.

**Processing:**
- **MediaPipe Hands** (CPU): detects 21 landmarks per hand, in normalised image space `[0, 1]`. Used for both hands.
- **RTMPose** (GPU via CUDA): detects body keypoints in pixel space, converted to `[0, 1]`. We only retain 6 joints: left/right shoulder, elbow, wrist.
- Combined output per frame: **48 joints × (x, y, z) = 144 features**.

**Output per take:** a sequence of 60 frames (2 seconds of motion).

**Code:** `src/capture.py`, `scripts/record_session.py`

### 4.2 Normalisation layer

Each take produces **two paired views**:

| View | Coordinate space | Purpose |
|---|---|---|
| **CLEAN** | shoulder-anchored, scaled by shoulder width | Strips signer body proportions — model learns motion patterns invariant to body size |
| **NOISY** | raw image space `[0, 1]` | Preserves natural variation in position and scale — model learns to handle real-world variation |

Treating the two views as independent training samples gives **free 2× augmentation** without re-recording. This is the principle paired-view augmentation: same motion, two semantic views.

**Code:** `src/v2/normalize.py`, `src/v2/schema.py`

### 4.3 Storage layer

| Tier | Stores | Why |
|---|---|---|
| **Local disk** | `data/sequences_v2/<label>/*.npy` | Fast iteration during recording |
| **Google Drive** | `SignLink/data/sequences_v2/` | Team-wide shared dataset |
| **GitHub** | code only | Data is too large + privacy-sensitive |

**Filename format** (`<signer>__<source>__<view>__<variant>.npy`) embeds metadata so the dataset can be filtered by signer or source without external indexing. Example:

```
data/sequences_v2/Hello/
├── Piseth__real__clean__0000.npy   ← shape (60, 48, 3) float32
├── Piseth__real__clean__0000.json  ← signer/source/view metadata
├── Piseth__real__noisy__0000.npy
└── Piseth__real__noisy__0000.json
```

**Sync tool:** `rclone sync` (via `scripts/drive_sync.py`) — propagates deletes, ensures Drive matches local.

### 4.4 Training layer

Runs on **Google Colab** with a T4 GPU. The 6-cell notebook (`notebooks/colab_train_v2.py`) handles:

1. Clone repo from GitHub
2. Mount Drive
3. Install dependencies (mostly pre-installed)
4. Sync `data/sequences_v2/` from Drive to Colab disk
5. Train the model — runs `src/v2/train.train(cfg)`
6. Push weights and training logs back to Drive

**Why Colab:**
- Free GPU access for student team
- No local PyTorch/CUDA install required for teammates
- Reproducible: every team member gets the same environment

### 4.5 Model: SignTransformer

```
Input  (B, 60, 144)  ← batch of skeleton sequences
   │
   ├─ Linear projection 144 → 256
   ├─ + learned positional embedding (60 × 256)
   │
   ├─ 4 × TransformerEncoderLayer
   │      • 8-head self-attention
   │      • feedforward 256 → 512 → 256 (GELU)
   │      • pre-LayerNorm, dropout 0.1
   │
   ├─ Mean-pool across 60 timesteps → (B, 256)
   │
   └─ Linear classifier 256 → num_classes
Output (B, num_classes)
```

**Key choices and why:**

| Choice | Reason |
|---|---|
| Transformer (not RNN/CNN) | Better at long-range temporal dependencies; standard in modern SLR work (Camgöz et al., 2020) |
| 4 layers | Trades capacity for sample efficiency at small data scale |
| 8 attention heads | Empirically robust for this size |
| Pre-LayerNorm | More stable training than post-LN |
| Mean pooling | Simpler than `[CLS]` token; works well for fixed-length classification |
| 2.16M parameters | Small enough to train on hundreds of samples without overfitting |

**Training:**
- **Loss:** `CrossEntropyLoss`
- **Optimiser:** `AdamW` (Adam + decoupled weight decay 1e-4)
- **LR schedule:** `CosineAnnealingLR` from 1e-3 → 0 over 100 epochs
- **Augmentation:** time warp, scale jitter, z-axis rotation, Gaussian noise, horizontal flip (training only)
- **Validation:** either random 85/15 split (smoke test) or **leave-one-signer-out (LOSO)** (research-grade)
- **Checkpointing:** save only when val accuracy improves

**Code:** `src/v2/model_transformer.py`, `src/v2/train.py`, `src/v2/dataset.py`, `src/v2/augment.py`

### 4.6 Visualisation layer (working)

Real-time avatar mirror to give the signer visual feedback during recording or demo.

**Stack:**
- **MediaPipe Tasks API** (browser) — `PoseLandmarker` + `HandLandmarker` running on the GPU via WebGL
- **Kalidokit** — converts landmark coordinates to bone rotations using inverse kinematics
- **three-vrm** — loads a VRM avatar
- **three.js** — WebGL renderer with `OrbitControls`

**Defensive engineering:**
- Visibility filter — joints with low MediaPipe visibility are skipped
- Anti-jump filter — per-frame rotation deltas above 60° are rejected (catches IK ambiguity flips when wrist crosses body midline)
- Hip-rotation disabled — avatar stays facing camera; Kalidokit's hip flip behaviour was unreliable

**Code:** `mannequin_web/index.html` (single HTML file, ~330 lines)

### 4.7 Inference layer (planned, Task 5)

To be built on top of `src/capture.py`:
- Continuous capture loop
- 60-frame circular buffer
- Every N frames: pass buffer to `SignTransformer`, output predicted label + softmax probabilities
- Overlay prediction on camera window

### 4.8 Application layer (planned, Tasks 3 + 4)

Web app combining inference and visualisation:
- **Frontend (React):** webcam feed on left, predicted text on right, confidence bar, correction button
- **Backend (FastAPI):** receives MediaPipe landmarks via WebSocket, runs the model, returns prediction; `/correct` endpoint logs misclassifications for future retraining

---

## 5. Data Contract

The interface between recording, training, and inference. Once locked, all three sides depend on it being stable.

```
data/sequences_v2/
└── <label>/
    ├── <signer>__real__clean__NNNN.npy    (60, 48, 3) float32
    ├── <signer>__real__clean__NNNN.json   metadata
    ├── <signer>__real__noisy__NNNN.npy    (60, 48, 3) float32
    └── <signer>__real__noisy__NNNN.json
```

**Joint order (48 total):**
1. left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist (6 body)
2. left hand: 21 MediaPipe Hands landmarks (wrist + 4 per finger × 5 fingers)
3. right hand: 21 MediaPipe Hands landmarks

**Metadata JSON:** `{ label, signer_id, source, view, fps, variant, notes }` — enables querying by signer / source / view without parsing filenames.

---

## 6. Tech Stack Summary

| Layer | Technology |
|---|---|
| Camera I/O | OpenCV |
| Hand pose | MediaPipe Hands (CPU) |
| Body pose | RTMPose via `rtmlib` (CUDA) |
| Data format | NumPy `.npy` + JSON |
| Sync | `rclone sync` to Google Drive |
| Code hosting | GitHub |
| Model framework | PyTorch |
| Architecture | Transformer encoder |
| Training compute | Google Colab T4 |
| Visualisation | MediaPipe Tasks API + Kalidokit + three-vrm + three.js |
| Web stack (planned) | FastAPI + React + WebSocket |

All components are open-source, peer-reviewed or industry-standard, and free at student scale.

---

## 7. Current State

| Component | Status | Notes |
|---|---|---|
| Camera capture | ✅ working | RTMPose + MediaPipe Hands integrated |
| Session recorder | ✅ working | streaming session, paired clean+noisy |
| Normalisation | ✅ working | shoulder-anchored + raw views |
| Drive sync | ✅ working | rclone configured on lead's machine |
| Training notebook | ✅ working | 6-cell Colab, runs end-to-end |
| SignTransformer | ✅ working | smoke-tested, converges in <10s on small data |
| First trained weights | ✅ available | solo-signer baseline (Piseth) |
| VRM mannequin | ✅ working | browser-based, 22+ fps |
| Real-time inference | ❌ not started | Task 5 |
| Web app frontend | ❌ not started | Task 3 |
| Web app backend | ❌ not started | Task 4 |
| Multi-signer dataset | ⏳ in progress | other team members recording |
| Real LOSO results | ⏳ blocked on data | requires ≥3 signers |

---

## 8. Roadmap

```
TERM 1  (now → next week)
└─ Foundation: pipeline + smoke test + presentation

TERM 2  (next 12 weeks)
├─ All 8 teammates record (target ~120 takes total)
├─ First real LOSO benchmark
├─ Real-time inference (Task 5)
├─ Web app skeleton (Tasks 3 + 4)
└─ Mid-term demo

TERM 3  (final term of Year 2)
├─ Iterate to ≥90% LOSO
├─ Confidence + correction loop (Task 8)
├─ Final paper draft
└─ Final presentation to Dr. May Thu

YEAR 3
├─ Term 1: Khmer text-to-speech extension
├─ Term 2: vocabulary expansion + continuous signing
└─ Term 3: deployment + paper submission
```

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Signers perform different versions of the same KSL sign | Task 6 — lock authoritative reference videos before any team-wide recording |
| Tracker fails when signer is partly out of frame | Visibility filter on the visualisation layer; recording instructions specify framing |
| Single-signer overfit | LOSO is the only metric we report — random splits are smoke tests only |
| Mannequin polish eats research time | Mannequin is feedback only; not in the LOSO accuracy loop |
| Synthetic data pipeline (Godot) | Descoped from Year 2; only attempted if real data alone falls short |

---

## 10. What to Show Dr. May (Term 1 presentation)

1. **Slide 1** — Title + research question
2. **Slide 2** — Problem (no KSL recognition exists today)
3. **Slide 3** — High-level architecture diagram (section 3 of this doc)
4. **Slide 4** — Capture pipeline detail (section 4.1)
5. **Slide 5** — Why skeleton-based (section 4.2 — paired views)
6. **Slide 6** — SignTransformer architecture (section 4.5)
7. **Slide 7** — Training procedure + smoke test result *(report honestly: "smoke test only — single signer, real LOSO is Term 2")*
8. **Slide 8** — Visualisation mannequin live demo
9. **Slide 9** — 8-task team plan + dependency map
10. **Slide 10** — Term 2 / Term 3 roadmap
11. **Slide 11** — Risks + mitigations
12. **Slide 12** — Closing: what we expect to deliver by end of Year 2

Honesty about the smoke test result is a strength, not a weakness. Reviewers respect *"we know what we don't know"* far more than overclaimed accuracy on a tiny val set.
