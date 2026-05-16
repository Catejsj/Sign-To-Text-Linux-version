# SignLink — Term 1 Presentation Material

Drop-in slide content for the Week-1 review with Dr. May Thu and the team.
Copy each section into a slide deck (Google Slides / PowerPoint / Keynote).
Diagrams below are ASCII; convert to draw.io / Excalidraw / Figma for visual polish.

---

## Slide 1 — Title

```
SignLink
Khmer Sign Language Recognition with
Skeleton-Based Temporal Convolutional Networks

Year-2 Research Project — CamTech University
Mentor: Dr. May Thu | Lead: Seng Piseth
Term 1 Review
```

---

## Slide 2 — The Problem

- Khmer Sign Language (KSL) has **no automatic recognition system**, no public dataset, and no published benchmark.
- Deaf KSL users have no real-time text translation tool.
- We're building one of the first.

> "To the best of our knowledge, no prior work addresses automatic recognition of Khmer Sign Language."

---

## Slide 3 — Research Question

> **Can a Temporal Convolutional Network trained on skeleton sequences generalise to KSL signers it has never seen before, using a small set of real recordings from ≤8 signers?**

**Year-2 target:** ≥90% leave-one-signer-out accuracy on **5 isolated signs**:
`Eat`, `Hello`, `No`, `Thank you`, `Yes`.

---

## Slide 4 — System Architecture (high level)

```
┌──────────────────────────────────────────────────────────┐
│                       SIGNER                             │
└─────────────────────────┬────────────────────────────────┘
                          │ webcam
                          ▼
            ┌───────────────────────────────┐
            │  POSE EXTRACTION              │
            │   MediaPipe Hands + Pose      │
            │   → 48 joints × 3 coords      │
            └─────────────┬─────────────────┘
                          ▼
            ┌───────────────────────────────┐
            │  NORMALISATION                │
            │   • CLEAN (signer-invariant)  │
            │   • NOISY (raw image space)   │
            └─────────────┬─────────────────┘
                          ▼
            ┌───────────────────────────────┐
            │  STORAGE                      │
            │   Local → Google Drive        │
            │   GitHub (code only)          │
            └─────────────┬─────────────────┘
                          ▼
            ┌───────────────────────────────┐
            │  TRAINING (Colab T4 GPU)      │
            │   SignTCN — main model        │
            │   Transformer — baseline      │
            └─────────────┬─────────────────┘
                          ▼
            ┌───────────────────────────────┐
            │  WEB APP (deployment)         │
            │   browser landmarks → REST    │
            │   → predicted sign as text    │
            └───────────────────────────────┘
```

Caption: *Five layers — capture, normalise, store, train, deploy.*

---

## Slide 5 — Why Skeleton, Not Video

**Raw video is brittle.** Lighting, skin tone, clothing, and background change every recording. A model trained on raw video tends to learn the *room*, not the *sign*.

**Skeleton sequences strip all that away.** What remains is pure motion.

```
Video                     Skeleton
─────                     ────────
+ all visual context      + invariant to lighting
+ photo-realistic         + invariant to skin tone
- needs huge datasets     + tiny payload (60×144 floats)
- privacy concern         + privacy-friendly (no faces stored)
- compute heavy           + runs on a phone GPU
```

**Paired views as built-in augmentation:**

- **CLEAN:** shoulder-anchored, scaled by shoulder width — learns signer-invariant motion
- **NOISY:** raw image-space — learns positional variation
- One recording session = two training samples = free 2× augmentation.

---

## Slide 6 — The Model: SignTCN

```
Input  (B, 60, 144)              ← 60 frames × 144 features
   │
   ├─ Linear projection 144 → 128
   │
   ├─ 4 × Residual Temporal Conv block
   │     • Conv1d kernel=5, dilation 1, 2, 4, 8
   │     • BatchNorm + GELU + Dropout
   │
   ├─ Global mean + max pooling over time → (B, 256)
   │
   └─ MLP head 256 → 128 → num_classes
Output (B, num_classes)

≈ 700K parameters
```

### Why TCN, not Transformer?

| | Transformer encoder | SignTCN ✅ |
|---|---|---|
| Parameters | 2.16M | ~0.7M |
| Inductive bias | Weak — needs lots of data | Strong — perfect for ≤1k samples |
| Training time on T4 | seconds | seconds |
| Deploys to web | harder | easier (smaller) |
| Risk at our scale | higher | lower |

**Plain English:** the TCN is smaller, faster, and gives the model a built-in assumption that *nearby frames matter most* — which is exactly what sign motion is. Less data hunger, less overfitting risk.

The Transformer remains as a published baseline for the paper.

---

## Slide 7 — Capture & Normalisation Detail

**Pose extraction stack:**
- **MediaPipe Hands** → 21 landmarks per hand
- **MediaPipe Pose** → 6 upper-body joints (shoulders, elbows, wrists)
- **Total: 48 joints × (x, y, z) = 144 features per frame**
- 60 frames = 2 seconds at 30 fps

**Per-frame metadata:**
- visibility flag — was MediaPipe confident?
- presence flag — was the joint in frame?

These let downstream code drop unreliable frames.

---

## Slide 8 — Data Pipeline & Storage

```
┌──────────────┐      record      ┌─────────────────────┐
│   Signer     │ ───────────────► │  data/sequences_v2/ │
│   webcam     │                  │  (local NPY + JSON) │
└──────────────┘                  └────────┬────────────┘
                                           │ rclone sync
                                           ▼
                                  ┌─────────────────────┐
                                  │  Google Drive       │
                                  │  SignLink/data/     │
                                  └────────┬────────────┘
                                           │ Colab pulls
                                           ▼
                                  ┌─────────────────────┐
                                  │  Colab GPU train    │
                                  │  saves .pt to Drive │
                                  └─────────────────────┘
```

**Filename = built-in metadata:**
```
<signer>__<source>__<view>__<NNNN>.npy
e.g.   Piseth__real__clean__0003.npy
```

No external database needed; the dataset is queryable by filename alone.

---

## Slide 9 — Training Strategy

- **Optimiser:** AdamW (lr 1e-3, weight decay 1e-4)
- **Schedule:** CosineAnnealingLR over 100 epochs
- **Loss:** Cross-Entropy
- **Augmentation:** time warp, scale jitter, z-rotation, Gaussian noise, joint dropout
- **Validation:**
  - **Smoke test (Term 1):** random 85/15 split — proves pipeline works
  - **Research-grade (Term 2+):** **Leave-One-Signer-Out** — proves cross-signer generalisation

---

## Slide 10 — Term 1 Smoke Test (Honest Result)

```
Setup:
  1 signer (Piseth)   |   30 takes   |   60 samples

Result:
  best val acc 1.0
  trained in <10 seconds on T4 GPU
```

**What this proves:**
✅ End-to-end pipeline runs (capture → train → save weights)
✅ Architecture converges
✅ Augmentation doesn't break training

**What this does NOT prove:**
❌ Generalisation to other signers
❌ Real-world accuracy

> "The 100% number is a smoke-test artifact, not a research result. The val set is 4 samples from one signer. Multi-signer LOSO is our Term 2 deliverable."

---

## Slide 11 — Visualisation Layer (mannequin)

```
            Browser
            ┌─────────────────────────────────────┐
            │  webcam   →  MediaPipe Tasks API    │
            │              (running on RTX 4060)  │
            │                  │                  │
            │                  ▼                  │
            │            Kalidokit IK             │
            │                  │                  │
            │                  ▼                  │
            │       three-vrm avatar (VRM 1.0)    │
            └─────────────────────────────────────┘
```

**Purpose:** real-time signer feedback during recording — confirms the system is tracking the signer correctly.

**Not in the inference loop.** This is visualisation only; the recognition model uses raw skeleton coords.

(Live demo at this slide.)

---

## Slide 12 — Team Plan (8 People, 8 Tasks)

```
Task 1  Data recording + quality check               ← everyone
Task 2  Mannequin investigation / final fix          ← solo
Task 3  Web app frontend (React)                     ← solo
Task 4  Web app backend (FastAPI)                    ← solo
Task 5  Real-time inference script                   ← solo
Task 6  KSL sign validation + reference videos       ← solo (priority!)
Task 7  Sign-to-text output design                   ← solo
Task 8  Prediction confidence + correction loop      ← solo
```

**Critical path:** Task 6 (sign validation) → Task 1 (recording) → training → Tasks 5 + 8 → Tasks 3 + 4.

---

## Slide 13 — Roadmap

```
┌────────────────────────────────────────────────────────┐
│ TERM 1  (now → next week)                              │
│   ✓ Pipeline built end-to-end                          │
│   ✓ Smoke test trained                                 │
│   ✓ Mannequin live demo                                │
│   ✓ Team plan + architecture doc                       │
└────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ TERM 2  (next 12 weeks)                                │
│   • All 8 teammates record (~120 takes)                │
│   • Real-time inference script                         │
│   • First real LOSO benchmark (target 70-80%)          │
│   • Web app skeleton                                   │
└────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ TERM 3  (final term of Year 2)                         │
│   • Iterate to ≥90% LOSO                               │
│   • Confidence + correction loop                       │
│   • Paper draft                                        │
│   • Final demo to Dr. May                              │
└────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ YEAR 3 (Sign-to-Speech extension)                      │
│   T1: Khmer TTS layer                                  │
│   T2: vocabulary expansion + continuous signing        │
│   T3: deployment + paper submission                    │
└────────────────────────────────────────────────────────┘
```

---

## Slide 14 — Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Different signers perform the same KSL sign differently | **Task 6 first** — lock authoritative reference videos before all team-wide recording |
| Tracker fails when signer is partly out of frame | Visibility filter + recording instructions specify framing |
| Single-signer overfit | LOSO is the only metric we report; random splits = smoke test only |
| Synthetic data (mannequin) eats time | Mannequin = visualisation only this year. No synthetic in inference loop. |
| Year-2 deadline slips | Tasks parallelised, 8-person team, Colab eliminates training infra |

---

## Slide 15 — Closing

**By end of Year 2 we will deliver:**

1. ✅ Working KSL recognition (camera → predicted sign as text)
2. ✅ Web app demo
3. ✅ Research paper documenting LOSO results on 5 signs
4. ✅ Open-source repo with full documentation

**Year 3:** extend to Khmer Sign Language → Speech.

---

## Q&A Prep — likely questions

**Q: Why a Transformer originally, then switch to TCN?**
A: We started with Transformer following recent SLR literature (Camgöz et al., 2020). On revisiting, TCN's stronger inductive bias is better-matched to our data scale (≤1k samples). We retain Transformer as a published baseline.

**Q: 100% val accuracy looks suspicious.**
A: Yes — it's a smoke-test result on 4 val samples from a single signer. The number proves the pipeline runs; it does not measure generalisation. Real LOSO on multi-signer data is the Term 2 deliverable.

**Q: How do you know the signs are correctly performed?**
A: Task 6 — KSL sign validation — has the team locking authoritative reference videos before mass recording. Every signer copies the same source.

**Q: Why not use a pre-trained model?**
A: No public KSL datasets or pretrained weights exist. ASL pretrained weights don't transfer because the signs differ. Year 3 we may explore ASL feature transfer.

**Q: What's the mannequin for?**
A: Real-time signer feedback during recording / demo. It confirms tracking is working. It is NOT in the inference path — recognition runs on raw skeleton coordinates.

**Q: Why MediaPipe and not RTMPose / OpenPose?**
A: MediaPipe runs in the browser via WebGL on the user's GPU. Using MediaPipe in both training and deployment eliminates train-deploy mismatch. RTMPose was used in our prototype but creates an inconsistency.

**Q: How big is your dataset right now?**
A: 30 takes (1 signer). Term 2 grows it to ~120 takes (8 signers × 3 takes × 5 signs). Small but sufficient for the 5-class isolated-sign problem.
