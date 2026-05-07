# SignLink — project overview

Year-2 research project at CamTech University.


---

## What we are building

A system that recognises isolated Khmer Sign Language (KSL) signs from a webcam
and converts them to text in real time. The end product is a web app:
camera on the left, predicted sign as text on the right.

The research question:

> **Can a Transformer trained on skeleton sequences generalise to signers
> it has never seen before — using a small set of real recordings
> plus synthetic data generated from Godot?**

Target: **10 signs, ≥90% accuracy on a held-out signer**.

---

## How the full system works

```
Webcam
  │
  ▼
MediaPipe + RTMPose ──────────────────── 48 joints per frame (x, y, z)
  │                                      6 body + 21 left hand + 21 right hand
  ▼
Normalize
  ├── CLEAN  (shoulder-anchored, signer-invariant)
  └── NOISY  (raw [0,1] image space, preserves signer variation)
  │
  ▼
data/sequences_v2/<label>/
  ├── signer__real__clean__0000.npy   shape (60, 48, 3)
  └── signer__real__noisy__0000.npy   shape (60, 48, 3)
  │
  ▼
Google Drive  ──▶  Colab training
                    SignTransformer: 4 layers · 8 heads · d_model 256 · 2.16M params
                    100 epochs · AdamW · CosineAnnealingLR
  │
  ▼
ksl_transformer_latest.pt
  │
  ├──▶  Real-time inference  ──▶  predicted sign + confidence score
  │
  └──▶  FastAPI backend  ──▶  WebSocket  ──▶  React frontend
                                               webcam + predicted text + correction
```

---

## Current state

| Component | Status |
|---|---|
| Camera capture (MediaPipe + RTMPose) | ✅ working |
| Session recorder (Windows) | ✅ working |
| Clean / noisy normalization | ✅ working |
| Drive sync (rclone) | ✅ lead machine only |
| Colab training notebook | ✅ built, not yet run |
| SignTransformer architecture | ✅ built, not yet trained |
| Godot live mannequin | ⚠️ moves but mapping is wrong |
| Real-time inference (v2) | ❌ not built |
| Web app | ❌ not built |
| Trained weights | ❌ no data collected yet |

**The immediate blocker is data. Nothing trains until people record.**

---

## Data pipeline

### Record on Windows
```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python scripts\record_session.py --signer yourname --no-stream
```

Signs to record — do not change this list without team agreement:
`hello` `thanks` `yes` `no` `please` `sorry` `good` `bad` `family` `name`

Target: **10 signs × 3 takes minimum per person.**

Session controls:
```
hello   [ENTER]  →  GET READY 1.5s  →  REC 2.0s  →  saved
        [ENTER]  →  another take of the same sign
thanks  [ENTER]  →  switch to a different sign
quit    [ENTER]  →  end session
```

### Delete a bad take
```powershell
ls data\sequences_v2\hello                        # see what you have
Remove-Item data\sequences_v2\hello\*__0001.*     # delete one specific take
Remove-Item -Recurse data\sequences_v2\hello      # delete all takes of a sign
Remove-Item -Recurse data\sequences_v2\*          # wipe everything
```

### Upload to Drive
Open `SignLink/data/sequences_v2/` in your browser.
Open `D:\Projects\Sign to Text\khmer_sign_recognizer\data\sequences_v2\` in File Explorer.
For each sign folder — open or create the matching folder in Drive, drag all 4 files in.

### Train on Colab
Open the Colab link from the lead. Run cells 1–6 in order.
Weights land in `SignLink/models/weights_v2/` automatically.

---

## Data contract

Every recording produces two paired files per take:

```
data/sequences_v2/
└── <label>/
    ├── <signer>__real__clean__0000.npy    shape (60, 48, 3) float32
    ├── <signer>__real__clean__0000.json
    ├── <signer>__real__noisy__0000.npy    shape (60, 48, 3) float32
    └── <signer>__real__noisy__0000.json
```

- `clean` = shoulder-anchored, scaled by shoulder width — signer-invariant
- `noisy` = raw [0,1] image-space coords — preserves natural variation

Joint order (48 total): L/R shoulder, L/R elbow, L/R wrist → left hand 21 joints → right hand 21 joints.

---

## Hard rules

1. **Commit before you train.** Colab pulls from GitHub — uncommitted code is invisible.
2. **Signer tag on every sample.** Use the same short lowercase tag every time you record.
3. **Code in Git. Data and weights in Drive.** Never commit `.npy` or `.pt` files.
4. **Do not add new signs without team agreement.** New sign = retrain from scratch.
5. **Do not edit `src/v2/` without talking to the lead.**

---

## The 8 tasks

Pick the one that interests you. Write a research report on your approach
before you build anything.

---

### Task 1 — Data recording
Record all 10 signs on your machine (3 takes minimum per sign) and upload
to the shared Drive. Follow the data pipeline above.
**This unblocks everything — training cannot start until at least 3 people have recorded.**

---

### Task 2 — Godot investigation
The mannequin mirrors movement but bone mapping and coordinate transforms are wrong.
Read `src/mapper.py` and `khmer-sign-mannequin2/`.
Write a report answering one question: **is it fixable in reasonable time,
or do we replace the Godot approach entirely?** Either answer is valid.

---

### Task 3 — Web app frontend
React app with live webcam on the left, predicted sign text on the right.
Shows confidence score. Has a correction button for wrong predictions.
Connects to the backend via WebSocket.

---

### Task 4 — Web app backend
FastAPI server that receives landmark data over WebSocket, runs it through
the trained model, and returns the predicted sign with a confidence score.
Expose a `/correct` endpoint so the frontend can flag wrong predictions.

---

### Task 5 — Real-time inference
Script that reads live camera frames, buffers the last 60 frames,
runs the v2 Transformer, and overlays the predicted sign and confidence
on the camera window. Built on top of `src/capture.py`.
Becomes the core of Task 4.
*Needs trained weights first.*

---

### Task 6 — KSL sign validation
Find authoritative sources for the 10 signs — Khmer deaf associations,
professor references, KSL dictionaries or videos.
Are these the correct standardized signs? Are there regional variations?
Produce a reference sheet with photos or video links that every signer
uses before recording. **Wrong signs recorded = wrong model.**

---

### Task 7 — Sign-to-text output design
After the model predicts a sign, what happens?
Text on screen? Multiple signs building into a sentence? Khmer TTS?
Research what sign-to-text UX looks like for deaf users, what similar apps
do, and write a design proposal that the web app team implements.

---

### Task 8 — Prediction confidence + correction loop
How do we handle uncertain predictions?
What threshold before we show a label?
What does the user see when confidence is low?
Can a user correction feed back into retraining?
Write a proposal covering the full correction loop.
*Needs trained weights first.*

---

## Dependency map

```
Task 6 (KSL vocab)   ──▶  Task 1 (record)  ──▶  training  ──▶  Task 5 (inference)
                                                           └──▶  Task 8 (confidence)

Task 7 (output design)  ──▶  Task 3 (frontend)
                         └──▶  Task 4 (backend)  ◀──  Task 5 (inference)

Task 2 (Godot)  ──  independent
```

Tasks that can start immediately: **1, 2, 3, 4, 6, 7**
Tasks that need training first: **5, 8**
