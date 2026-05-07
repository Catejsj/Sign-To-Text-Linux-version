# SignLink — Khmer Sign Language Recognizer

Year-2 research project at CamTech University.


---

## What is SignLink

SignLink is a system that recognises isolated Khmer Sign Language (KSL) signs
from a standard webcam and converts them to text in real time.

A person performs a sign in front of their camera. The system captures their
skeleton — 48 joints covering the body, both hands and wrists — normalises the
movement to remove differences between signers, and passes it through a
Transformer model that outputs the predicted sign as text.

The end product is a web app: camera feed on the left, predicted sign as text
on the right, with a confidence score and a way to correct wrong predictions.

---

## Research question

> **Can a Transformer trained on skeleton sequences generalise to signers
> it has never seen before — using a small set of real recordings
> plus synthetic data generated from Godot?**

We target **10 signs, ≥90% accuracy on a held-out signer**.

The key challenge is signer generalisation: a model that only works for
the person who recorded the training data is not useful. We measure this
with leave-one-signer-out validation — train on all signers except one,
test on the one left out.

---

## How the system works

```
Person performs a sign in front of webcam
  │
  ▼
MediaPipe + RTMPose extract 48 joints per frame
(6 body joints + 21 left hand + 21 right hand)
  │
  ▼
2 seconds of movement = 60 frames
Each frame normalised two ways:
  ├── CLEAN  — shoulder-anchored, scale-normalised (removes signer differences)
  └── NOISY  — raw image-space coords (keeps natural variation)
  │
  ▼
SignTransformer (2.16M parameters)
4 layers · 8 attention heads · trained on Colab
  │
  ▼
Predicted sign + confidence score
  │
  ▼
Web app displays text — user can correct wrong predictions
```

---

## Why skeleton, not video

Raw video is too sensitive to lighting, clothing, skin tone, and camera angle.
Skeleton sequences strip all of that away — what remains is pure movement.
This is what lets the model generalise across different people and environments.

The paired clean/noisy approach gives us free domain randomisation:
the clean view teaches the model signer-invariant patterns,
the noisy view teaches it to handle natural variation.
One recording session produces both automatically.

---

## Current state

| Component | Status |
|---|---|
| Camera capture (MediaPipe + RTMPose) | ✅ working |
| Session recorder | ✅ working |
| Clean / noisy normalisation | ✅ working |
| Drive sync and Colab training notebook | ✅ ready to run |
| SignTransformer model architecture | ✅ built, awaiting data |
| Godot live mannequin | ⚠️ moves but mapping is wrong |
| Real-time inference (v2) | ❌ not yet built |
| Web app | ❌ not yet built |
| Trained weights | ❌ data collection not started |

---

## Team and tasks

The project is split into 8 tasks — one per team member.
See [`WORKFLOW.md`](WORKFLOW.md) for the full task descriptions,
data pipeline, and dependency map.

New team members: read [`docs/TEAM_ONBOARDING.md`](docs/TEAM_ONBOARDING.md) first.

---

## Repo layout

```
khmer_sign_recognizer/
├── README.md                       this file — project overview
├── WORKFLOW.md                     tasks, data pipeline, hard rules
├── src/
│   ├── capture.py                  camera + MediaPipe + RTMPose
│   ├── send_to_wsl.py              UDP bridge to Godot
│   └── v2/
│       ├── schema.py               data contract (filenames, shapes)
│       ├── normalize.py            clean + noisy normalisation
│       ├── augment.py              training augmentations
│       ├── dataset.py              PyTorch dataset + train/val splits
│       ├── model_transformer.py    SignTransformer architecture
│       └── train.py                training loop + config
├── scripts/
│   ├── record_session.py           streaming recorder (paired clean+noisy)
│   └── drive_sync.py               rclone wrapper for Drive sync
├── notebooks/
│   └── colab_train_v2.py           Colab training script (6 cells)
├── khmer-sign-mannequin2/          Godot 4.6 Y-Bot mannequin
├── docs/
│   ├── TEAM_ONBOARDING.md          how teammates join and contribute
│   ├── COMMANDS.md                 all commands in one place
│   ├── FIRST_TIME_COLAB.md         lead setup guide
│   └── ARCHITECTURE.md             detailed architecture notes
├── config/settings.json            camera params, IPs, filter tuning
├── update_ips.py                   refresh WSL IPs after reboot
└── data/, models/, logs/           gitignored — synced via Drive
```
