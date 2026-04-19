# SignLink workflow (v2)

This file is the team map. If you are new to the project, read this first.

## Two tracks in one repo

We keep two working tracks so nobody blocks anybody else:

| Track           | Lives in                    | Owned by        |
|-----------------|-----------------------------|-----------------|
| **v2** (main)   | `src/v2/`, `notebooks/colab_train_v2.py` | Piseth + leads |
| **legacy CNN**  | `src/model.py`, `src/normalizer.py`, `src/dataset.py`, `notebooks/train_ksl.py` | anyone who wants to keep iterating on the CNN |

Both tracks share the `src/capture.py` pipeline and the Godot demo.
Never edit v2 files *and* legacy files in the same commit.

## Hard rules

1. **Commit before you train.** Colab pulls from GitHub; anything uncommitted is invisible to it.
2. **One data schema.** v2 samples are `(60, 48, 3)` float32 with a sibling `.json` describing signer and source. No exceptions.
3. **Signer tag on every sample.** No "anonymous" samples — we can't measure signer-generalization without the tag.
4. **Code lives in Git; data and weights live in Drive.** Never commit `.npy` or `.pt` files.

## v2 data contract

```
data/sequences_v2/
└── <label>/
    ├── <signer>__real__0000.npy           shape (60, 48, 3) float32
    ├── <signer>__real__0000.json          SampleMeta
    ├── <signer>__synthetic__0000.npy      shape (60, 48, 3) float32
    └── <signer>__synthetic__0000.json     SampleMeta (signer_id="synthetic_<rigname>")
```

Joint order (48 total): `L/R shoulder, L/R elbow, L/R wrist` then `left_hand[0..20]`, then `right_hand[0..20]`. Defined in `src/v2/schema.py`.

## Drive layout

One folder per artefact kind — nothing else in there:

```
MyDrive/SignLink/
├── data/sequences_v2/        (mirrors local data/sequences_v2/)
├── models/weights_v2/        (trained weights land here)
└── logs/v2/                  (training history .json)
```

## Piseth's workflow (leads)

```powershell
# one-time setup
rclone config                              # add remote named "ksldrive"

# daily cycle
python scripts/record_motion.py --label hello --signer piseth --count 5
python scripts/drive_sync.py push-data     # upload new samples to Drive
git add -A && git commit -m "…" && git push

# Colab — open notebooks/colab_train_v2.py in Colab, run all cells
# (it clones, pulls data, trains, pushes weights back)

python scripts/drive_sync.py pull-weights  # grab latest weights locally
python inference.py                        # (legacy) sanity-check predictions
```

## Teammate workflow (CNN track)

```powershell
git pull                                   # get latest shared data format, if any
# edit src/model.py / src/dataset.py / notebooks/train_ksl.py freely
python notebooks/train_ksl.py              # local training
# or upload to Colab as-is
git add src/model.py notebooks/train_ksl.py && git commit -m "cnn: …" && git push
```

**Do not** edit `src/v2/*`. If you want a feature in v2, open an issue or a PR.

## Colab bootstrap (2 commands)

In a new Colab notebook:

```python
from google.colab import drive; drive.mount('/content/drive')

!git clone https://github.com/Catejsj/Sign-to-Text.git /content/Sign-to-Text \
  || (cd /content/Sign-to-Text && git fetch && git reset --hard origin/main)
```

Then follow the cells in `notebooks/colab_train_v2.py`.

## Synthetic data (Pipeline C — Godot headless)

Not yet implemented. Target: given one real motion clip per sign, render ~50 variants from Godot with randomized camera angle, lighting, background, and mannequin skin, run MediaPipe on the renders, and save as `synthetic__NNNN.npy`.

See `docs/synthesis.md` (TODO) for the plan.

## Web app (FastAPI + React)

Not yet implemented. Target: single-page app that captures webcam, runs MediaPipe in-browser, sends keypoints over WebSocket, server returns the predicted sign.

See `docs/webapp.md` (TODO) for the plan.

## Research question (working)

> *Can synthetic skeleton-only data, domain-randomized from one-shot per-sign recordings, close the gap to real multi-signer data for isolated Khmer Sign Language recognition?*

Targets for proof-of-concept: **10 signs, ≥90% accuracy on a held-out signer**, using ≤2 real signers + Godot-synthetic augmentation.

## Commands cheat sheet

| I want to…                | Command                                              |
|---------------------------|------------------------------------------------------|
| record a real sign        | `python scripts/record_motion.py --label X --signer Y --count N` |
| push data to Drive        | `python scripts/drive_sync.py push-data`             |
| pull data from Drive      | `python scripts/drive_sync.py pull-data`             |
| push trained weights      | `python scripts/drive_sync.py push-weights`          |
| pull trained weights      | `python scripts/drive_sync.py pull-weights`          |
| check Drive connection    | `python scripts/drive_sync.py doctor`                |
| train locally (GPU)       | `python notebooks/colab_train_v2.py`                 |
| train on Colab            | open `notebooks/colab_train_v2.py` in Colab          |
