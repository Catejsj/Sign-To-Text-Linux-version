# khmer_sign_recognizer

The code root. Project overview and results are in the
[repository README](../README.md); this file is the map of what is here and
where to start.

---

## Start here

```bash
source venv/bin/activate
./run_web.sh
```

That opens the **control panel** — a browser page for the controls, with the
live camera and 3D mannequin in their own desktop window. It has two modes:

- **Record** — pick a language, add sign labels, record takes, review and
  delete them, generate synthetic variants.
- **Recognize** — load a saved model and get live predictions with confidence.

No setup yet? → [`docs/setup/`](docs/setup/). Every command the project
uses → [`docs/guides/COMMANDS.md`](docs/guides/COMMANDS.md).

---

## Layout

```
khmer_sign_recognizer/
├── run_web.sh / run_web.bat        launch the control panel
├── run_mannequin.bat               launch the 3D viewer alone (Windows)
│
├── src/
│   ├── capture.py                  camera + MediaPipe + RTMPose, device pick
│   ├── cuda_setup.py               GPU library loading
│   └── v2/
│       ├── schema.py               THE data contract — filenames, shapes
│       ├── normalize.py            clean + noisy normalisation
│       ├── retarget.py             synthetic signers (skeletal retargeting)
│       ├── augment.py              time-warp / noise / rotation
│       ├── dataset.py              sample discovery + splits
│       ├── baseline_data.py        feature extraction for classical ML
│       ├── baseline_eval.py        metrics and reports
│       ├── algorithms.py           the algorithm registry
│       ├── recognizer.py           saved-model bundles + live prediction
│       ├── model_tcn.py            SignTCN
│       ├── model_transformer.py    SignTransformer
│       └── train.py                deep-model training loop
│
├── scripts/
│   ├── record_session.py           CLI recorder (paired clean + noisy)
│   ├── mannequin_local.py          3D viewer / playback
│   ├── generate_synthetic.py       build synthetic body-variants
│   ├── import_takes.py             pool teammates' uploaded folders
│   ├── import_dataset.py           convert an external dataset (AUTSL)
│   ├── verify_pool.py              validate pooled data before training
│   ├── check_labels.py             diff labels.json against the team's
│   ├── relabel.py                  fix a wrong label safely
│   ├── export_recordings.py        collect your own takes for upload
│   ├── drive_sync.py               rclone wrapper
│   └── run_baseline.py             train ONE algorithm, report metrics
│
├── webapp/                         the control panel
│   ├── __main__.py                 supervisor loop (owns the main thread)
│   ├── app.py                      Flask routes
│   ├── engine.py                   capture/record/recognize state machine
│   ├── library.py                  language + take scanning
│   └── static/index.html           the whole UI, no build step
│
├── algo_comparison/                experiment drivers → charts + .docx
├── custom_algos/                   drop a .py here to add an algorithm
├── signlang_image_lab/             separate image-based side experiment
├── notebooks/                      Colab entrypoints
├── config/settings.json            camera params, filter tuning
└── docs/                           indexed in docs/README.md
```

**Not in git** (large, machine-specific, or regenerable): `data/`, `models/`,
`venv/`, `logs/`, `exports/`, and the charts under `algo_comparison/results*/`.
Recordings travel by Drive. The `.docx` reports and every `labels.json` **are**
committed, so the team shares one set of results and one label vocabulary.

---

## The data contract

One take is stored as **two** `.npy` files — the same movement in two views:

```
data/sequences_v2/<language>/<label>/<signer>__<source>__<view>__<nnnn>.npy
                                     └ tag    └ real     └ clean  └ take
                                                synthetic   noisy
```

Every array is `(60, 48, 3)` — 60 frames, 48 joints, xyz. 48 joints = 6 body
+ 21 left hand + 21 right hand. Each `.npy` has a `.json` sidecar carrying the
label, signer, source and view; the sidecar is authoritative and the filename
is the fallback.

- **clean** — shoulder-anchored, scale-normalised, de-rolled. Signer-invariant.
- **noisy** — raw image-space coordinates. Keeps natural variation.

One recording session produces both automatically, which gives the model free
domain randomisation. `src/v2/schema.py` is the single source of truth — if
you change the shape or naming, change it there.

---

## Threading contract

The browser is **controls only**. Flask runs in a daemon thread and does
nothing but flip shared state; **every OpenCV and Open3D call happens on the
main thread** in the supervisor loop in `webapp/__main__.py`. Both libraries
crash or hang if driven from a worker thread. Keep it that way.

---

## Component status

| | |
|---|---|
| Camera capture (MediaPipe + RTMPose) | working |
| CLI recorder + web recorder | working |
| Clean / noisy normalisation | working |
| Synthetic signer generation | working |
| Control panel — Record mode | working |
| Control panel — Recognize mode | working |
| Classical-ML comparison + reports | done — see `docs/results/` |
| SignTCN / SignTransformer | built and trainable; classical ML is the current results path |
| Godot mannequin + WSL UDP bridge | **removed** — replaced by an in-process Open3D window |

---

## Adding an algorithm

Put a file in `custom_algos/`. You never edit shared code, so nobody's
addition collides with anyone else's:

```python
from sklearn.linear_model import RidgeClassifier

ALGORITHMS = {"ridge": ("Ridge Classifier", lambda: RidgeClassifier(alpha=1.0))}
```

```bash
python scripts/run_baseline.py --list
python scripts/run_baseline.py --algo ridge --lang khmer_var --mode real
```

Details and the gotchas: [`custom_algos/README.md`](custom_algos/README.md).
