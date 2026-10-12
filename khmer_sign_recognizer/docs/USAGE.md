# Usage

Every command the project uses, in the order you would use them. Run them from
inside `khmer_sign_recognizer/` with the virtual environment active:

```bash
cd khmer_sign_recognizer
source venv/bin/activate          # Windows: venv\Scripts\activate
```

Not installed yet? See [SETUP.md](SETUP.md).

---

## Record

```bash
./run_web.sh                      # Windows: run_web.bat
```

The browser page at **http://127.0.0.1:8000** holds the controls. The camera
and 3D mannequin open in their own desktop window.

1. Pick a language folder (or create one), and type your **signer tag**.
2. Add a label, or click an existing one, and press **Record**.
3. A 1.5-second countdown, then it records for the set duration (default 3 s,
   adjustable 0.5 to 8). Press **space** in the camera window to stop early.

**Keep your signer tag identical every session.** It goes into every filename,
and it is what makes leave-one-signer-out possible: you cannot test on an unseen
person if you cannot tell people apart.

| control | what it does |
|---|---|
| **Capture** | what each take saves. See [Recording modes](#recording-modes) |
| Live view | camera, mannequin, or both, in the desktop window |
| Mannequins | how many 3D figures to show. Display only |
| Synthetic per take | also generate N synthetic body-variants of each take. Landmarks only |
| Duration | seconds per take |

### Recording modes

The **Capture** switch picks what a take saves. Any combination works except
none at all.

| | saves | where |
|---|---|---|
| **Landmarks** (default) | 48 joints per frame, clean and noisy views | `data/sequences_v2/<lang>/<label>/` |
| **Video** | the raw camera image as an `.mp4` | `data/media/<lang>/<label>/` |
| **Frames** | every raw camera frame as a `.jpg` | `data/media/<lang>/<label>/<take>_frames/` |

**Landmarks only is the default** because every existing recording, result and
privacy statement in this project assumes it.

Four things about video and frames worth knowing:

- **They are the raw camera image, with nothing drawn on it.** The window shows
  the skeleton over your hands, but that drawing is not saved. A model trained
  on frames with the skeleton painted on would learn to read the drawing, and
  the image stream would just be the landmark stream again.
- **A take's files share one number across every modality.** A take recorded
  with landmarks and video saves `ann__real__clean__0004.npy` and
  `ann__real__0004.mp4`, so they join on (signer, label, take). A take recorded
  **without** landmarks is named `ann__mediaonly__0000.mp4` instead, so it never
  takes a landmark number. That matters: synthetic copies find their parent
  take by number, and a gap in the numbering would attach them to the wrong one.
- **Each take also gets a `.json` sidecar** with the frame count, the frame rate
  actually achieved, and, for every camera frame, its timestamp and whether a
  body pose was found. Landmark frame *i* is the *i*-th frame where pose is
  true, which is how you line up pixels with joints.
- **Privacy changes.** Video and frames keep faces. They are never committed to
  git (`data/`, `*.mp4` and `*.jpg` are all ignored), but anyone recorded on
  video needs to have agreed to video specifically, not just to landmarks.

> ⚠️ **Pooling video across people is not supported yet.** `export_recordings`,
> `import_takes` and `verify_pool` only know about `data/sequences_v2/`. Worse,
> `import_takes` renumbers incoming takes onto the end of the pool, so a
> teammate's video would no longer match their landmark take number after
> import. Recording video on one machine works end to end; teaching the import
> to carry `data/media/` along with the renumbering is the next step before a
> multi-person video round.

### Check the camera first

```bash
python scripts/check_camera.py --seconds 20     # --no-preview over SSH
```

Sign normally while it runs. It reports how often the tracker finds each hand.
**Run this in the actual room before recording or demoing.** In one measured
room, hands were found in 42% of frames while signing against 92.6% while held
still, with the body at 99.7% throughout. That is motion blur on fast hands,
not darkness, and the fix is more light on the hands, not a software filter.

---

## Before sharing your recordings

Two checks that catch mistakes which never raise an error. The model just
quietly learns the wrong thing.

```bash
# does your labels.json agree with the team's?
python scripts/check_labels.py data/sequences_v2/khmer/labels.json

# duplicate signer tags, missing sidecars, wrong shapes, orphaned synthetic
python scripts/verify_pool.py --lang khmer --conditions
```

## Share and pool recordings

**Export** just your own takes:

```bash
python scripts/export_recordings.py --signer YOURTAG --lang khmer
```

That writes `exports/YOURTAG__khmer/`. Upload the whole folder to the shared
Drive, inside a folder with your name:

```
SignLink/
├── TaskB/
│   ├── ann/sl_001/…        ✅ paste the sign FOLDERS
│   └── dara/…
└── data/sequences_v2/      ← do NOT put uploads in here
```

> **Paste the `sl_001` … `sl_007` folders, never the files inside them.** Every
> sign folder contains the same filenames, so pasting their contents into one
> place makes each sign overwrite the last. Tested: 12 files in, 6 survived,
> 6 of 7 signs gone. That loss happens in the file manager, before any script
> runs, so nothing can recover it.

**Your Drive folder name becomes your signer identity**, so two people who both
left the tag as `me` still separate correctly.

**Import** everyone's uploads. Any nesting, a Drive `.zip`, or a re-upload of
the same person all work. Incoming takes are renumbered onto the end of what is
there and nothing is overwritten; identical takes are detected by content and
skipped.

```bash
python scripts/import_takes.py ~/Downloads/TaskB --dry-run    # look first
python scripts/import_takes.py ~/Downloads/TaskB              # then import
python scripts/import_takes.py ~/Downloads/whatever --lang khmer
```

Optional Drive sync through `rclone` (one-time `rclone config` needed, remote
named `ksldrive`):

```bash
python scripts/drive_sync.py push-data      # copies, never deletes
python scripts/drive_sync.py pull-data
```

---

## Synthetic data

Rebuilds each real take on differently proportioned bodies: four bone scales per
body (shoulder width, upper arm, forearm, hand), each from 0.8 to 1.2. Landmark
takes only.

```bash
python scripts/generate_synthetic.py --language khmer --per-take 6 --clean
python scripts/verify_pool.py --lang khmer          # check the ratio after
```

> **Always pass `--clean`.** Without it a second run *adds* another set instead
> of replacing it, the synthetic-to-real ratio stops being a whole number, and
> take-aware splitting attaches copies to the wrong parent. That leaks test data
> into training and silently inflates every score. It has happened here once.

Synthetic is not exported on purpose: it regenerates in seconds, so shipping it
only makes uploads seven times bigger.

`--method ik` is an alternative generator that keeps the hand at the same
body-relative location. It exists but has not been compared with the default.

## Fix a wrong label

Renames a label and rewrites the label inside every recording's sidecar. Signer
tags are kept, nothing is deleted.

```bash
python scripts/relabel.py --lang khmer --rename sl_001:sl_002
```

To fix only how a sign is *displayed* (a spelling), edit its text in
`data/sequences_v2/<lang>/labels.json`. Saved models carry their own copy of the
text, so retrain or patch them to match.

---

## Train one model

```bash
python scripts/run_baseline.py --list                       # what is available
python scripts/run_baseline.py --algo rf --lang khmer --mode real
python scripts/run_baseline.py --algo rf --lang khmer --mode both --holdout ann
python scripts/run_baseline.py --algo rf --lang khmer --mode both --save
```

| flag | meaning |
|---|---|
| `--mode real` / `both` | real takes only, or real plus synthetic. Evaluation is always real |
| `--holdout NAME` | leave one signer out: train without them, test only on them |
| `--split-random 0.2` | hold out a fraction of takes instead, when there is one signer |
| `--save` | write `models/recognizers/<lang>__<algo>.joblib` for Recognize mode |

> **A saved model has no held-out score.** `--save` retrains on every take,
> including the evaluation split, because a shipped model should use all the
> data. The accuracy printed during the run belongs to the model *before* that
> retrain. Never quote it as the shipped model's accuracy. And
> `--holdout ann --save` produces a model that **has** seen ann.

For anything you will report, run without `--save`.

## Run the full comparison

Fifteen models (nine classical, six neural), each scored on the same signer and
on an unseen signer, plus the integrity audit.

```bash
# the run to quote: every signer capped at 30 takes per sign
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --cap-per-sign 30 --tag cap30
python algo_comparison/make_task_a_report.py --lang khmer --tag cap30 --title "Khmer Sign Language Recognition"

# the variation grid, with held-out-condition tests
python algo_comparison/run_task_a.py --lang khmer_var
```

| flag | meaning |
|---|---|
| `--quick` | classical models only, about a minute |
| `--charts-only` | redraw charts from the existing `results.json` |
| `--cap-per-sign N` | at most N takes per signer per sign, so an uneven corpus reports one protocol |
| `--no-conditions` | the corpus has no lighting/distance grid |
| `--tag NAME` | suffix the output folder so two runs coexist |
| `--folds` / `--epochs` | defaults 5 and 60 |

Output lands in `algo_comparison/results_<lang>_taskA[_tag]/`: `results.json`
with every number, the charts, and the `.docx` report. The full run takes about
6 minutes on a GPU. Every seed is fixed, so the same data gives the same numbers.
What the numbers mean: [RESULTS.md](RESULTS.md).

---

## Recognize live

Train with `--save`, then `./run_web.sh`, switch the panel to **Recognize**, pick
the model, and sign.

Mid-sign guesses show as muted **"Reading…"**; only the answer committed when
you stop moving shows as **"Recognized"**. The mid-sign guess comes from a
partial sign and is roughly 11 points worse, so sign, then pause fully.

If live is much worse than expected, run `check_camera.py` before suspecting the
model.

## Look at saved takes in 3D

Plays recordings back on the mannequin, no camera needed. The quickest check that
an import actually looks like the sign.

```bash
python scripts/mannequin_local.py --playback data/sequences_v2/khmer --count 5 --fps 8
python scripts/mannequin_local.py --playback data/sequences_v2/khmer --view noisy
```

## Send a trained model to someone

```bash
python scripts/share_model.py khmer tcn
python scripts/share_model.py --list
```

Writes `share/khmer__tcn.zip` with the bundle, a README and a `check_model.py`
the receiver runs first. They need this repository, not just the file. The
accuracy stored inside a bundle is measured on its own training data and is not
a benchmark; the honest number is in [RESULTS.md](RESULTS.md).

## Add your own algorithm

Drop a file in `custom_algos/`. You never edit shared code, so nobody's addition
collides with anyone else's.

```python
# custom_algos/ridge.py
from sklearn.linear_model import RidgeClassifier

def make_ridge():
    return RidgeClassifier(alpha=1.0, random_state=0)

ALGORITHMS = {"ridge": ("Ridge Classifier", make_ridge)}
```

It then appears in `run_baseline.py --list` and joins `run_task_a.py`. Return
the model, do not fit it; features are already standardised, so do not add a
scaler; pass `random_state=0` wherever accepted. A broken file never blocks the
built-ins. `custom_algos/bagging.py` is a working example and is one of the nine
models in the comparison.

---

## How the code fits together

```
webapp/            the control panel: record, recognize, settings
  __main__.py      supervisor loop. Owns the main thread
  app.py           Flask routes
  engine.py        capture / record / recognize state machine
  media.py         video and frame capture for a take
  library.py       language and take folders; counts always come from disk
  static/index.html  the whole UI, no build step
src/               camera capture, GPU setup
src/v2/            normalisation, features, models, training, recognizer
scripts/           record, check, share, import, synthetic, train
algo_comparison/   the 15-model comparison and its report
custom_algos/      drop-in algorithms
config/settings.json   camera and capture settings
```

**Data contract.** A landmark take is two files, the same movement in two views:

```
data/sequences_v2/<lang>/<label>/<signer>__<real|synthetic>__<clean|noisy>__<NNNN>.npy
```

Each `.npy` is `(frames, 48, 3)` float32 with a `.json` sidecar (label, signer,
source, view, fps). 48 joints = 6 upper body + 21 left hand + 21 right hand.
*clean* is centred and scaled on the shoulders and rotated level; *noisy* is raw
image space. `src/v2/schema.py` is the single source of truth; change shapes or
naming there and nowhere else. A take's take-id is `(signer, label, NNNN)`, and
every split in the project groups on it.

**Threading contract.** The browser is controls only. Flask runs in a daemon
thread and only flips shared state; **every OpenCV and Open3D call happens on
the main thread** in `webapp/__main__.py`. Both libraries crash or hang if
driven from a worker thread.

**Three things that break the page.** The mode switch sets
`style.display = 'grid'` on `#mode-record`, so that layout must stay a grid.
About 25 classes are generated at runtime by `loadLabels()`, so a missing CSS
rule means unstyled output. Engine states are exactly `IDLE`, `COUNTDOWN`,
`RECORDING`, `RECOGNIZING`, and the badge class is built from them.
**Screenshot the page in light and dark mode after any CSS change**; a
dark-mode contrast bug shipped here twice and was invisible in the source.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| live recognition far worse than the reported scores | run `check_camera.py`. It is almost always hand tracking, not the model |
| the answer flickers while signing | that is the mid-sign window. Wait for "Recognized" |
| a teammate's takes do not appear after import | they uploaded files instead of sign folders. Re-run with `--dry-run` to see what was found |
| scores jumped suspiciously | check the synthetic ratio with `verify_pool.py`. A `generate_synthetic.py` run without `--clean` is the usual cause |
| a take was "discarded" | fewer than 5 frames had a body pose (or, without landmarks, fewer than 5 camera frames). Step back so shoulders and hands are in frame |
| the Capture switch will not turn off the last mode | intended. A take must save something |
| `run_web.sh: Permission denied` | `chmod +x run_web.sh` |
| GPU not used / everything slow | see the onnxruntime repair in [SETUP.md](SETUP.md) |
