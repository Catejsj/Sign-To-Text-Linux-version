# Usage

Record, train, evaluate, recognize and share. Run everything from inside
`khmer_sign_recognizer/` with the virtual environment active:

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
| Synthetic per take | also save N synthetic body-variants of each take. Landmarks only. The corpus uses **6** |
| Duration | seconds per take |

**Light the hands, not the room.** The tracker loses fast-moving hands to motion
blur long before it loses the body, so a take can look fine on screen and still
be missing a hand in many frames. More light on the hands is the fix.

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

- **Video and frames are the raw camera image, with nothing drawn on it.** The
  window shows the skeleton over your hands, but that drawing is not saved. A
  model trained on frames with the skeleton painted on would learn to read the
  drawing, and the image stream would just be the landmark stream again.
- **A take's files share one number across every modality.** Landmarks and
  video from one take are `ann__real__clean__0004.npy` and `ann__real__0004.mp4`,
  so they join on (signer, label, take). A take recorded **without** landmarks
  is named `ann__mediaonly__0000.mp4` and never uses a landmark number, because
  synthetic copies find their parent take by number.
- **Each take also gets a `.json` sidecar** with the frame count, the frame rate
  actually achieved, and for every camera frame its timestamp and whether a body
  pose was found. Landmark frame *i* is the *i*-th frame where pose is true.
- **Privacy changes.** Video and frames keep faces. They are never committed to
  git, but anyone recorded on video must have agreed to video specifically.

> ⚠️ **Pooling video across people is not supported.** `import_takes` only knows
> about `data/sequences_v2/`, and it renumbers incoming takes, so a teammate's
> video would stop matching their landmark take after import. Recording video on
> one machine works end to end.

---

## Pool everyone's recordings

Each person uploads their language folder to the shared Drive, inside a folder
with their own name:

```
SignLink/
├── TaskB/
│   ├── ann/sl_001/…        ✅ the sign FOLDERS, as they are on disk
│   └── dara/…
```

> **Copy the `sl_001` … `sl_007` folders, never the files inside them.** Every
> sign folder contains the same filenames, so pasting their contents into one
> place makes each sign overwrite the last. Tested: 12 files in, 6 survived,
> 6 of 7 signs gone. That happens in the file manager, before any script runs,
> so nothing can recover it.

**The Drive folder name becomes that person's signer identity**, so two people
who both left the tag as `me` still separate correctly.

Then import. Any nesting, a Drive `.zip`, or the same person twice all work.
Incoming takes are renumbered onto the end of what is already there and nothing
is overwritten; takes already present are detected by content and skipped.

```bash
python scripts/import_takes.py ~/Downloads/TaskB --dry-run    # look first
python scripts/import_takes.py ~/Downloads/TaskB              # then import
python scripts/import_takes.py ~/Downloads/whatever --lang khmer
```

### Fix how a sign is displayed

Edit its text in `data/sequences_v2/<lang>/labels.json`. The slug (`sl_001`) is
what models predict; the text is only what is shown. Saved models carry their
own copy of the text, so retrain them (or patch the bundle) to match.

---

## Train a model

```bash
python -m src.v2.train --lang khmer --model tcn --save
```

That writes `models/recognizers/khmer__tcn.joblib`, which Recognize mode loads.
One file per language and model, so a second model adds a file and the same pair
overwrites it.

| flag | meaning |
|---|---|
| `--lang` | **always pass it.** Left off, it trains on every language folder mixed together |
| `--model` | `tcn` (what we ship), `gru`, `lstm`, `transformer`, `bigru`, `bilstm` |
| `--holdout NAME` | validate on one signer the model never trains on |
| `--epochs` | default 100 |
| `--device` | `cuda` (default) or `cpu` |

**Only `tcn`, `gru`, `lstm` and `transformer` suit live use.** `bigru` and `bilstm`
read the clip backwards as well as forwards, so they need the sign to be
finished before they can answer.

> **The accuracy printed and stored with a model is not a benchmark.** It is
> the validation score used to pick the best epoch, on a random 15% of takes
> that usually includes the same people as training. The honest number comes
> from the evaluation below.

## Evaluate

The full comparison: fifteen models (nine classical, six neural), each scored on
a signer it has seen and on one it has never seen, plus the integrity audit
(duplicate checks, shuffled labels, a signer-identity probe).

```bash
# the run to quote: every signer capped at 30 takes per sign
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --cap-per-sign 30 --tag cap30

# the variation grid, which also tests held-out lighting, distance and position
python algo_comparison/run_task_a.py --lang khmer_var

# classical models only, about a minute
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --quick
```

| flag | meaning |
|---|---|
| `--cap-per-sign N` | at most N takes per signer per sign, so an uneven corpus reports one protocol |
| `--no-conditions` | the corpus has no lighting/distance grid |
| `--quick` | classical models only |
| `--charts-only` | redraw the charts from the existing `results.json` |
| `--tag NAME` | suffix the output folder so two runs coexist |
| `--folds` / `--epochs` | defaults 5 and 60 |

Output lands in `algo_comparison/results_<lang>_taskA[_tag]/`: `results.json`
with every number, and the charts. The full run takes about 6 minutes on a GPU.
Every seed is fixed, so the same data gives the same numbers. What they mean:
[RESULTS.md](RESULTS.md).

**Read the unseen-signer column.** The same features identify *who* is signing
at 98%, so a same-signer score partly measures recognising the person.

### Add your own algorithm

Drop a file in `custom_algos/`. Nobody edits shared code, so nobody collides.

```python
# custom_algos/ridge.py
from sklearn.linear_model import RidgeClassifier

def make_ridge():
    return RidgeClassifier(alpha=1.0, random_state=0)

ALGORITHMS = {"ridge": ("Ridge Classifier", make_ridge)}
```

The next `run_task_a.py` run includes it. Return the model, do not fit it;
features are already standardised, so do not add a scaler; pass
`random_state=0` wherever accepted. `custom_algos/bagging.py` is a working
example and one of the nine compared models.

---

## Recognize live

`./run_web.sh`, switch the panel to **Recognize**, pick a trained model, sign.

Mid-sign guesses show as muted **"Reading…"**; only the answer committed when
you stop moving shows as **"Recognized"**. The mid-sign guess comes from a
partial sign and is roughly 11 points worse, so sign, then pause fully. If live
is much worse than expected, look at the lighting on your hands before
suspecting the model.

## Share a trained model

```bash
python scripts/share_model.py khmer tcn
python scripts/share_model.py --list
```

Writes `share/khmer__tcn.zip` with the model, a README and a `check_model.py`
the receiver runs first. They need this repository, not just the file. Dropped
into `models/recognizers/`, it appears in Recognize mode without a restart.

## Look at saved takes in 3D

Plays recordings back on the mannequin, no camera needed. The quickest check
that an import actually looks like the sign.

```bash
python scripts/mannequin_local.py --playback data/sequences_v2/khmer --count 5 --fps 8
```

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
scripts/           record_session + mannequin_local (used by the web app),
                   import_takes, share_model
algo_comparison/   run_task_a.py, the 15-model evaluation
custom_algos/      drop-in algorithms
config/settings.json   camera and capture settings
fonts/             the Khmer font the camera window draws labels with
```

**Data contract.** A landmark take is two files, the same movement in two views:

```
data/sequences_v2/<lang>/<label>/<signer>__<real|synthetic>__<clean|noisy>__<NNNN>.npy
```

Each `.npy` is `(frames, 48, 3)` float32 with a `.json` sidecar (label, signer,
source, view, fps). 48 joints = 6 upper body + 21 left hand + 21 right hand.
*clean* is centred and scaled on the shoulders and rotated level; *noisy* is raw
image space. `src/v2/schema.py` is the single source of truth. A take's id is
`(signer, label, NNNN)`, and every split in the project groups on it.

**Threading contract.** The browser is controls only. Flask runs in a daemon
thread and only flips shared state; **every OpenCV and Open3D call happens on
the main thread** in `webapp/__main__.py`. Both libraries crash or hang if
driven from a worker thread.

**Three things that break the page.** The mode switch sets
`style.display = 'grid'` on `#mode-record`, so that layout must stay a grid.
About 25 classes are generated at runtime by `loadLabels()`, so a missing CSS
rule means unstyled output. Engine states are exactly `IDLE`, `COUNTDOWN`,
`RECORDING`, `RECOGNIZING`, and the badge class is built from them.
**Screenshot the page in light and dark mode after any CSS change.**

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| live recognition far worse than the reported scores | almost always hand tracking: more light on the hands, keep shoulders and hands in frame |
| the answer flickers while signing | that is the mid-sign window. Wait for "Recognized" |
| a teammate's takes do not appear after import | they uploaded files instead of sign folders. Re-run with `--dry-run` to see what was found |
| a model trained on a mix of languages | `--lang` was left off `src.v2.train` |
| a take was "discarded" | fewer than 5 frames had a body pose (without landmarks, fewer than 5 camera frames). Step back so shoulders and hands are in frame |
| the Capture switch will not turn off the last mode | intended. A take must save something |
| `run_web.sh: Permission denied` | `chmod +x run_web.sh` |
| GPU not used, everything slow | see the onnxruntime repair in [SETUP.md](SETUP.md) |
