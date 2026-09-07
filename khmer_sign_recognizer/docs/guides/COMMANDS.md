# SignLink — command reference

Every command the project actually uses, grouped by what you are trying to do.

Run all of them from inside `khmer_sign_recognizer/` with the virtual
environment active:

```bash
cd khmer_sign_recognizer
source venv/bin/activate
```

> **Windows / macOS:** the commands are identical apart from the activate line
> (`venv\Scripts\Activate.ps1` on Windows). See
> [setup/SETUP_WINDOWS.md](../setup/SETUP_WINDOWS.md) and
> [setup/SETUP_MACOS.md](../setup/SETUP_MACOS.md).

---

## 1. Record

The control panel is the normal way in. It opens a browser page for the
controls; the camera and 3D mannequin stay in their own desktop window.

```bash
./run_web.sh
```

Then: pick a language → add or pick a label → **Record**.

The CLI recorder does the same job without the browser, and is useful when you
are on a remote machine or scripting a session:

```bash
python scripts/record_session.py --signer YOURTAG --lang khmer_var --duration 2
```

| flag | meaning |
|---|---|
| `--signer` | your short tag — it goes in every filename, so keep it identical every session |
| `--lang` | which language folder to record into (`khmer_var` = Task A, `khmer` = Task B) |
| `--duration` | seconds per take (default 2 → 60 frames) |
| `--synthetic N` | also generate N synthetic body-variants per take |
| `--mannequin N` | how many mannequins in the 3D window; `0` disables it |

---

## 2. Check your data before sharing it

Do these two **before** uploading. They catch the failures that produce no
error at training time — the model just quietly learns the wrong thing.

```bash
# does your labels.json agree with the rest of the team's?
python scripts/check_labels.py data/sequences_v2/khmer_var/labels.json

# duplicate signer tags, missing sidecars, wrong shapes, orphaned synthetic
python scripts/verify_pool.py --lang khmer_var --conditions
```

Then collect just your own takes into one folder to drop on Drive:

```bash
python scripts/export_recordings.py --signer YOURTAG --lang khmer
```

Output lands in `exports/YOURTAG__khmer/`. Drop that whole folder on Drive.

**One folder per person.** If two people share a machine, run it once per
signer tag — the tags keep the files apart, and separate folders mean whoever
imports can add one person at a time and hold another out for testing.

**Synthetic is deliberately not included.** It regenerates from the real takes
in one command, so shipping it triples the upload and invites a second
generation run on top of the first — the mistake that silently leaked test
data into training once already. After importing, run:

```bash
python scripts/generate_synthetic.py --language khmer --per-take 6 --clean
```

**6 per take** is the ratio both corpora use. Check it afterwards with
`verify_pool.py`; anything that is not a clean integer breaks take-aware
splitting.

---

## 3. Pool everyone's recordings

When the Drive folders come back, import them. The importer accepts whatever
people actually uploaded — any nesting, missing or broken sidecars, repeated
imports of the same person — and renumbers incoming takes onto the end of what
is already there. Nothing is overwritten.

```bash
python scripts/import_takes.py ~/Downloads/TaskA        # -> khmer_var
python scripts/import_takes.py ~/Downloads/TaskB        # -> khmer

python scripts/import_takes.py ~/Downloads/TaskA --dry-run   # look first
```

Identical clips are detected by content hash, so importing the same folder
twice does not double your data.

---

## 4. Generate synthetic signers

Rebuilds each real take on differently-proportioned bodies. Same motion,
different body — pure maths, no camera. How it works:
[reference/SYNTHETIC_RETARGETING.md](../reference/SYNTHETIC_RETARGETING.md).

```bash
python scripts/generate_synthetic.py --language khmer_var --per-take 6 --clean
```

> **Always pass `--clean`.** Without it a second run *adds* another set of
> copies instead of replacing them, and the take-aware split then groups a
> clip with the wrong parent — which leaks test data into training and
> silently inflates every score. This has bitten this project once already.

Verify the ratio afterwards:

```bash
python scripts/verify_pool.py --lang khmer_var
```

---

## 5. Fix a wrong label

Renames a label and rewrites the label stored inside every recording's
sidecar. Files keep their signer tag, so nothing gets mixed up and nothing is
deleted.

```bash
python scripts/relabel.py --lang khmer_var --rename sl_001:sl_002
```

---

## 6. Train one algorithm

```bash
python scripts/run_baseline.py --list                    # what's available
python scripts/run_baseline.py --algo rf --lang khmer_var --mode real
python scripts/run_baseline.py --algo rf --lang khmer_var --mode both
```

`real` is the baseline; `both` adds synthetic. The held-out evaluation set is
**always real**, so the comparison is honest. Results append to
`data/experiments/baseline_results.csv`.

| flag | meaning |
|---|---|
| `--holdout NAME` | leave-one-signer-out: train on everyone except this person, test on only them |
| `--split-random 0.2` | hold out a fraction of takes instead — for when there is only one signer |
| `--save` | write the model to `models/recognizers/` so the web app's Recognize mode can load it |

Adding your own algorithm needs no changes to shared code — drop a file in
`custom_algos/` and it is picked up automatically. See
[`custom_algos/README.md`](../../custom_algos/README.md).

---

## 7. Run a full comparison and produce the report

```bash
# Task A (khmer_var, the 12-cell grid)
python algo_comparison/run_task_a.py --lang khmer_var
python algo_comparison/make_task_a_report.py --lang khmer_var

# Task B (khmer, freestyle) — capped for the report, full for what ships
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --cap-per-sign 30 --tag cap30
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --tag full
python algo_comparison/make_task_a_report.py --lang khmer --tag cap30 --title "Task B — Khmer Sign Language Recognition"
```

`run_task_a.py` replaced the older `run_var_experiment.py` / `run_comparison.py`
pair in September 2026: 15 models instead of 9, both a same-signer and an
unseen-signer score for each, and the adversarial audit.

Charts and a `.docx` land in `algo_comparison/results_*/`. What the numbers
mean: [results/TASK_A_FOR_TEACHER.md](../results/TASK_A_FOR_TEACHER.md).

---

## 8. Recognize live

Train a model with `--save`, then:

```bash
./run_web.sh
```

Switch the panel to **Recognize**, pick the saved model, and sign at the
camera. Predictions and confidence stream into the page.

---

## 9. Inspect data in 3D

Plays saved takes back on the mannequin — no camera needed. The quickest way
to confirm an import actually looks like the sign.

```bash
python scripts/mannequin_local.py --playback data/sequences_v2/khmer_var --count 5 --fps 8
```

`--view noisy` shows the raw image-space view instead of the normalised one.

---

## 10. Import an external dataset

Reads AUTSL parquet landmarks and rewrites them into our `(60, 48, 3)` schema
so they can be trained and augmented like our own recordings. The original
dataset is only read, never modified.

```bash
python scripts/import_dataset.py --format autsl \
  --src PATH/TO/AUTSL_processed_landmark \
  --signs 8,14,20,42 \
  --lang autsl
```

Class IDs are listed in `data/external/SignList_ClassId_TR_EN.csv`
(`8=aile`, `14=anne`, `20=baba`, `42=cocuk`).

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `run_web.sh: Permission denied` | `chmod +x run_web.sh` — the executable bit is lost on some clones |
| camera window never opens | another process holds the camera; close the other app or replug it |
| `onnxruntime` falls back to CPU | the GPU package is missing or shadowed — see [setup/LINUX_REBUILD.md](../setup/LINUX_REBUILD.md) §3 |
| a teammate's takes don't appear after import | they uploaded the folder rather than its contents; re-run with `--dry-run` to see what was found |
| scores jumped a suspicious amount | check the synthetic ratio with `verify_pool.py` — a `generate_synthetic.py` run without `--clean` is the usual cause |
| **live recognition is much worse than the reported scores** | **run `python scripts/check_camera.py` first.** 92% of the model's signal is the hands, and poor light drops hand detection to 33–42% (PROBLEM_LOG D4). Check the camera before suspecting the model. |
| the answer flickers while you sign | that is the mid-sign sliding window, measured 11 points below the committed answer. Watch for **"Recognized"**, not "Reading…" — sign, then pause fully. |

---

## 11. Check the camera before blaming the model

```bash
python scripts/check_camera.py --seconds 20
```

Sign normally while it runs. It reports how often the tracker actually finds
each hand and grades the result against the lighting measurements in
PROBLEM_LOG D4 (warm bulb 33%, white bulb 42%, enhanced ~57% and still
insufficient).

**Why this matters more than it sounds:** feature importance puts 92% of the
signal on the hands, and only 0.2% of the recorded corpus is "hand seen then
lost" — so the flickering pattern bad light produces is one the models have
effectively never been trained on. Recognition can be far worse live than any
offline score suggests, with nothing wrong in the code.

---

## 12. The Task A report

One command computes every number, a second writes the document.

```bash
python algo_comparison/run_task_a.py
python algo_comparison/make_task_a_report.py
```

Output lands in `algo_comparison/results_khmer_var_taskA/` — `Task_A_Report.docx`,
six charts, and `results.json` with every figure the report quotes.

| flag | |
|---|---|
| `--lang khmer` | run it on the other corpus instead |
| `--quick` | classical models only, about a minute |
| `--charts-only` | redraw the charts from the existing `results.json`, seconds |
| `--cap-per-sign N` | use at most N real takes per signer per sign, so an uneven corpus reports a uniform protocol without deleting anything |
| `--no-conditions` | the corpus has no lighting/distance grid (Task B is freestyle; only Task A was gridded) |
| `--tag NAME` | suffix the output folder so two runs on one corpus coexist |

**Task B (`khmer`) takes two runs** — one for the report, one for what ships:

```bash
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --cap-per-sign 30 --tag cap30
python algo_comparison/run_task_a.py --lang khmer --grid 30 --no-conditions --tag full
python algo_comparison/make_task_a_report.py --lang khmer --tag cap30 --title "Task B — Khmer Sign Language Recognition"
```

The capped run is the one to quote: several people recorded more than the 30
per sign Task B asked for. The extra takes are worth about +0.9 macro-F1,
which is inside the fold noise, so they belong in the shipped model and not in
a protocol claim.

The run takes roughly 6 minutes with a GPU: 9 classical algorithms across 3
feature sets, 6 neural networks, and the audit. Every seed is fixed, so two
runs on the same data give the same numbers.
