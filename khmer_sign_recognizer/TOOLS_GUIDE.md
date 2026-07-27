# SignLink — Tools & Commands Guide

Reference for the data tools we added (convert a dataset, fix labels, export,
make synthetic, run the experiment). Run everything from inside
`khmer_sign_recognizer\` with the virtual environment active:

```powershell
cd Sign-to-Text\khmer_sign_recognizer
.\venv\Scripts\Activate.ps1
```

---

## 1. Convert a downloaded dataset into our format

`scripts/import_dataset.py` reads a downloaded dataset (AUTSL parquet
landmarks) and rewrites it into our `(60, 48, 3)` `.npy` schema, so it can
be trained and augmented like our own recordings.

```powershell
python scripts\import_dataset.py --format autsl `
  --src "PATH\TO\AUSTL_processed_landmark" `
  --signs 8,14,20,42 `
  --lang autsl or turkey , or name what ever you want since it not really a problem because we can mount the drive normally
```

- `--src` — folder of the downloaded dataset
- `--signs` — which class IDs to import. `8=aile, 14=anne, 20=baba, 42=cocuk`.
  The full list of all 226 IDs is in `data\external\SignList_ClassId_TR_EN.csv`.
- `--lang` — the folder name the converted data goes into
- Output: `data\sequences_v2\<lang>\<sign>\` plus a `labels.json`

Nothing is downloaded or modified in the original dataset — it only reads it
and writes our own copies.

---

## 2. Make synthetic body-variants

`scripts/generate_synthetic.py` takes the real takes and rebuilds them on
different body shapes (different arm length, shoulder width, hand size). Same
motion, different body. Pure math — no camera.

```powershell
python scripts\generate_synthetic.py --per-take 1 --clean
```

- `--per-take N` — how many synthetic bodies to create per real take
- `--clean` — delete old synthetic files before regenerating
- Each variant is saved tagged `synthetic`, keeping the original signer tag


this is at the top is more or so experiments only you have the use the old command if you were to record normally 

python scripts\record_session.py --signer piseth --lang autsl --synthetic 1 this is example 
---

## 3. Fix a wrong label (without deleting or re-recording)

`scripts/relabel.py` renames a label and fixes the label stored inside every
recording's `.json` file. Use it when a recording was saved under the wrong
name (e.g. someone typed the English `family` instead of `aile`).

```powershell
python scripts\relabel.py --lang autsl --rename family:aile mother:anne father:baba child:cocuk
```

- Each pair is `oldname:newname` (include only the ones you need)
- Files keep their signer tag, so nobody's recordings get mixed up
- Nothing is deleted — files are moved/renamed and the label inside is updated

---

## 4. Export your own recordings to send

`scripts/export_recordings.py` collects only YOUR takes into one folder ready
to upload. It skips the shared dataset base, so you don't upload gigabytes.

```powershell
python scripts\export_recordings.py --signer YOURNAME
```

- Makes `exports\YOURNAME\`
- Upload that folder to the shared Drive at `SignLink/recordings/YOURNAME/`

---

## 5. Visually check the data in 3D

`scripts/mannequin_local.py --playback` animates saved takes on the 3D
mannequin. No camera needed. Good for confirming an import or a recording
actually looks like the sign.

```powershell
python scripts\mannequin_local.py --playback data\sequences_v2\autsl --count 5
```

- `--count N` — how many random takes to play
- `--fps 8` — slow the playback down so it's easy to follow
- `--view noisy` — show the raw view instead of the normalized one

---

## 6. Run the experiment (does synthetic help?)

`scripts/run_baseline.py` trains ONE algorithm and reports accuracy,
macro-F1, per-signer F1, and a confusion matrix. Run it twice — once on real
data, once on real+synthetic — and compare. If macro-F1 goes up with
synthetic, synthetic helped.

```powershell
python scripts\run_baseline.py --algo <your-algo> --lang autsl --mode real
python scripts\run_baseline.py --algo <your-algo> --lang autsl --mode both
```

- `--algo` — pick the algorithm you're assigned. Available:
  `knn`, `logreg`, `rf`, `svm`, `nb`, `tree`, `gboost`, `lda`
- `--mode` — `real` (baseline) or `both` (real + synthetic)
- `--eval-on` — `val` (default, for development) or `test` (final number only)
- `--features` — `summary` (default) or `flat`
- The held-out test set is always real signers, so the comparison is honest.
- Results append to `data\experiments\baseline_results.csv`.

---

## Full run, start to finish

```powershell
# 1. convert the dataset (one time)
python scripts\import_dataset.py --format autsl --src "PATH" --signs 8,14,20,42 --lang autsl

# 2. record your own takes (see the setup guide for recording)

# 3. fix any wrong labels if needed
python scripts\relabel.py --lang autsl --rename family:aile

# 4. make synthetic variants
python scripts\generate_synthetic.py --per-take 1 --clean

# 5. run your assigned algorithm, two modes, and compare
python scripts\run_baseline.py --algo <your-algo> --lang autsl --mode real
python scripts\run_baseline.py --algo <your-algo> --lang autsl --mode both
```
