# ruff: noqa
"""Colab entrypoint for the SIMPLE-ALGORITHM (classical ML) experiment.

Pooled DATA, per-person ALGORITHM. Everyone's recordings are merged into one
shared dataset; each teammate opens this notebook, sets ALGO to their chosen
algorithm, and runs. The data is identical for everyone — only the algorithm
differs — so the results table is a fair comparison.

HOW TO USE IN COLAB
-------------------
File -> Open notebook -> GitHub -> Catejsj/Sign-to-Text -> notebooks/colab_baseline.py
Then run each CELL below top to bottom. Only CELL 5 changes per person (ALGO).

DRIVE LAYOUT EXPECTED
---------------------
    /MyDrive/SignLink/data/sequences_v2/autsl/   <- AUTSL base (uploaded ONCE)
    /MyDrive/SignLink/recordings/<name>/sequences_v2/...  <- each person's export

You make these by:
  - one person uploads their data/sequences_v2/autsl (real files only) once
  - everyone runs scripts/export_recordings.py and drags exports/<name> into
    /MyDrive/SignLink/recordings/<name>/

No rclone needed — Colab mounts Drive directly.
"""

# ============================================================================
# CELL 1 — mount Drive
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/SignLink'

# ============================================================================
# CELL 2 — clone or update the repo, install deps
# ============================================================================
import os
get_ipython().system('git config --global --add safe.directory /content/Sign-to-Text')
if not os.path.isdir('/content/Sign-to-Text'):
    get_ipython().system('git clone https://github.com/Catejsj/Sign-to-Text.git /content/Sign-to-Text')
else:
    get_ipython().system('cd /content/Sign-to-Text && git fetch && git reset --hard origin/main')
get_ipython().run_line_magic('cd', '/content/Sign-to-Text/khmer_sign_recognizer')
get_ipython().system('pip install -q scikit-learn numpy')

# ============================================================================
# CELL 3 — pull ALL data from Drive, route each file by its OWN metadata
# ============================================================================
# We do NOT depend on the Drive folder structure. Every .npy has a .json
# next to it that says its language, label, signer, and source. We glob
# every .npy under the Drive folder and place it where its metadata says it
# belongs. So it works no matter how anyone nested the folders on Drive
# (turkish/piseth/aile, recordings/piseth, whatever).
#
# >>> SET THIS to the folder that holds your recordings. Point it at the
#     specific folder (NOT all of MyDrive) so it doesn't grab old Khmer data.
SRC = '/content/drive/MyDrive/SignLink/data/sequences_v2/turkish'

# Force EVERY file into one language folder, no matter what each person
# typed (--lang autsl vs turkish vs ...). This normalizes the "who used
# which language name" mess so all recordings merge into one dataset. The
# json's language field is rewritten to match so synthetic generation and
# training stay consistent. Set to None to keep each file's own language.
FORCE_LANG = 'autsl'

import shutil, json, zipfile
from pathlib import Path

DATA = Path('/content/Sign-to-Text/khmer_sign_recognizer/data/sequences_v2')
DATA.mkdir(parents=True, exist_ok=True)

# Unzip the AUTSL base (real Turkish signers — they carry the held-out
# val/test split your teammates don't have). Expected next to the
# recordings on Drive as autsl_base.zip.
base_zip = Path(SRC) / 'autsl_base.zip'
if base_zip.exists():
    with zipfile.ZipFile(base_zip) as z:
        z.extractall(DATA)
    print('unzipped AUTSL base ->', DATA / 'autsl')
else:
    print('WARNING: no autsl_base.zip at', base_zip,
          '\n  -> training will have NO test signers. Upload the base first.')

copied = skipped = 0
by_signer: dict[str, int] = {}
for npy in Path(SRC).rglob('*.npy'):
    js = npy.with_suffix('.json')
    if not js.exists():
        skipped += 1
        continue
    try:
        meta = json.loads(js.read_text(encoding='utf-8'))
    except Exception:
        skipped += 1
        continue
    label = meta.get('label', npy.parent.name)
    lang = FORCE_LANG or meta.get('language', 'unknown')
    meta['language'] = lang                       # normalize the field too
    dst = DATA / lang / label
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npy, dst / npy.name)
    (dst / js.name).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    copied += 1
    sg = meta.get('signer_id', '?')
    by_signer[sg] = by_signer.get(sg, 0) + 1

print(f'routed {copied} samples into language="{FORCE_LANG}" '
      f'(skipped {skipped} with no json)')
print('by signer:', by_signer)

# ============================================================================
# CELL 4 — (re)generate synthetic on the POOLED real data
# ============================================================================
# Set LANG to the language you're training (matches the json 'language'
# field, e.g. 'autsl'). 1 synthetic per real take; bump --per-take for more.
LANG = 'autsl'
get_ipython().system(f'python scripts/generate_synthetic.py --language {LANG} --per-take 1 --clean')

# ============================================================================
# CELL 5 — >>> SET YOUR ALGORITHM HERE <<<  then run the two comparisons
# ============================================================================
# Pick ONE: lda | logreg | rf | svm | nb | tree | knn  (lda was best for us)
ALGO = 'lda'

print('=== RUN A: real only ===')
get_ipython().system(f'python scripts/run_baseline.py --algo {ALGO} --lang {LANG} --mode real')

print('\n=== RUN B: real + synthetic ===')
get_ipython().system(f'python scripts/run_baseline.py --algo {ALGO} --lang {LANG} --mode both')

# ============================================================================
# CELL 6 — copy the shared results table back to Drive
# ============================================================================
# So the whole team's runs land in one place.
res = Path('/content/Sign-to-Text/khmer_sign_recognizer/data/experiments/baseline_results.csv')
if res.exists():
    out = Path(DRIVE) / 'experiments'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(res, out / f'baseline_results_{ALGO}.csv')
    print('saved results to', out / f'baseline_results_{ALGO}.csv')
    print(res.read_text())
