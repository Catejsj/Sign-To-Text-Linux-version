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
# CELL 3 — pull the POOLED data: AUTSL base + everyone's recordings
# ============================================================================
# Copies the shared AUTSL base, then merges every teammate's recordings on
# top. Signer tags in the filenames keep files from colliding, so merging is
# a plain copy.
import shutil, glob
from pathlib import Path

DATA = Path('/content/Sign-to-Text/khmer_sign_recognizer/data/sequences_v2')
DATA.mkdir(parents=True, exist_ok=True)

# 3a. AUTSL base (real Turkish data, uploaded once to Drive)
src_autsl = Path(DRIVE) / 'data' / 'sequences_v2' / 'autsl'
if src_autsl.exists():
    shutil.copytree(src_autsl, DATA / 'autsl', dirs_exist_ok=True)
    print('copied AUTSL base')
else:
    print('WARNING: no AUTSL base at', src_autsl)

# 3b. everyone's recordings (recordings/<name>/sequences_v2/...)
rec_root = Path(DRIVE) / 'recordings'
merged = 0
if rec_root.exists():
    for person in sorted(rec_root.iterdir()):
        seq = person / 'sequences_v2'
        if not seq.is_dir():
            continue
        for npy in seq.rglob('*.npy'):
            rel = npy.relative_to(seq)
            dst = DATA / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(npy, dst)
            j = npy.with_suffix('.json')
            if j.exists():
                shutil.copy2(j, dst.with_suffix('.json'))
            merged += 1
        print('merged recordings from', person.name)
print(f'merged {merged} recorded takes total')

# ============================================================================
# CELL 4 — (re)generate synthetic on the POOLED real data
# ============================================================================
# 1 synthetic per real take (matches the local setting). Bump --per-take for
# a stronger effect.
get_ipython().system('python scripts/generate_synthetic.py --per-take 1 --clean')

# ============================================================================
# CELL 5 — >>> SET YOUR ALGORITHM HERE <<<  then run the two comparisons
# ============================================================================
# Pick ONE: knn | logreg | rf | svm | nb | tree | gboost | lda
ALGO = 'knn'

print('=== RUN A: real only ===')
get_ipython().system(f'python scripts/run_baseline.py --algo {ALGO} --lang autsl --mode real')

print('\n=== RUN B: real + synthetic ===')
get_ipython().system(f'python scripts/run_baseline.py --algo {ALGO} --lang autsl --mode both')

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
