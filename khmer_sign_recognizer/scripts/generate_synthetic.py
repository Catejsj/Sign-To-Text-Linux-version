"""Batch synthetic-signer generator — the parametric mannequin.

Retargets EXISTING real recordings onto randomized body proportions: each
real take of a sign becomes several synthetic takes of the same sign performed
by different-bodied signers. Pure offline math on the saved .npy files —
no video, no camera. Video and frames are never touched: synthetic data is
landmarks only.

The web app can already make synthetic takes while recording ("Synthetic per
take"). Use THIS script for takes recorded with that set to 0, such as the
khmer_mm multimodal set, or to regenerate a folder with different settings.

USAGE
-----
    python scripts/generate_synthetic.py --language khmer_mm
    python scripts/generate_synthetic.py --language khmer_mm --clean   # redo

Two guards, both learned the hard way:
  - `--language` is required. The old default processed every folder.
  - A folder that already has synthetic takes is refused unless `--clean` is
    given. Running twice without it doubles the copies, and the parent lookup
    (`variant // ratio`) then assigns children to the wrong take, which leaks
    test takes into training. That once produced a fake +47-point result.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.schema import SEQ_LEN, NUM_JOINTS, NUM_COORDS, load_sample  # noqa: E402
from src.v2.retarget import generate_variants                          # noqa: E402


def _count(lang_dir: Path, pattern: str) -> Counter:
    """Files matching `pattern` per (signer, label)."""
    c: Counter = Counter()
    for p in lang_dir.glob(f"*/{pattern}"):
        c[(p.name.split("__")[0], p.parent.name)] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make synthetic body-variant copies of every real take "
                    "in one language folder.")
    ap.add_argument("--language", required=True,
                    help="the language folder to process, e.g. khmer_mm")
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    ap.add_argument("--per-take", type=int, default=6,
                    help="synthetic signers per real take (the corpus uses 6)")
    ap.add_argument("--jitter", type=float, default=0.20,
                    help="body-proportion range: scales in [1-j, 1+j]. "
                         "0.20 = +/-20%%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete this folder's existing synthetic takes first")
    args = ap.parse_args()

    lang_dir = Path(args.root) / args.language
    if not lang_dir.is_dir():
        raise SystemExit(f"no language folder {lang_dir}")

    existing = sorted(lang_dir.glob("*/*__synthetic__*"))
    if existing and not args.clean:
        raise SystemExit(
            f"{args.language} already has {len(existing)} synthetic files.\n"
            "Generating again would double them and break the parent lookup.\n"
            "Re-run with --clean to replace them.")
    if args.clean:
        for p in existing:
            p.unlink()
        print(f"removed {len(existing)} old synthetic files from {args.language}")

    # Sorted, so each (signer, label) is processed in take order 0, 1, 2 ...
    # and synthetic take k*per_take .. k*per_take+per_take-1 belongs to real k.
    real = sorted(lang_dir.glob("*/*__real__noisy__*.npy"))
    if not real:
        raise SystemExit(f"no real takes in {lang_dir}")
    print(f"{len(real)} real takes in {args.language}: making {args.per_take} "
          f"synthetic each (body jitter +/-{args.jitter*100:.0f}%)")

    rng = np.random.default_rng(args.seed)
    made = skipped = 0
    for npy in real:
        clip, meta = load_sample(npy)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            print(f"  skip (bad shape {clip.shape}): {npy.name}")
            skipped += 1
            continue
        made += generate_variants(
            Path(args.root), clip, label=meta.label, signer_id=meta.signer_id,
            fps=meta.fps, n=args.per_take, jitter=args.jitter, rng=rng,
            language=meta.language)

    # Check the invariant the parent lookup depends on, rather than trust it.
    n_real = _count(lang_dir, "*__real__clean__*.npy")
    n_syn = _count(lang_dir, "*__synthetic__clean__*.npy")
    wrong = {k: (n_syn[k], n) for k, n in n_real.items()
             if n_syn[k] != n * args.per_take}
    print(f"done: {made} synthetic takes"
          f"{f', {skipped} skipped' if skipped else ''}")
    if wrong:
        print("WARNING: synthetic count is not per-take x real for:")
        for (s, lab), (got, n) in sorted(wrong.items()):
            print(f"  {s} / {lab}: {got} synthetic for {n} real")
    else:
        print(f"checked: every signer and sign has exactly {args.per_take} "
              "synthetic per real take")


if __name__ == "__main__":
    main()
