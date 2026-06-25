"""Batch synthetic-signer generator — the parametric mannequin.

Retargets EXISTING real recordings onto randomized body proportions: each
real take of a sign becomes many synthetic takes of the same sign performed
by different-bodied signers. Pure offline math on the saved .npy files —
no video, no camera.

NOTE: scripts/record_session.py can already auto-generate synthetic takes
as you record (--synthetic N). Use THIS script to:
  - generate synthetic from takes recorded before that feature, or
  - regenerate everything from scratch with different settings (--clean).

USAGE
-----
    python scripts/generate_synthetic.py --per-take 6
    python scripts/generate_synthetic.py --per-take 10 --jitter 0.25
    python scripts/generate_synthetic.py --clean        # wipe old synthetic first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.schema import SEQ_LEN, NUM_JOINTS, NUM_COORDS, load_sample  # noqa: E402
from src.v2.retarget import generate_variants                          # noqa: E402


def discover_real_noisy(root: Path, language: str | None = None) -> list[Path]:
    """Every real take's NOISY .npy under root/<lang>/<label>/.
    We retarget the raw-proportion view. If `language` is given, only that
    language folder is processed (so you don't re-roll other folders)."""
    out = []
    if not root.exists():
        return out
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if language is not None and lang_dir.name != language:
            continue
        for label_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            out.extend(sorted(label_dir.glob("*__real__noisy__*.npy")))
    return out


def wipe_synthetic(root: Path, language: str | None = None) -> int:
    """Delete previously-generated synthetic samples. Returns count.
    If `language` is given, only that language folder is wiped."""
    n = 0
    if not root.exists():
        return 0
    for lang_dir in (p for p in root.iterdir() if p.is_dir()):
        if language is not None and lang_dir.name != language:
            continue
        for label_dir in (p for p in lang_dir.iterdir() if p.is_dir()):
            for p in label_dir.glob("*__synthetic__*"):
                p.unlink()
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    ap.add_argument("--language", default=None,
                    help="only process this language folder (e.g. autsl10). "
                         "Default: all language folders.")
    ap.add_argument("--per-take", type=int, default=6,
                    help="synthetic signers to generate per real take")
    ap.add_argument("--jitter", type=float, default=0.20,
                    help="body-proportion range: scales sampled in "
                         "[1-jitter, 1+jitter]. 0.20 = +/-20%%.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete existing synthetic samples before generating")
    args = ap.parse_args()

    root = Path(args.root)
    rng = np.random.default_rng(args.seed)

    if args.clean:
        print(f"removed {wipe_synthetic(root, args.language)} old synthetic files")

    real = discover_real_noisy(root, args.language)
    if not real:
        print(f"No real takes found under {root}.")
        print("Record some first:  python scripts/record_session.py --signer you")
        return

    print(f"Found {len(real)} real takes. Generating {args.per_take} synthetic "
          f"signers each (body jitter +/-{args.jitter*100:.0f}%)...\n")

    made = skipped = 0
    for npy in real:
        clip, meta = load_sample(npy)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            print(f"  skip (bad shape {clip.shape}): {npy.name}")
            skipped += 1
            continue
        made += generate_variants(
            root, clip, label=meta.label, signer_id=meta.signer_id,
            fps=meta.fps, n=args.per_take, jitter=args.jitter, rng=rng,
            language=meta.language,
        )

    print(f"\nDone. {made} synthetic takes generated"
          f"{f', {skipped} skipped' if skipped else ''}.")
    print("Real signer IDs are preserved on synthetic samples — "
          "leave-one-signer-out stays leak-free.")


if __name__ == "__main__":
    main()
