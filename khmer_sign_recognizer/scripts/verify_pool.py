"""Check a pooled recording folder before training on it.

Eight people copying takes into one folder is where data quietly goes wrong.
This catches the failure modes that produce no error at training time:

  * two people using the same signer tag (files overwrite on merge)
  * one person using two spellings of their tag (`Sok` and `sok`) — their own
    data then lands in BOTH train and test, so leave-one-signer-out lies
  * a .npy with no .json sidecar (unreadable)
  * a clean view with no matching noisy view (or vice versa)
  * wrong array shape or dtype
  * a label folder that isn't in labels.json, or an entry with no folder
  * synthetic that no longer divides evenly into its real parents (orphaned by
    a deleted take — it gets silently dropped from training)
  * takes recorded into the wrong language folder (unexpected counts)

USAGE
-----
    python scripts/verify_pool.py --lang khmer
    python scripts/verify_pool.py --lang khmer_var --expect-takes 12
    python scripts/verify_pool.py --lang khmer_var --expect-takes 12 --conditions

Exit code is 1 if anything failed, so it can gate a training run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ROOT / "data" / "sequences_v2"

STEM = re.compile(
    r"^(?P<signer>.+)__(?P<source>[a-z]+)__(?P<view>[a-z]+)__(?P<variant>\d+)$")

SHAPE = (60, 48, 3)
L_SHOULDER, R_SHOULDER = 0, 1
LEFT_HAND, RIGHT_HAND = slice(6, 27), slice(27, 48)

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def note(msg: str) -> None:
    notes.append(msg)


# --------------------------------------------------------------- structure

def scan(lang_dir: Path):
    """Return {(label, signer, source, variant): {view: path}}."""
    takes: dict = defaultdict(dict)
    for label_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
        for npy in sorted(label_dir.glob("*.npy")):
            m = STEM.match(npy.stem)
            if not m:
                fail(f"unparseable filename: {npy.relative_to(lang_dir)}")
                continue
            key = (label_dir.name, m["signer"], m["source"], int(m["variant"]))
            takes[key][m["view"]] = npy
    return takes


def check_labels(lang_dir: Path) -> dict:
    f = lang_dir / "labels.json"
    if not f.exists():
        fail("no labels.json — every teammate must use the canonical copy")
        return {}
    labels = json.loads(f.read_text(encoding="utf-8"))
    on_disk = {p.name for p in lang_dir.iterdir() if p.is_dir()}
    for slug in sorted(on_disk - set(labels)):
        fail(f"folder '{slug}' has no labels.json entry")
    for slug in sorted(set(labels) - on_disk):
        note(f"labels.json lists '{slug}' but the folder is missing (no takes "
             f"recorded yet?)")
    return labels


def check_tags(takes: dict) -> None:
    signers = {k[1] for k in takes}
    by_lower: dict = defaultdict(set)
    for s in signers:
        by_lower[s.lower()].add(s)
    for low, variants in sorted(by_lower.items()):
        if len(variants) > 1:
            fail(f"signer tag '{low}' appears with different spellings: "
                 f"{sorted(variants)} — pick ONE and rename the files "
                 f"(and their .json 'signer_id' field) to match")


def check_files(takes: dict, lang_dir: Path) -> None:
    for (label, signer, source, variant), views in sorted(takes.items()):
        tag = f"{label}/{signer}__{source}__{variant:04d}"
        for view, npy in views.items():
            if not npy.with_suffix(".json").exists():
                fail(f"{tag} ({view}) has no .json sidecar — unreadable")
            try:
                arr = np.load(npy)
            except Exception as exc:                       # noqa: BLE001
                fail(f"{tag} ({view}) will not load: {exc}")
                continue
            if arr.shape != SHAPE:
                fail(f"{tag} ({view}) shape {arr.shape}, expected {SHAPE}")
            if not np.isfinite(arr).all():
                fail(f"{tag} ({view}) contains NaN or inf")
        missing = {"clean", "noisy"} - set(views)
        if missing:
            fail(f"{tag} is missing its {', '.join(sorted(missing))} view")


def check_synthetic(takes: dict) -> None:
    """Synthetic variant v belongs to real take v // ratio. If the counts do
    not divide evenly the parent link is underivable and the data is dropped."""
    counts: dict = defaultdict(lambda: {"real": 0, "synthetic": 0})
    for (label, signer, source, _), _views in takes.items():
        if source in ("real", "synthetic"):
            counts[(label, signer)][source] += 1
    for (label, signer), c in sorted(counts.items()):
        real, synth = c["real"], c["synthetic"]
        if synth and not real:
            fail(f"{label}/{signer}: {synth} synthetic with NO real parents — "
                 f"orphaned, regenerate")
        elif synth and real and synth % real:
            fail(f"{label}/{signer}: {synth} synthetic / {real} real does not "
                 f"divide evenly — orphaned by a deleted take. Regenerate "
                 f"synthetic for this language.")


def check_counts(takes: dict, expect: int | None) -> None:
    real: dict = defaultdict(int)
    for (label, signer, source, _), _v in takes.items():
        if source == "real":
            real[(signer, label)] += 1
    if not real:
        fail("no real takes found")
        return

    signers = sorted({s for s, _ in real})
    labels = sorted({l for _, l in real})
    width = max(len(s) for s in signers) + 2

    print("\nreal takes per signer x label")
    print("  " + "signer".ljust(width) + "".join(l.rjust(9) for l in labels))
    for s in signers:
        row = "".join(str(real.get((s, l), 0)).rjust(9) for l in labels)
        print("  " + s.ljust(width) + row)

    for s in signers:
        missing = [l for l in labels if not real.get((s, l))]
        if missing:
            fail(f"signer '{s}' has no takes for: {', '.join(missing)} — "
                 f"everyone must record ALL signs, or the model learns the "
                 f"person instead of the sign")
        if expect:
            off = [f"{l}={real.get((s, l), 0)}" for l in labels
                   if real.get((s, l), 0) != expect]
            if off:
                fail(f"signer '{s}' expected {expect} takes per sign, got: "
                     f"{', '.join(off)}")


# --------------------------------------------------------------- conditions

def hand_missing_rate(clip: np.ndarray) -> float:
    """Fraction of frames where a hand was not detected.

    fill_nans() writes exact 0.0 for a joint never seen, and repeats the last
    known value once a hand is lost, so both are recoverable after the fact.
    """
    bad = 0
    for sl in (LEFT_HAND, RIGHT_HAND):
        block = clip[:, sl, :]
        zero = np.all(block == 0.0, axis=(1, 2))
        held = np.zeros(len(block), dtype=bool)
        held[1:] = np.all(block[1:] == block[:-1], axis=(1, 2))
        bad += int(np.count_nonzero(zero | held))
    return bad / (2 * len(clip))


def report_conditions(takes: dict, grid: int) -> None:
    """Per-slot geometry for the fixed grid, measured from the noisy view."""
    slots: dict = defaultdict(list)
    for (label, signer, source, variant), views in takes.items():
        if source != "real" or "noisy" not in views:
            continue
        clip = np.load(views["noisy"])
        if clip.shape != SHAPE:
            continue
        mid_x = float(np.nanmean((clip[:, L_SHOULDER, 0]
                                  + clip[:, R_SHOULDER, 0]) / 2.0))
        width = float(np.nanmean(np.linalg.norm(
            clip[:, L_SHOULDER, :2] - clip[:, R_SHOULDER, :2], axis=1)))
        slots[(signer, variant % grid)].append(
            (mid_x, width, hand_missing_rate(clip)))

    if not slots:
        note("no real noisy takes to measure conditions from")
        return

    print(f"\ncondition check — variant % {grid}, measured from the noisy view")
    print("  signer          slot   centre-x   shoulder-w   hand-missing")
    for signer in sorted({s for s, _ in slots}):
        for slot in range(grid):
            rows = slots.get((signer, slot))
            if not rows:
                continue
            a = np.array(rows)
            print(f"  {signer:<15} {slot:>3}   {a[:, 0].mean():>8.3f}   "
                  f"{a[:, 1].mean():>10.3f}   {a[:, 2].mean():>11.1%}")

        xs = [np.mean([r[0] for r in slots[(signer, s)]])
              for s in range(grid) if slots.get((signer, s))]
        if xs and (max(xs) - min(xs)) < 0.05:
            note(f"'{signer}': body centre barely moves across slots "
                 f"(span {max(xs) - min(xs):.3f}) — the left/right steps may "
                 f"not have been done")

        ws = [np.mean([r[1] for r in slots[(signer, s)]])
              for s in range(grid) if slots.get((signer, s))]
        if ws and min(ws) > 0 and (max(ws) / min(ws)) > 1.5:
            note(f"'{signer}': shoulder width varies {max(ws) / min(ws):.1f}x "
                 f"across slots — distance was supposed to stay constant")


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="khmer", help="language folder to check")
    ap.add_argument("--data", default=str(SEQUENCES), help="data root")
    ap.add_argument("--expect-takes", type=int, default=None,
                    help="real takes each signer should have per sign "
                         "(12 for Task A, 30 for Task B)")
    ap.add_argument("--conditions", action="store_true",
                    help="also measure the recording conditions per slot")
    ap.add_argument("--grid", type=int, default=12,
                    help="takes per sign in the condition grid (default 12)")
    args = ap.parse_args()

    lang_dir = Path(args.data) / args.lang
    if not lang_dir.is_dir():
        sys.exit(f"no language folder at {lang_dir}")

    print(f"checking {lang_dir}")
    check_labels(lang_dir)
    takes = scan(lang_dir)
    if not takes:
        sys.exit("no takes found")

    check_tags(takes)
    check_files(takes, lang_dir)
    check_synthetic(takes)
    check_counts(takes, args.expect_takes)
    if args.conditions:
        report_conditions(takes, args.grid)

    print()
    for n in notes:
        print(f"note: {n}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these before training.")
        sys.exit(1)
    print(f"OK — {len(takes)} takes, no problems found.")


if __name__ == "__main__":
    main()
