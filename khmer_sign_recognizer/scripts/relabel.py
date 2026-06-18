"""Merge / rename a label folder into another, fixing the JSON + labels.json.

Use when recordings landed under the wrong label name — e.g. the recorder
made `aile_1` instead of merging into `aile`, or a teammate typed the
English `family` instead of the Turkish `aile`. Files keep their signer
tag, so merging never loses track of who recorded what.

USAGE
-----
    # merge the accidental _1 duplicates back into the real classes:
    python scripts/relabel.py --lang autsl \
        --rename aile_1:aile anne_1:anne baba_1:baba cocuk_1:cocuk

    # fix a teammate who used English labels:
    python scripts/relabel.py --lang autsl \
        --rename family:aile mother:anne father:baba child:cocuk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _next_free(dst_dir: Path, stem_wo_variant: str) -> int:
    """Lowest variant index not already used by stem in dst_dir."""
    i = 0
    while (dst_dir / f"{stem_wo_variant}__{i:04d}.npy").exists():
        i += 1
    return i


def merge_label(lang_dir: Path, old: str, new: str) -> int:
    src = lang_dir / old
    dst = lang_dir / new
    if not src.exists():
        print(f"  [skip] {old}: folder not found")
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    moved = 0
    for npy in sorted(src.glob("*.npy")):
        js = npy.with_suffix(".json")
        # parts: signer__source__view__variant
        parts = npy.stem.split("__")
        if len(parts) != 4:
            print(f"  [warn] odd filename, skipping: {npy.name}")
            continue
        signer, source, view, _ = parts
        stem_wo_variant = f"{signer}__{source}__{view}"

        # avoid filename collision in the destination
        target_npy = dst / npy.name
        if target_npy.exists():
            v = _next_free(dst, stem_wo_variant)
            new_stem = f"{stem_wo_variant}__{v:04d}"
            target_npy = dst / f"{new_stem}.npy"
            target_js = dst / f"{new_stem}.json"
        else:
            target_js = dst / js.name

        # rewrite the label field inside the json
        if js.exists():
            meta = json.loads(js.read_text(encoding="utf-8"))
            meta["label"] = new
            target_js.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8")
            js.unlink()
        npy.rename(target_npy)
        moved += 1

    # remove the now-empty source folder
    leftover = list(src.iterdir())
    if not leftover:
        src.rmdir()
    else:
        print(f"  [note] {old}/ still has {len(leftover)} non-sample files")
    print(f"  merged {old} -> {new}: {moved} files")
    return moved


def rebuild_labels(lang_dir: Path) -> None:
    """Rewrite labels.json to match the folders that actually exist."""
    labels_path = lang_dir / "labels.json"
    old = {}
    if labels_path.exists():
        old = json.loads(labels_path.read_text(encoding="utf-8"))
    new = {}
    for d in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
        slug = d.name
        new[slug] = old.get(slug, slug)   # keep display if known, else slug
    labels_path.write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rebuilt {labels_path}: {list(new)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="language folder")
    ap.add_argument("--rename", nargs="+", required=True,
                    metavar="OLD:NEW",
                    help="one or more old:new label pairs")
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    args = ap.parse_args()

    lang_dir = Path(args.root) / args.lang
    if not lang_dir.exists():
        sys.exit(f"no language folder at {lang_dir}")

    total = 0
    for pair in args.rename:
        if ":" not in pair:
            sys.exit(f"bad --rename pair (need OLD:NEW): {pair!r}")
        old, new = pair.split(":", 1)
        total += merge_label(lang_dir, old, new)

    rebuild_labels(lang_dir)
    print(f"\ndone. {total} files relabeled.")


if __name__ == "__main__":
    main()
