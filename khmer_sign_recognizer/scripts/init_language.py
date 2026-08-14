"""Create a language folder that reuses another language's labels.

Needed because the label text is Khmer and most of us have no Khmer keyboard.
Instead of retyping ជម្រាបសួរ into a new folder, clone `labels.json` from a
language that already has it, so the slugs and the text match exactly.

USAGE
-----
    # make khmer_var with the same 7 signs as khmer:
    python scripts/init_language.py --from khmer --to khmer_var

    # see what it would do without writing anything:
    python scripts/init_language.py --from khmer --to khmer_var --dry-run

If the target already exists, its labels.json is compared against the source
and any difference is reported — a mismatch there is the one error that
silently puts takes into the wrong class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ROOT / "data" / "sequences_v2"


def read_labels(lang_dir: Path) -> dict:
    f = lang_dir / "labels.json"
    if not f.exists():
        sys.exit(f"no labels.json in {lang_dir}")
    return json.loads(f.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default="khmer",
                    help="language to copy labels from (default: khmer)")
    ap.add_argument("--to", dest="dst", required=True,
                    help="language folder to create")
    ap.add_argument("--data", default=str(SEQUENCES), help="data root")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    args = ap.parse_args()

    root = Path(args.data)
    src_dir, dst_dir = root / args.src, root / args.dst

    if not src_dir.is_dir():
        sys.exit(f"source language '{args.src}' not found at {src_dir}")

    labels = read_labels(src_dir)
    if not labels:
        sys.exit(f"'{args.src}' has an empty labels.json — nothing to clone")

    print(f"source: {args.src}  ({len(labels)} labels)")
    for slug, text in labels.items():
        print(f"  {slug}  {text}")

    if dst_dir.exists():
        existing = read_labels(dst_dir)
        if existing == labels:
            print(f"\n'{args.dst}' already exists and its labels match. "
                  f"Nothing to do.")
        else:
            print(f"\nMISMATCH — '{args.dst}' exists with different labels.")
            for slug in sorted(set(labels) | set(existing)):
                a, b = labels.get(slug), existing.get(slug)
                if a != b:
                    print(f"  {slug}: {args.src}={a!r}  {args.dst}={b!r}")
            print("\nFix this before recording — takes would land in the "
                  "wrong class.")
            sys.exit(1)
        return

    if args.dry_run:
        print(f"\n[dry-run] would create {dst_dir} with "
              f"{len(labels)} label folders")
        return

    dst_dir.mkdir(parents=True)
    (dst_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for slug in labels:
        (dst_dir / slug).mkdir(exist_ok=True)

    print(f"\ncreated {dst_dir}")
    print(f"  labels.json + {len(labels)} empty label folders")
    print(f"\nPick '{args.dst}' in the web panel's language dropdown before "
          f"recording.")


if __name__ == "__main__":
    main()
