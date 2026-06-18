"""Collect THIS machine's own recordings into one folder to drop on Drive.

Each teammate records locally, then runs this to gather only THEIR takes
(by signer tag) into `exports/<signer>/`. Upload that one folder to the
shared Drive — no rclone, just drag-and-drop.

It deliberately SKIPS the imported AUTSL base data (signer ids starting
with `autsl_`) so nobody re-uploads the Turkish dataset everyone already
has. Only real human recordings travel.

USAGE
-----
    # gather everything you recorded (any signer tag that isn't AUTSL):
    python scripts/export_recordings.py

    # or just one person's tag:
    python scripts/export_recordings.py --signer piseth

The output mirrors the data layout, so merging on the other side is a
plain copy — signer tags keep files from colliding.

    exports/piseth/
      sequences_v2/autsl/anne/piseth__real__clean__0000.npy
      sequences_v2/autsl/anne/piseth__real__clean__0000.json
      ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", default=None,
                    help="only export this signer tag (default: every "
                         "non-AUTSL signer found).")
    ap.add_argument("--data", default=str(ROOT / "data" / "sequences_v2"),
                    help="source data root")
    ap.add_argument("--out", default=str(ROOT / "exports"),
                    help="where to write the export folder")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="also copy synthetic variants (default: skip them; "
                         "they regenerate from real takes anyway).")
    args = ap.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        sys.exit(f"no data at {data_root}")

    name = args.signer or "all"
    out_root = Path(args.out) / name / "sequences_v2"

    copied = 0
    skipped_autsl = 0
    signers_seen: set[str] = set()

    for lang_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            for npy in sorted(label_dir.glob("*.npy")):
                signer = npy.stem.split("__")[0]
                source = npy.stem.split("__")[1] if "__" in npy.stem else ""

                if signer.startswith("autsl_"):
                    skipped_autsl += 1
                    continue
                if args.signer and signer != args.signer:
                    continue
                if source == "synthetic" and not args.include_synthetic:
                    continue

                signers_seen.add(signer)
                dst_dir = out_root / lang_dir.name / label_dir.name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(npy, dst_dir / npy.name)
                json_path = npy.with_suffix(".json")
                if json_path.exists():
                    shutil.copy2(json_path, dst_dir / json_path.name)
                copied += 1

    if copied == 0:
        print("No recordings found to export.")
        print("(AUTSL base data is skipped on purpose — record some signs "
              "first with scripts/record_session.py.)")
        return

    print(f"exported {copied} takes from signer(s): "
          f"{', '.join(sorted(signers_seen))}")
    print(f"skipped {skipped_autsl} AUTSL base files (not re-uploaded)")
    print(f"\nfolder ready to upload to Drive:\n  {Path(args.out) / name}")
    print("\nUpload that folder into the shared Drive, e.g.:")
    print(f"  SignLink/recordings/{name}/")


if __name__ == "__main__":
    main()
