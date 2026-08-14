"""Thin wrapper around `rclone` for push/pull between local disk and Google Drive.

Setup (once per machine):
    1. Install rclone: https://rclone.org/install/
    2. `rclone config` → add remote named `ksldrive` of type `drive`
       (use the student Google account; choose "auto config" with browser)
    3. Inside that remote, manually create a folder called `SignLink` at the root.
       All data/weights/logs live under `SignLink/`.

Usage:
    python scripts/drive_sync.py push-data      # copies; never deletes
    python scripts/drive_sync.py push-data --mirror   # deletes too (asks first)
    python scripts/drive_sync.py pull-weights
    python scripts/drive_sync.py pull-data   # for teammates
    python scripts/drive_sync.py push-weights
    python scripts/drive_sync.py doctor
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REMOTE = "ksldrive:SignLink"
ROOT = Path(__file__).resolve().parents[1]

MAP = {
    "data":    (ROOT / "data" / "sequences_v2",   f"{REMOTE}/data/sequences_v2"),
    "weights": (ROOT / "models" / "weights_v2",   f"{REMOTE}/models/weights_v2"),
    "logs":    (ROOT / "logs" / "v2",             f"{REMOTE}/logs/v2"),
}

def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        sys.exit("rclone not found on PATH. Install from https://rclone.org/install/")
    return exe

def _run(args: list[str]) -> None:
    print(">", " ".join(args))
    subprocess.run(args, check=True)

def _transfer(src: str, dst: str, mirror: bool, dry_run: bool) -> None:
    """Copy by default; only mirror (which DELETES) when asked, and only after
    showing what would go and getting a yes.

    `rclone sync` makes the destination identical to the source, so it removes
    anything the source does not have. Pushing from a machine that is missing a
    folder therefore deletes it for everyone, and pulling deletes local takes
    that were never pushed. Neither is ever what you want when eight people are
    pooling recordings, so it is opt-in.
    """
    exe = _rclone()
    if not mirror:
        _run([exe, "copy", "--progress", src, dst])
        return

    print("Checking what mirroring would DELETE at the destination...\n")
    preview = subprocess.run(
        [exe, "sync", "--dry-run", src, dst], capture_output=True, text=True)
    deletions = [l for l in (preview.stderr + preview.stdout).splitlines()
                 if "Skipped delete" in l or "Deleted" in l]
    if deletions:
        print(f"{len(deletions)} file(s) would be DELETED:")
        for line in deletions[:20]:
            print(f"  {line.strip()}")
        if len(deletions) > 20:
            print(f"  ... and {len(deletions) - 20} more")
    else:
        print("nothing would be deleted.")

    if dry_run:
        print("\n--dry-run: stopping here.")
        return
    if deletions and input("\nType 'delete' to go ahead: ").strip() != "delete":
        sys.exit("cancelled — nothing changed.")
    _run([exe, "sync", "--progress", src, dst])


def push(kind: str, mirror: bool = False, dry_run: bool = False) -> None:
    local, remote = MAP[kind]
    local.mkdir(parents=True, exist_ok=True)
    _transfer(str(local), remote, mirror, dry_run)

def pull(kind: str, mirror: bool = False, dry_run: bool = False) -> None:
    local, remote = MAP[kind]
    local.mkdir(parents=True, exist_ok=True)
    _transfer(remote, str(local), mirror, dry_run)

def doctor() -> None:
    exe = _rclone()
    _run([exe, "lsd", f"{REMOTE}/"])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[
        "push-data", "pull-data", "push-weights", "pull-weights",
        "push-logs", "pull-logs", "doctor",
    ])
    ap.add_argument("--mirror", action="store_true",
                    help="make the destination EXACTLY match the source, "
                         "deleting anything else there. Shows what would go "
                         "and asks first. Default is copy, which never "
                         "deletes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --mirror, list the deletions and stop")
    args = ap.parse_args()
    if args.cmd == "doctor":
        doctor(); return
    verb, kind = args.cmd.split("-")
    (push if verb == "push" else pull)(kind, args.mirror, args.dry_run)

if __name__ == "__main__":
    main()
