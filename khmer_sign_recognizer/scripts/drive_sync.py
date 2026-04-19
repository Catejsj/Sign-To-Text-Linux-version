"""Thin wrapper around `rclone` for push/pull between local disk and Google Drive.

Setup (once per machine):
    1. Install rclone: https://rclone.org/install/
    2. `rclone config` → add remote named `ksldrive` of type `drive`
       (use the student Google account; choose "auto config" with browser)
    3. Inside that remote, manually create a folder called `SignLink` at the root.
       All data/weights/logs live under `SignLink/`.

Usage:
    python scripts/drive_sync.py push-data
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

def push(kind: str) -> None:
    local, remote = MAP[kind]
    local.mkdir(parents=True, exist_ok=True)
    _run([_rclone(), "sync", "--progress", str(local), remote])

def pull(kind: str) -> None:
    local, remote = MAP[kind]
    local.mkdir(parents=True, exist_ok=True)
    _run([_rclone(), "sync", "--progress", remote, str(local)])

def doctor() -> None:
    exe = _rclone()
    _run([exe, "lsd", f"{REMOTE}/"])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[
        "push-data", "pull-data", "push-weights", "pull-weights",
        "push-logs", "pull-logs", "doctor",
    ])
    args = ap.parse_args()
    if args.cmd == "doctor":
        doctor(); return
    verb, kind = args.cmd.split("-")
    (push if verb == "push" else pull)(kind)

if __name__ == "__main__":
    main()
