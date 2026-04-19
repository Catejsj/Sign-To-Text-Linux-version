"""Record one 60-frame clip using the v2 schema.

Lightweight replacement for record_signs.py — saves (.npy, .json) pairs
under data/sequences_v2/<label>/<signer>__real__0000.npy so that
v2.dataset can pick them up without guesswork.

Usage:
    python scripts/record_motion.py --label hello --signer piseth --count 5

This imports the existing capture stack (src/capture.py) so it will only
work on the Windows machine that has RTMPose + MediaPipe installed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.capture import CaptureSystem  # existing module
from src.v2.schema import SampleMeta, Source, save_sample
from src.v2.normalize import frame_from_landmarks, clip_from_frames


def _hand_to_dict(landmarks) -> dict:
    if landmarks is None:
        return {}
    return {str(i): {"x": float(p["x"]), "y": float(p["y"]),
                     "z": float(p.get("z", 0.0))}
            for i, p in enumerate(landmarks)}


def _body_to_dict(pose) -> dict:
    """Adapt whatever CaptureSystem returns for body to a key→{x,y,z} dict."""
    if pose is None:
        return {}
    # CaptureSystem is known to emit a dict keyed by joint-name strings.
    out = {}
    for k, v in pose.items():
        if isinstance(v, dict) and "x" in v:
            out[k] = {"x": float(v["x"]), "y": float(v["y"]),
                      "z": float(v.get("z", 0.0))}
    return out


def record_one(cap: CaptureSystem, seconds: float = 2.0) -> np.ndarray:
    frames = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        data = cap.get_frame()
        if data is None:
            continue
        body = _body_to_dict(data.get("pose"))
        lh = _hand_to_dict(data.get("left_hand"))
        rh = _hand_to_dict(data.get("right_hand"))
        frames.append(frame_from_landmarks(body, lh, rh))
    return clip_from_frames(frames)


def next_variant(root: Path, label: str, signer: str) -> int:
    folder = root / label
    if not folder.exists():
        return 0
    existing = [p.stem for p in folder.glob(f"{signer}__real__*.npy")]
    nums = [int(s.rsplit("__", 1)[-1]) for s in existing if s.rsplit("__", 1)[-1].isdigit()]
    return (max(nums) + 1) if nums else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    args = ap.parse_args()

    root = Path(args.root)
    cap = CaptureSystem()
    try:
        for i in range(args.count):
            input(f"[{i+1}/{args.count}] press ENTER to record '{args.label}' "
                  f"for {args.seconds:.1f}s…")
            clip = record_one(cap, args.seconds)
            variant = next_variant(root, args.label, args.signer)
            meta = SampleMeta(label=args.label, signer_id=args.signer,
                              source=Source.REAL, variant=variant)
            path = save_sample(root, clip, meta)
            print(f"saved {path}")
    finally:
        if hasattr(cap, "close"):
            cap.close()


if __name__ == "__main__":
    main()
