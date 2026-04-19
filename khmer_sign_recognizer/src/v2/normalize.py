"""Per-frame and per-clip normalization to the v2 (60, 48, 3) contract."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .schema import (
    SEQ_LEN, NUM_JOINTS, NUM_COORDS,
    L_SHOULDER, R_SHOULDER, LEFT_HAND, RIGHT_HAND,
)

BODY_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]


def _pt(d: dict, key: str) -> Optional[np.ndarray]:
    if key not in d:
        return None
    p = d[key]
    return np.array([p["x"], p["y"], p.get("z", 0.0)], dtype=np.float32)


def frame_from_landmarks(
    body: dict,
    left_hand: Optional[dict],
    right_hand: Optional[dict],
) -> np.ndarray:
    """Merge RTMPose body + MediaPipe hands → unnormalized (48, 3) array.

    Missing joints become NaN; the clip-level normalizer fills them."""
    out = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    for i, k in enumerate(BODY_KEYS):
        pt = _pt(body, k)
        if pt is not None:
            out[i] = pt
    if left_hand:
        for i in range(21):
            pt = _pt(left_hand, str(i))
            if pt is not None:
                out[6 + i] = pt
    if right_hand:
        for i in range(21):
            pt = _pt(right_hand, str(i))
            if pt is not None:
                out[27 + i] = pt
    return out


def normalize_clip(frames: np.ndarray) -> np.ndarray:
    """Shoulder-centric normalization. Input/output: (T, 48, 3).

    Anchor: midpoint of shoulders per frame.
    Scale: shoulder distance per frame (falls back to median if missing).
    """
    assert frames.ndim == 3 and frames.shape[1:] == (NUM_JOINTS, NUM_COORDS)
    out = frames.copy()

    l_sh = out[:, L_SHOULDER]
    r_sh = out[:, R_SHOULDER]
    anchor = (l_sh + r_sh) / 2.0  # (T, 3)
    width = np.linalg.norm(l_sh - r_sh, axis=1)  # (T,)

    valid = np.isfinite(width) & (width > 1e-6)
    if not valid.any():
        return np.zeros_like(out)
    fallback = float(np.median(width[valid]))
    width = np.where(valid, width, fallback)

    out = out - anchor[:, None, :]
    out = out / width[:, None, None]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def resample_time(frames: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    """Linear time-resample clip to `target` frames."""
    T = frames.shape[0]
    if T == target:
        return frames.astype(np.float32)
    src_idx = np.linspace(0, T - 1, target)
    lo = np.floor(src_idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (src_idx - lo).astype(np.float32)[:, None, None]
    return ((1 - w) * frames[lo] + w * frames[hi]).astype(np.float32)


def clip_from_frames(frames_list: list[np.ndarray]) -> np.ndarray:
    """Full pipeline: list of per-frame (48,3) → normalized (60, 48, 3)."""
    if not frames_list:
        return np.zeros((SEQ_LEN, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    stacked = np.stack(frames_list, axis=0)
    resampled = resample_time(stacked, SEQ_LEN)
    return normalize_clip(resampled)
