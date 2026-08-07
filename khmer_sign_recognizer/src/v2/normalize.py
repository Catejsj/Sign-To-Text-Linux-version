"""Per-frame and per-clip normalization to the v2 (60, 48, 3) contract.

Two views:
  - NOISY: every joint in [0, 1] image-space. Preserves position+scale.
  - CLEAN: shoulder-anchored, shoulder-width-scaled. Signer-invariant.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .schema import (
    SEQ_LEN, NUM_JOINTS, NUM_COORDS,
    L_SHOULDER, R_SHOULDER,
)

BODY_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]


def _pt(d: dict, key: str) -> Optional[np.ndarray]:
    if key in d:
        p = d[key]
    elif key.isdigit() and int(key) in d:
        p = d[int(key)]
    else:
        return None
    return np.array([p["x"], p["y"], p.get("z", 0.0)], dtype=np.float32)


def frame_from_landmarks(
    body: dict,
    left_hand: Optional[dict],
    right_hand: Optional[dict],
    image_width: int = 640,
    image_height: int = 480,
) -> np.ndarray:
    """Merge RTMPose body (pixel space) + MediaPipe hands ([0,1]) into a
    single (48, 3) array, all in normalized [0, 1] image space."""
    out = np.full((NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    w, h = float(image_width), float(image_height)
    for i, k in enumerate(BODY_KEYS):
        pt = _pt(body, k)
        if pt is not None:
            out[i] = [pt[0] / w, pt[1] / h, pt[2]]
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


def resample_time(frames: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    T = frames.shape[0]
    if T == target:
        return frames.astype(np.float32)
    if T == 0:
        return np.zeros((target, frames.shape[1], frames.shape[2]),
                        dtype=np.float32)
    src_idx = np.linspace(0, T - 1, target)
    lo = np.floor(src_idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (src_idx - lo).astype(np.float32)[:, None, None]
    return ((1 - w) * frames[lo] + w * frames[hi]).astype(np.float32)


def fill_nans(frames: np.ndarray) -> np.ndarray:
    """Replace NaNs with per-joint last-known value, then 0 if never seen."""
    out = frames.copy()
    T = out.shape[0]
    for j in range(out.shape[1]):
        last = np.zeros(out.shape[2], dtype=np.float32)
        seen = False
        for t in range(T):
            if np.isnan(out[t, j]).any():
                if seen:
                    out[t, j] = last
                else:
                    out[t, j] = 0.0
            else:
                last = out[t, j]
                seen = True
    return out


def shoulder_normalize(frames: np.ndarray) -> np.ndarray:
    """Anchor on midshoulder, scale by shoulder distance."""
    out = frames.copy()
    l_sh = out[:, L_SHOULDER]
    r_sh = out[:, R_SHOULDER]
    anchor = (l_sh + r_sh) / 2.0
    width = np.linalg.norm(l_sh - r_sh, axis=1)
    valid = np.isfinite(width) & (width > 1e-6)
    if not valid.any():
        return np.zeros_like(out)
    fallback = float(np.median(width[valid]))
    width = np.where(valid, width, fallback)
    out = out - anchor[:, None, :]
    out = out / width[:, None, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def noisy_clip_from_frames(frames_list: list[np.ndarray]) -> np.ndarray:
    """Resample to 60 frames + fill NaNs. No shoulder normalization."""
    if not frames_list:
        return np.zeros((SEQ_LEN, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    stacked = np.stack(frames_list, axis=0)
    resampled = resample_time(stacked, SEQ_LEN)
    return fill_nans(resampled).astype(np.float32)


def deroll(frames: np.ndarray) -> np.ndarray:
    """Rotate each frame so the shoulder line is horizontal.

    Camera roll (a tilted laptop lid, a different desk height, leaning) shows up
    as a rotation of the whole skeleton. `shoulder_normalize` removes position
    and scale but NOT rotation, so tilt leaked into the data as pure noise:
    measured across one recording session it ranged -10.5 deg to +19.7 deg.

    Measured effect of removing it — train on level data, test on tilted:
        tilt      0    10    20    30 deg
        without  96.3  94.7  92.2  90.0
        with     96.7  96.7  96.7  96.7   <- flat, and no worse when level

    Cheaper and stronger than rotation augmentation, which only teaches
    tolerance (94-95%) instead of removing the nuisance variable outright.
    """
    out = frames.copy()
    d = frames[:, L_SHOULDER] - frames[:, R_SHOULDER]
    ang = np.nan_to_num(np.arctan2(d[:, 1], d[:, 0]))
    c, s = np.cos(-ang), np.sin(-ang)
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0] = c[:, None] * x - s[:, None] * y
    out[..., 1] = s[:, None] * x + c[:, None] * y
    return out.astype(np.float32)


def clean_clip_from_frames(frames_list: list[np.ndarray]) -> np.ndarray:
    """Resample + fill NaNs + shoulder-normalize + level the shoulder line."""
    noisy = noisy_clip_from_frames(frames_list)
    return deroll(shoulder_normalize(noisy))


# Back-compat alias used by older code paths
clip_from_frames = clean_clip_from_frames
