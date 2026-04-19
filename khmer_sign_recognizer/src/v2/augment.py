"""Augmentation ops on normalized clips. All accept (T, 48, 3) and return same."""
from __future__ import annotations

import numpy as np
from .schema import LEFT_HAND, RIGHT_HAND, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST


def scale_jitter(clip: np.ndarray, lo: float = 0.85, hi: float = 1.15,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    s = rng.uniform(lo, hi)
    return (clip * s).astype(np.float32)


def gaussian_noise(clip: np.ndarray, sigma: float = 0.01,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (clip + rng.normal(0.0, sigma, clip.shape)).astype(np.float32)


def z_rotate(clip: np.ndarray, max_deg: float = 15.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return clip @ R.T


def horizontal_flip(clip: np.ndarray) -> np.ndarray:
    """Mirror across the sagittal plane and swap left/right joint indices."""
    out = clip.copy()
    out[..., 0] = -out[..., 0]
    # swap body pairs
    for lhs, rhs in [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST)]:
        out[:, [lhs, rhs]] = out[:, [rhs, lhs]]
    # swap hand blocks
    left = out[:, LEFT_HAND].copy()
    out[:, LEFT_HAND] = out[:, RIGHT_HAND]
    out[:, RIGHT_HAND] = left
    return out.astype(np.float32)


def time_warp(clip: np.ndarray, min_frames: int = 45, max_frames: int = 75,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Pretend the clip was recorded at a different speed, then resample back."""
    from .normalize import resample_time
    rng = rng or np.random.default_rng()
    T = clip.shape[0]
    stretched_len = rng.integers(min_frames, max_frames + 1)
    stretched = resample_time(clip, int(stretched_len))
    return resample_time(stretched, T)


def augment_clip(clip: np.ndarray, flip_prob: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Default augmentation chain used during training.

    flip_prob default 0.0 — flipping changes the sign identity for KSL,
    turn it on per-label only if the sign is palm-symmetric."""
    rng = rng or np.random.default_rng()
    clip = time_warp(clip, rng=rng)
    clip = scale_jitter(clip, rng=rng)
    clip = z_rotate(clip, rng=rng)
    clip = gaussian_noise(clip, rng=rng)
    if rng.random() < flip_prob:
        clip = horizontal_flip(clip)
    return clip
