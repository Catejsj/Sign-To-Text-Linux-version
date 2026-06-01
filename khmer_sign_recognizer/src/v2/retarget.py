"""Parametric mannequin — retarget recorded motion onto varied body shapes.

Pure math on landmark arrays. No video, no camera, no rendering. It takes a
recorded (60, 48, 3) clip and rebuilds the skeleton with different bone
lengths — same motion (every joint angle preserved), different body.

Used by:
  - scripts/record_session.py   (auto-generates synthetic takes on save)
  - scripts/generate_synthetic.py  (batch-generates from existing recordings)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .schema import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    LEFT_HAND, RIGHT_HAND,
    Source, View, SampleMeta, save_sample, next_variant,
)
from .normalize import shoulder_normalize


def retarget(clip: np.ndarray, shoulder_s: float, upperarm_s: float,
             forearm_s: float, hand_s: float) -> np.ndarray:
    """Retarget a (60, 48, 3) clip onto a body with different bone lengths.

    Bone DIRECTIONS are untouched (so every joint angle — the sign itself —
    is preserved). Only bone LENGTHS are scaled. The skeleton is rebuilt
    outward from the shoulders so it stays connected.
    """
    f = clip
    sh_L, sh_R = f[:, L_SHOULDER], f[:, R_SHOULDER]   # each (60, 3)
    el_L, el_R = f[:, L_ELBOW],    f[:, R_ELBOW]
    wr_L, wr_R = f[:, L_WRIST],    f[:, R_WRIST]

    # Shoulders move in/out from the shoulder midpoint.
    mid = (sh_L + sh_R) / 2.0
    new_sh_L = mid + (sh_L - mid) * shoulder_s
    new_sh_R = mid + (sh_R - mid) * shoulder_s

    # Upper arm: same direction, scaled length, anchored to the new shoulder.
    new_el_L = new_sh_L + (el_L - sh_L) * upperarm_s
    new_el_R = new_sh_R + (el_R - sh_R) * upperarm_s

    # Forearm: same direction, scaled length, anchored to the new elbow.
    new_wr_L = new_el_L + (wr_L - el_L) * forearm_s
    new_wr_R = new_el_R + (wr_R - el_R) * forearm_s

    out = f.copy()
    out[:, L_SHOULDER], out[:, R_SHOULDER] = new_sh_L, new_sh_R
    out[:, L_ELBOW],    out[:, R_ELBOW]    = new_el_L, new_el_R
    out[:, L_WRIST],    out[:, R_WRIST]    = new_wr_L, new_wr_R

    # Hands: scale each hand around its wrist, then re-anchor to the new wrist.
    out[:, LEFT_HAND] = (new_wr_L[:, None, :]
                         + (f[:, LEFT_HAND] - wr_L[:, None, :]) * hand_s)
    out[:, RIGHT_HAND] = (new_wr_R[:, None, :]
                          + (f[:, RIGHT_HAND] - wr_R[:, None, :]) * hand_s)
    return out.astype(np.float32)


def generate_variants(root: Path, noisy_clip: np.ndarray, *,
                      label: str, signer_id: str, fps: int,
                      n: int, jitter: float,
                      rng: np.random.Generator,
                      language: str = "khmer") -> int:
    """Generate `n` synthetic body-variant takes from one real noisy clip.

    Each variant is saved as a clean+noisy pair tagged source=SYNTHETIC.
    The ORIGINAL signer_id and language are kept so leave-one-signer-out
    stays leak-free and synthetic data lands in the right language folder.
    Returns the number of synthetic takes saved.
    """
    lo, hi = 1.0 - jitter, 1.0 + jitter
    made = 0
    for _ in range(n):
        sh = float(rng.uniform(lo, hi))
        ua = float(rng.uniform(lo, hi))
        fa = float(rng.uniform(lo, hi))
        hd = float(rng.uniform(lo, hi))

        noisy = retarget(noisy_clip, sh, ua, fa, hd)
        clean = shoulder_normalize(noisy)

        # One variant index shared by this synthetic take's clean + noisy.
        variant = max(
            next_variant(root, label, signer_id, Source.SYNTHETIC,
                         View.CLEAN, language),
            next_variant(root, label, signer_id, Source.SYNTHETIC,
                         View.NOISY, language),
        )
        note = f"retarget sh{sh:.2f} ua{ua:.2f} fa{fa:.2f} hd{hd:.2f}"
        for clip_v, view in ((clean, View.CLEAN), (noisy, View.NOISY)):
            m = SampleMeta(label=label, signer_id=signer_id,
                           source=Source.SYNTHETIC, view=view,
                           language=language, fps=fps,
                           variant=variant, notes=note)
            save_sample(root, clip_v.astype(np.float32), m)
        made += 1
    return made
