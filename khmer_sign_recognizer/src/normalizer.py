"""
Local Normalizer — Standalone normalization for recording & inference.
Ports the WSL mapper normalization logic so we can run entirely on Windows
without needing WSL during recording or inference.

Produces two versions of each frame:
  clean: shoulder-anchored, shoulder-width-scaled, One Euro smoothed
  raw:   pixel-to-01-normalized only (no body-relative normalization)
"""

import numpy as np
from typing import Dict, Tuple, Optional

# ── Joint layout ──
# Every frame is stored as a (51, 3) array with this fixed order:

BODY_JOINT_NAMES = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow',    'right_elbow',
    'left_wrist',    'right_wrist',
    'left_hip',      'right_hip',
]  # 9 joints → indices 0–8

NUM_BODY_JOINTS   = len(BODY_JOINT_NAMES)   # 9
NUM_HAND_LANDMARKS = 21                      # MediaPipe 0–20
TOTAL_JOINTS       = NUM_BODY_JOINTS + 2 * NUM_HAND_LANDMARKS  # 51


class OneEuroFilter:
    """Lightweight One Euro Filter for smoothing."""

    def __init__(self, min_cutoff=1.0, beta=0.05):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.x_prev     = None
        self.dx_prev    = 0.0
        self.t_prev     = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx      = (x - self.x_prev) / dt
        edx     = self._alpha(self.min_cutoff, dt)
        dx_hat  = edx * dx + (1 - edx) * self.dx_prev
        cutoff  = self.min_cutoff + self.beta * abs(dx_hat)
        ex      = self._alpha(cutoff, dt)
        x_hat   = ex * x + (1 - ex) * self.x_prev
        self.x_prev  = x_hat
        self.dx_prev = dx_hat
        self.t_prev  = t
        return x_hat

    def _alpha(self, cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)


class FrameNormalizer:
    """
    Converts one frame of skeleton data into a (51, 3) numpy array.

    Two output modes:
      clean — shoulder-anchored, shoulder-width-scaled, smoothed
      raw   — coordinates in [0,1] image space, no body-relative transform
    """

    def __init__(self, smoothing: bool = True,
                 min_cutoff: float = 1.0, beta: float = 0.05):
        self.smoothing = smoothing
        self.filters: Dict[str, OneEuroFilter] = {}

        if smoothing:
            for j in range(TOTAL_JOINTS):
                for axis in ('x', 'y', 'z'):
                    self.filters[f'{j}_{axis}'] = OneEuroFilter(min_cutoff, beta)

    # ─────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────

    def normalize(self,
                  pose: Dict,
                  left_hand: Dict,
                  right_hand: Dict,
                  img_w: int = 640,
                  img_h: int = 480,
                  timestamp: float = 0.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        clean : ndarray (51, 3)
            Body-centric normalized + smoothed.
        raw   : ndarray (51, 3)
            Pixel-to-[0,1] normalized only (no shoulder transform).
        """
        raw_frame = self._build_raw_frame(pose, left_hand, right_hand,
                                          img_w, img_h)
        clean_frame = self._build_clean_frame(raw_frame, timestamp)
        return clean_frame, raw_frame

    # ─────────────────────────────────────────
    #  Raw frame — just 0-1 pixel normalization
    # ─────────────────────────────────────────

    def _build_raw_frame(self, pose, left_hand, right_hand,
                         img_w, img_h) -> np.ndarray:
        frame = np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)

        # Body joints (RTMPose gives pixel coords)
        for i, name in enumerate(BODY_JOINT_NAMES):
            if name in pose:
                lm = pose[name]
                frame[i] = [
                    float(lm['x']) / img_w,
                    float(lm['y']) / img_h,
                    float(lm.get('z', 0.0))
                ]

        # Left hand (MediaPipe gives 0-1 coords already)
        offset = NUM_BODY_JOINTS  # 9
        for idx in range(NUM_HAND_LANDMARKS):
            key = str(idx)
            if key in left_hand:
                lm = left_hand[key]
                frame[offset + idx] = [
                    float(lm['x']),
                    float(lm['y']),
                    float(lm.get('z', 0.0))
                ]

        # Right hand
        offset = NUM_BODY_JOINTS + NUM_HAND_LANDMARKS  # 30
        for idx in range(NUM_HAND_LANDMARKS):
            key = str(idx)
            if key in right_hand:
                lm = right_hand[key]
                frame[offset + idx] = [
                    float(lm['x']),
                    float(lm['y']),
                    float(lm.get('z', 0.0))
                ]

        return frame

    # ─────────────────────────────────────────
    #  Clean frame — body-centric + smoothing
    # ─────────────────────────────────────────

    def _build_clean_frame(self, raw: np.ndarray,
                           timestamp: float) -> np.ndarray:
        frame = raw.copy()

        # Anchor = mid-shoulder (indices 1, 2)
        l_shoulder = raw[1]
        r_shoulder = raw[2]
        anchor = (l_shoulder + r_shoulder) / 2.0
        scale  = np.linalg.norm(l_shoulder - r_shoulder)

        if scale < 1e-6:
            # Shoulders not detected or overlapping — return raw
            return frame

        # Normalize all joints relative to mid-shoulder
        for j in range(TOTAL_JOINTS):
            frame[j] = (frame[j] - anchor) / scale

        # Apply smoothing
        if self.smoothing:
            for j in range(TOTAL_JOINTS):
                frame[j, 0] = self.filters[f'{j}_x'](frame[j, 0], timestamp)
                frame[j, 1] = self.filters[f'{j}_y'](frame[j, 1], timestamp)
                frame[j, 2] = self.filters[f'{j}_z'](frame[j, 2], timestamp)

        return frame

    def reset(self):
        """Reset smoothing filters (call between recordings)."""
        for f in self.filters.values():
            f.x_prev  = None
            f.dx_prev = 0.0
            f.t_prev  = None
