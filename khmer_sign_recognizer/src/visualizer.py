"""
Python Stick Figure Visualizer — Replaces Godot for data verification.

Renders a live 2D stick figure from the same skeleton data that Godot uses.
No game engine, no UDP, no latency. Runs entirely in Python with OpenCV.

Can operate in two modes:
  1. LIVE MODE:   Overlay stick figure on the camera feed in real-time
  2. REPLAY MODE: Load a saved .npy recording and play it back as animation

This does NOT replace Godot for the final demo — Godot renders the pretty
3D Y-Bot mannequin. This is for quick data verification only.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.normalizer import BODY_JOINT_NAMES, NUM_BODY_JOINTS, NUM_HAND_LANDMARKS, TOTAL_JOINTS


# ── Skeleton connections for drawing ──

BODY_CONNECTIONS = [
    # Torso
    (1, 2),   # left_shoulder ↔ right_shoulder
    (1, 3),   # left_shoulder → left_elbow
    (3, 5),   # left_elbow → left_wrist
    (2, 4),   # right_shoulder → right_elbow
    (4, 6),   # right_elbow → right_wrist
    (1, 7),   # left_shoulder → left_hip
    (2, 8),   # right_shoulder → right_hip
    (7, 8),   # left_hip ↔ right_hip
    (0, 1),   # nose → left_shoulder (neck approx)
    (0, 2),   # nose → right_shoulder (neck approx)
]

# MediaPipe hand connections (landmark indices 0-20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


class StickFigure:
    """Draws a 2D stick figure from a (51, 3) joint array."""

    def __init__(self, canvas_size: Tuple[int, int] = (640, 480)):
        self.width, self.height = canvas_size

        # Colors
        self.body_color       = (0, 255, 0)     # green
        self.left_hand_color  = (255, 100, 50)   # blue-ish
        self.right_hand_color = (50, 100, 255)   # red-ish
        self.joint_color      = (255, 255, 255)  # white dots
        self.bg_color         = (30, 30, 30)     # dark gray

    def draw_frame(self, joints: np.ndarray,
                   canvas: Optional[np.ndarray] = None,
                   normalized: bool = True) -> np.ndarray:
        """
        Draw stick figure for one frame.

        Parameters
        ----------
        joints : (51, 3) array
        canvas : optional background image (draws on top if provided)
        normalized : if True, joints are in body-centric coords (centered at 0)
                     if False, joints are in 0-1 image coords
        """
        if canvas is None:
            canvas = np.full((self.height, self.width, 3),
                             self.bg_color, dtype=np.uint8)

        # Convert joint coords to pixel positions
        pixels = self._to_pixels(joints, normalized)

        # Draw body
        self._draw_connections(canvas, pixels, BODY_CONNECTIONS,
                               self.body_color, thickness=3)

        # Draw left hand (indices 9-29)
        hand_offset = NUM_BODY_JOINTS  # 9
        self._draw_hand(canvas, pixels, hand_offset, self.left_hand_color)

        # Draw right hand (indices 30-50)
        hand_offset = NUM_BODY_JOINTS + NUM_HAND_LANDMARKS  # 30
        self._draw_hand(canvas, pixels, hand_offset, self.right_hand_color)

        # Draw joint dots (body only — hands are too dense)
        for i in range(NUM_BODY_JOINTS):
            x, y = pixels[i]
            if x > 0 or y > 0:  # skip zero joints
                cv2.circle(canvas, (x, y), 5, self.joint_color, -1)
                cv2.circle(canvas, (x, y), 5, self.body_color, 1)

        return canvas

    def _to_pixels(self, joints: np.ndarray,
                   normalized: bool) -> List[Tuple[int, int]]:
        """Convert joint coordinates to pixel positions."""
        pixels = []
        for j in range(TOTAL_JOINTS):
            x, y = float(joints[j, 0]), float(joints[j, 1])

            if normalized:
                # Body-centric coords: center of canvas, scale to fit
                px = int(self.width  / 2 + x * self.width * 0.35)
                py = int(self.height / 3 + y * self.height * 0.5)
            else:
                # 0-1 image coords
                px = int(x * self.width)
                py = int(y * self.height)

            pixels.append((px, py))
        return pixels

    def _draw_connections(self, canvas, pixels, connections, color,
                          thickness=2):
        for i, j in connections:
            if i >= len(pixels) or j >= len(pixels):
                continue
            p1, p2 = pixels[i], pixels[j]
            # Skip if either point is at origin (undetected)
            if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
                continue
            cv2.line(canvas, p1, p2, color, thickness)

    def _draw_hand(self, canvas, pixels, offset, color):
        for i, j in HAND_CONNECTIONS:
            gi, gj = offset + i, offset + j
            if gi >= len(pixels) or gj >= len(pixels):
                continue
            p1, p2 = pixels[gi], pixels[gj]
            if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
                continue
            cv2.line(canvas, p1, p2, color, 1)

        # Fingertip dots
        for tip_idx in [4, 8, 12, 16, 20]:
            gi = offset + tip_idx
            if gi < len(pixels):
                px, py = pixels[gi]
                if px > 0 or py > 0:
                    cv2.circle(canvas, (px, py), 3, color, -1)


def replay_recording(npy_path: str, fps: int = 30):
    """
    Play back a saved .npy recording as a stick figure animation.

    Usage:
        python -m src.visualizer data/sequences/hello/hello_001_clean.npy
    """
    import time

    seq = np.load(npy_path)  # (60, 51, 3)
    print(f"Loaded: {npy_path}")
    print(f"Shape: {seq.shape}")
    print(f"Playing at {fps} FPS — press Q to quit, R to replay")

    figure  = StickFigure(canvas_size=(800, 600))
    is_clean = '_clean' in npy_path

    while True:
        for i, frame_joints in enumerate(seq):
            canvas = figure.draw_frame(frame_joints, normalized=is_clean)

            # Add info text
            label = npy_path.split('/')[-1].split('\\')[-1]
            cv2.putText(canvas, f"REPLAY: {label}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
            cv2.putText(canvas, f"Frame {i+1}/{len(seq)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 150), 1)
            mode_text = "CLEAN (normalized)" if is_clean else "RAW (pixel coords)"
            cv2.putText(canvas, mode_text,
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 200, 100), 1)

            cv2.imshow('Sign Replay', canvas)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                break  # restart loop

        # After loop ends naturally, pause on last frame
        cv2.putText(canvas, "DONE — press R to replay, Q to quit",
                    (10, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.imshow('Sign Replay', canvas)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.visualizer <path_to_recording.npy>")
        print("Example: python -m src.visualizer data/sequences/hello/hello_001_clean.npy")
        sys.exit(1)

    replay_recording(sys.argv[1])
