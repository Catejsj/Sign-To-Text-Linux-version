"""Side-by-side comparison: RTMPose vs MediaPipe Pose on ONE camera feed.

Why this exists
---------------
SignLink has two body-pose options:
  - RTMPose  (used by the Python recorder, GPU via onnxruntime)
  - MediaPipe Pose (used by the browser mannequin)

To decide which to trust, they must be measured on the SAME frames. Three
separate processes can't share a webcam and would see different frames, so
this single process opens the camera once and runs both detectors per frame.

What you see
------------
  - GREEN  dots  = RTMPose joints
  - MAGENTA dots = MediaPipe Pose joints
  - ORANGE line  = the disagreement between them for that joint
  - On-screen panel: per-joint disagreement + per-detector jitter
  - On exit: summary stats in the terminal

Two numbers that matter:
  - DISAGREE  — px distance between the two detectors for a joint.
                High = at least one detector is wrong (or they define the
                joint differently).
  - JITTER    — px a joint moves frame-to-frame. Measure this while holding
                STILL: lower jitter = the more stable detector.

Usage
-----
    python scripts/compare_detectors.py
    python scripts/compare_detectors.py --camera 1
    python scripts/compare_detectors.py --log compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# DirectShow is Windows-only; V4L2 is its Linux counterpart.
if sys.platform == 'win32':
    CAMERA_BACKEND = cv2.CAP_DSHOW
elif sys.platform.startswith('linux'):
    CAMERA_BACKEND = cv2.CAP_V4L2
else:
    CAMERA_BACKEND = cv2.CAP_ANY

# joint label -> (RTMPose COCO-wholebody index, MediaPipe Pose index)
JOINTS = {
    "L_shoulder": (5, 11),
    "R_shoulder": (6, 12),
    "L_elbow":    (7, 13),
    "R_elbow":    (8, 14),
    "L_wrist":    (9, 15),
    "R_wrist":    (10, 16),
}

RTM_COLOR = (0, 255, 0)      # green   (BGR)
MP_COLOR  = (255, 0, 255)    # magenta
GAP_COLOR = (0, 165, 255)    # orange


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EMA:
    """Exponential moving average — smooths the live metric readouts."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.val: float | None = None

    def update(self, x: float) -> float:
        self.val = x if self.val is None else self.alpha * x + (1 - self.alpha) * self.val
        return self.val

    def get(self) -> float:
        return self.val if self.val is not None else 0.0


def load_camera_cfg(config_path: str) -> tuple[int, int, int]:
    try:
        cfg = json.loads(Path(config_path).read_text())
        cap = cfg.get("capture", {})
        return (cap.get("camera_id", 0),
                cap.get("width", 1280),
                cap.get("height", 720))
    except Exception:
        return 0, 1280, 720


def draw_overlay(frame, disagree, jit_rtm, jit_mp, fps) -> None:
    H = frame.shape[0]
    panel_h = 26 + 18 * len(JOINTS) + 28
    cv2.rectangle(frame, (8, H - panel_h - 8), (430, H - 8), (0, 0, 0), -1)
    y = H - panel_h + 6
    cv2.putText(frame, f"fps {fps:4.1f}   RTMPose=green  MediaPipe=magenta",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, "joint        disagree   jit-RTM   jit-MP",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
    y += 18
    for j in JOINTS:
        line = (f"{j:11s}  {disagree[j].get():7.1f}px "
                f"{jit_rtm[j].get():7.2f}  {jit_mp[j].get():7.2f}")
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y += 18


def print_summary(disagree, jit_rtm, jit_mp) -> None:
    print("\n" + "=" * 58)
    print("  SUMMARY  (averaged over the session)")
    print("=" * 58)
    print(f"  {'joint':12s} {'disagree':>10s} {'jit-RTM':>10s} {'jit-MP':>10s}")
    tot_r = tot_m = 0.0
    for j in JOINTS:
        r, m = jit_rtm[j].get(), jit_mp[j].get()
        tot_r += r
        tot_m += m
        print(f"  {j:12s} {disagree[j].get():9.1f}px {r:10.2f} {m:10.2f}")
    print("-" * 58)
    print(f"  {'mean jitter':12s} {'':>11s} {tot_r / len(JOINTS):10.2f} "
          f"{tot_m / len(JOINTS):10.2f}")
    winner = "RTMPose" if tot_r < tot_m else "MediaPipe"
    print(f"\n  Lower jitter (more stable detector): {winner}")
    print("  Note: judge jitter while holding STILL. Disagreement just")
    print("  means they place the joint differently — not which is right.")
    print("=" * 58)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.json"))
    ap.add_argument("--camera", type=int, default=None,
                    help="Override camera id from config.")
    ap.add_argument("--log", default=None,
                    help="Optional CSV path — logs per-frame metrics.")
    args = ap.parse_args()

    cam_id, W, H = load_camera_cfg(args.config)
    if args.camera is not None:
        cam_id = args.camera

    print("Loading RTMPose Wholebody (GPU)...")
    # Must happen before the first ONNX session, or onnxruntime drops to CPU
    # without saying so.
    sys.path.insert(0, str(ROOT))
    from src.cuda_setup import preload_cuda_libs
    preload_cuda_libs()

    from rtmlib import Wholebody
    wholebody = Wholebody(to_openpose=False, mode="balanced",
                          backend="onnxruntime", device="cuda")
    wholebody(np.zeros((H, W, 3), dtype=np.uint8))   # warmup
    print("RTMPose ready")

    print("Loading MediaPipe Pose...")
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(model_complexity=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    print("MediaPipe Pose ready")

    cap = cv2.VideoCapture(cam_id, CAMERA_BACKEND)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print("ERROR: camera failed to open")
        return

    disagree = {j: EMA() for j in JOINTS}
    jit_rtm  = {j: EMA() for j in JOINTS}
    jit_mp   = {j: EMA() for j in JOINTS}
    prev_rtm: dict[str, tuple[float, float]] = {}
    prev_mp:  dict[str, tuple[float, float]] = {}
    fps_ema = EMA(alpha=0.1)

    log_writer = None
    log_file = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        log_writer = csv.writer(log_file)
        cols = ["frame", "t"]
        for j in JOINTS:
            cols += [f"{j}_disagree", f"{j}_jitRTM", f"{j}_jitMP"]
        log_writer.writerow(cols)

    print("\nRunning. Stand still to read jitter; move to read disagreement.")
    print("Press q in the window to quit.\n")

    frame_i = 0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            frame_i += 1

            # ── RTMPose ──
            rtm_pts: dict[str, tuple[float, float]] = {}
            keypoints, scores = wholebody(frame)
            if keypoints is not None and len(keypoints) > 0:
                kp, sc = keypoints[0], scores[0]
                for j, (ri, _) in JOINTS.items():
                    if ri < len(kp) and sc[ri] > 0.3:
                        rtm_pts[j] = (float(kp[ri][0]), float(kp[ri][1]))

            # ── MediaPipe Pose ──
            mp_pts: dict[str, tuple[float, float]] = {}
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for j, (_, mi) in JOINTS.items():
                    if mi < len(lms) and lms[mi].visibility > 0.5:
                        mp_pts[j] = (lms[mi].x * W, lms[mi].y * H)

            # ── metrics + draw ──
            for j in JOINTS:
                r = rtm_pts.get(j)
                m = mp_pts.get(j)
                if r:
                    cv2.circle(frame, (int(r[0]), int(r[1])), 7, RTM_COLOR, -1)
                    if j in prev_rtm:
                        jit_rtm[j].update(dist(r, prev_rtm[j]))
                    prev_rtm[j] = r
                if m:
                    cv2.circle(frame, (int(m[0]), int(m[1])), 9, MP_COLOR, 2)
                    if j in prev_mp:
                        jit_mp[j].update(dist(m, prev_mp[j]))
                    prev_mp[j] = m
                if r and m:
                    cv2.line(frame, (int(r[0]), int(r[1])),
                             (int(m[0]), int(m[1])), GAP_COLOR, 2)
                    disagree[j].update(dist(r, m))

            dt = time.time() - t0
            fps_ema.update(1.0 / dt if dt > 0 else 0.0)
            draw_overlay(frame, disagree, jit_rtm, jit_mp, fps_ema.get())

            if log_writer:
                row = [frame_i, f"{time.time():.3f}"]
                for j in JOINTS:
                    row += [f"{disagree[j].get():.2f}",
                            f"{jit_rtm[j].get():.3f}",
                            f"{jit_mp[j].get():.3f}"]
                log_writer.writerow(row)

            cv2.imshow("RTMPose (green) vs MediaPipe (magenta) - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        if log_file:
            log_file.close()
            print(f"\nper-frame metrics written to {args.log}")
        print_summary(disagree, jit_rtm, jit_mp)


if __name__ == "__main__":
    main()
