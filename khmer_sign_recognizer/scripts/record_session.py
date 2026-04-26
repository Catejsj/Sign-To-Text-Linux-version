"""Streaming session recorder — one process, many signs, paired clean+noisy output.

The flow:
  1. Camera + MediaPipe + RTMPose run live (same as run_windows.py).
  2. Landmarks are also streamed to WSL → Godot so the mannequin mirrors you.
  3. In the terminal you type a sign label and press ENTER:
       - 1.5-second countdown overlay
       - 2-second recording
       - saves as a paired (CLEAN, NOISY) sample
       - prompts again
  4. Press ENTER on an empty line → repeat the same label (next take).
  5. Type "quit" or press 'q' in the camera window → end the session.

Every take produces TWO files:
  data/sequences_v2/<label>/<signer>__real__clean__NNNN.npy
  data/sequences_v2/<label>/<signer>__real__noisy__NNNN.npy

Usage:
  python scripts/record_session.py --signer alex
  python scripts/record_session.py --signer sam --no-stream
  python scripts/record_session.py --signer alex --duration 2.5
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.capture import LandmarkCapture            # noqa: E402
from src.send_to_wsl import WSLBridge              # noqa: E402
from src.utils import load_config, setup_logging   # noqa: E402
from src.v2.normalize import (                     # noqa: E402
    frame_from_landmarks, clean_clip_from_frames, noisy_clip_from_frames,
)
from src.v2.schema import save_pair                # noqa: E402


COUNTDOWN_S = 1.5
DEFAULT_RECORD_S = 2.0


def stdin_thread(q: Queue) -> None:
    """Push each line typed by the user onto the queue. Sentinel None on quit."""
    while True:
        try:
            line = input()
        except EOFError:
            q.put(None)
            return
        line = line.strip()
        if line.lower() == "quit":
            q.put(None)
            return
        q.put(line)


def overlay(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 40
    for text, color in lines:
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        y += 35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True,
                    help="Your short tag — used in filenames so we can do "
                         "leave-one-signer-out validation. Use the same tag "
                         "every time you record.")
    ap.add_argument("--root", default=str(ROOT / "data" / "sequences_v2"))
    ap.add_argument("--duration", type=float, default=DEFAULT_RECORD_S,
                    help="Seconds per take (default 2.0)")
    ap.add_argument("--no-stream", action="store_true",
                    help="Skip sending to WSL (no live mannequin)")
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.json"))
    args = ap.parse_args()

    root = Path(args.root)
    cfg = load_config(args.config)
    setup_logging(cfg)

    print("=" * 60)
    print(f"  SignLink session recorder — signer: {args.signer}")
    print(f"  saving to: {root}")
    print("=" * 60)
    print("Type a label + ENTER to record a take.")
    print("Empty ENTER repeats the last label.")
    print("Type 'quit' or press 'q' in the camera window to end.\n")

    capture = LandmarkCapture(cfg)
    bridge = None if args.no_stream else WSLBridge(cfg)

    if not capture.start():
        print("ERROR: camera failed to start", file=sys.stderr)
        return

    label_q: Queue = Queue()
    threading.Thread(target=stdin_thread, args=(label_q,), daemon=True).start()

    img_w = cfg["capture"]["width"]
    img_h = cfg["capture"]["height"]

    state = "IDLE"           # IDLE | COUNTDOWN | RECORDING
    current_label: str | None = None
    countdown_start = 0.0
    record_start = 0.0
    take_buffer: list[np.ndarray] = []
    take_count = 0

    try:
        while True:
            # ---- pull next frame + landmarks ----
            ret, frame = capture.read_frame()
            if not ret or frame is None:
                continue
            landmarks = capture.process_landmarks(frame)
            if landmarks and bridge is not None:
                bridge.send(landmarks)

            now = time.time()

            # ---- check terminal input ----
            if state == "IDLE":
                try:
                    line = label_q.get_nowait()
                    if line is None:
                        break
                    if line == "" and current_label:
                        pass   # repeat current_label
                    elif line:
                        current_label = line
                    if current_label:
                        state = "COUNTDOWN"
                        countdown_start = now
                except Empty:
                    pass

            # ---- state transitions ----
            if state == "COUNTDOWN" and now - countdown_start >= COUNTDOWN_S:
                state = "RECORDING"
                record_start = now
                take_buffer = []

            if state == "RECORDING":
                if landmarks:
                    body = landmarks.get("pose", {})
                    lh = landmarks.get("left_hand", {})
                    rh = landmarks.get("right_hand", {})
                    take_buffer.append(
                        frame_from_landmarks(body, lh, rh, img_w, img_h)
                    )
                if now - record_start >= args.duration:
                    if len(take_buffer) >= 5 and current_label:
                        clean = clean_clip_from_frames(take_buffer)
                        noisy = noisy_clip_from_frames(take_buffer)
                        p_clean, p_noisy = save_pair(
                            root, clean, noisy,
                            label=current_label, signer_id=args.signer,
                            fps=cfg["capture"].get("fps", 30),
                        )
                        take_count += 1
                        print(f"saved [{take_count}] {current_label}: "
                              f"{p_clean.name} + {p_noisy.name} "
                              f"({len(take_buffer)} raw frames)")
                    else:
                        print(f"WARN: only {len(take_buffer)} frames captured, "
                              f"discarded — try again, hold the sign longer or "
                              f"check your camera framing.")
                    state = "IDLE"

            # ---- overlay ----
            color = {"IDLE": (200, 200, 200),
                     "COUNTDOWN": (0, 200, 255),
                     "RECORDING": (0, 0, 255)}[state]
            lines = [(state, color),
                     (f"signer: {args.signer}", (255, 255, 255)),
                     (f"takes saved: {take_count}", (255, 255, 255))]
            if state == "COUNTDOWN":
                remaining = max(0.0, COUNTDOWN_S - (now - countdown_start))
                lines.insert(1, (f"GET READY: {remaining:.1f}s",
                                 (0, 200, 255)))
                if current_label:
                    lines.insert(2, (f"label: {current_label}", color))
            elif state == "RECORDING":
                remaining = max(0.0, args.duration - (now - record_start))
                lines.insert(1, (f"REC {remaining:.1f}s left", (0, 0, 255)))
                if current_label:
                    lines.insert(2, (f"label: {current_label}", color))
            elif current_label:
                lines.insert(1, (f"last: {current_label}", (180, 180, 180)))
            overlay(frame, lines)

            cv2.imshow("SignLink session recorder — q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.stop()
        if bridge is not None:
            bridge.close()
        cv2.destroyAllWindows()
        print(f"\nsession ended. {take_count} takes saved under {root}.")


if __name__ == "__main__":
    main()
