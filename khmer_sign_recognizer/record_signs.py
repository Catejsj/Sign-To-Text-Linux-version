"""
Recording UI — Capture sign language data for training.

Uses the existing dual pipeline (RTMPose body + MediaPipe hands)
and saves sequences as .npy files for training.

Controls:
  Type keys     → Build sign label
  ENTER         → Confirm label
  BACKSPACE     → Delete last character
  SPACE         → Start / Stop recording
  N             → New label (clear and type again)
  Q             → Quit
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, setup_logging
from capture import LandmarkCapture
from recorder import SignRecorder


# ── States ──
STATE_TYPING    = 0   # User is typing a label
STATE_READY     = 1   # Label confirmed, waiting to record
STATE_RECORDING = 2   # Recording in progress


def draw_ui(frame, state, label, recorder_status):
    """Draw recording UI overlay on the camera frame."""
    h, w = frame.shape[:2]

    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Semi-transparent footer bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if state == STATE_TYPING:
        # Show label being typed
        display_label = label + "_" if int(time.time() * 2) % 2 == 0 else label
        cv2.putText(frame, f"SIGN LABEL: {display_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Type label, press ENTER to confirm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    elif state == STATE_READY:
        cv2.putText(frame, f"SIGN: {label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to start recording | N for new label",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    elif state == STATE_RECORDING:
        # Flashing red REC indicator
        frames_done = recorder_status['frames']
        max_frames  = recorder_status['max_frames']
        progress    = frames_done / max_frames if max_frames > 0 else 0

        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (20, 25), 8, (0, 0, 255), -1)

        cv2.putText(frame, f"REC  {label}  [{frames_done}/{max_frames}]",
                    (35, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 10, 50, w - 20, 15
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * min(progress, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)

        cv2.putText(frame, "Press SPACE to stop",
                    (10, 60 + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)

    # Footer — saved count
    total = recorder_status.get('total_saved', 0)
    cv2.putText(frame, f"Saved: {total} sequences",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    cv2.putText(frame, "Q=quit  N=new label",
                (w - 220, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1)

    return frame


def main():
    # Load config
    config = load_config('config/settings.json')
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("SIGN LANGUAGE DATA RECORDER")
    logger.info("=" * 60)

    # Initialize
    capture  = LandmarkCapture(config)
    recorder = SignRecorder(config)

    if not capture.start():
        logger.error("Failed to start camera!")
        return

    state = STATE_TYPING
    current_label = ""

    logger.info("Camera ready. Type a sign label in the window.")
    logger.info("Controls: type label → ENTER → SPACE to record → SPACE to stop")

    try:
        while True:
            # Read frame with skeleton overlays
            ret, frame = capture.read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            # Process landmarks (runs the dual pipeline)
            landmarks = capture.process_landmarks(frame)

            # If recording and we have landmarks, add frame
            if state == STATE_RECORDING and landmarks:
                recorder.add_frame(
                    landmarks.get('pose', {}),
                    landmarks.get('left_hand', {}),
                    landmarks.get('right_hand', {}),
                    landmarks.get('image_width', 640),
                    landmarks.get('image_height', 480),
                )

                # Auto-stop if we hit max frames
                status = recorder.get_status()
                if status['frames'] >= status['max_frames']:
                    result = recorder.stop_recording()
                    if result:
                        logger.info(f"✅ Auto-saved (max frames reached)")
                    state = STATE_READY

            # Draw UI overlay
            status = recorder.get_status()
            frame = draw_ui(frame, state, current_label, status)

            # Resize for display
            display_frame = frame
            if 'window_scale' in config['capture']:
                scale = config['capture']['window_scale']
                if scale != 1.0:
                    new_w = int(frame.shape[1] * scale)
                    new_h = int(frame.shape[0] * scale)
                    display_frame = cv2.resize(frame, (new_w, new_h))

            cv2.namedWindow('Sign Recorder', cv2.WINDOW_NORMAL)
            cv2.imshow('Sign Recorder', display_frame)

            # ── Handle keyboard ──
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Stop recording first if active
                if state == STATE_RECORDING:
                    recorder.stop_recording()
                break

            elif state == STATE_TYPING:
                if key == 13:  # ENTER — confirm label
                    if current_label:
                        state = STATE_READY
                        logger.info(f"Label set: '{current_label}'")
                    else:
                        logger.warning("Label cannot be empty!")

                elif key == 8:  # BACKSPACE
                    current_label = current_label[:-1]

                elif 32 <= key < 127 and key != ord(' '):
                    # Printable ASCII (excluding space)
                    current_label += chr(key)

            elif state == STATE_READY:
                if key == ord(' '):
                    recorder.start_recording(current_label)
                    state = STATE_RECORDING

                elif key == ord('n'):
                    current_label = ""
                    state = STATE_TYPING
                    logger.info("Cleared label. Type a new one.")

            elif state == STATE_RECORDING:
                if key == ord(' '):
                    result = recorder.stop_recording()
                    if result:
                        logger.info(f"✅ Sequence saved!")
                    state = STATE_READY

    except KeyboardInterrupt:
        if state == STATE_RECORDING:
            recorder.stop_recording()
        logger.info("Interrupted.")

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        logger.info(f"Done. Total sequences saved: {recorder.total_saved}")


if __name__ == "__main__":
    main()
