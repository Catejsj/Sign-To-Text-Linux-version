"""WebSocket landmarks bridge — runs MediaPipe in Python (with GPU when
available) and streams the landmark arrays to the browser at
ws://localhost:8765 so the avatar in mannequin_web/index.html can use
them instead of running MediaPipe in-browser.

Why this exists:
  - Browser MediaPipe is bottlenecked by WebGL and the small Holistic
    hand model. Hands flake out at distance.
  - Python MediaPipe runs the full hand model + pose model and can use
    your CUDA GPU.
  - Same Kalidokit + VRM rendering on the browser side.

Run from the repo root (NOT from mannequin_web/):
    python scripts\\landmarks_ws.py

First-time install:
    pip install websockets
"""
from __future__ import annotations

import asyncio
import json
import sys

import cv2

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run:")
    print("  pip install mediapipe==0.10.35")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run:")
    print("  pip install websockets")
    sys.exit(1)


# ─── Tuning ──────────────────────────────────────────────────────────
PORT       = 8765
CAM_INDEX  = 0
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

POSE_OPTS = dict(
    model_complexity=1,           # 0 / 1 / 2 — 1 is the sweet spot
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
HAND_OPTS = dict(
    max_num_hands=2,
    model_complexity=1,           # 0 / 1
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# ─────────────────────────────────────────────────────────────────────


clients: set[websockets.WebSocketServerProtocol] = set()


async def register(ws):
    clients.add(ws)
    print(f"client connected ({len(clients)} total)")
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"client disconnected ({len(clients)} left)")


def landmarks_to_list(lms, with_visibility=False):
    out = []
    for lm in lms.landmark:
        item = {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
        if with_visibility:
            item["visibility"] = float(getattr(lm, "visibility", 1.0))
            item["presence"]   = float(getattr(lm, "presence",   1.0))
        out.append(item)
    return out


def process_frame(frame, pose, hands):
    """Run MediaPipe on one BGR frame, return Kalidokit-compatible dict."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    pose_res  = pose.process(rgb)
    hands_res = hands.process(rgb)

    payload = {
        "poseLandmarks":      None,
        "poseWorldLandmarks": None,
        "leftHandLandmarks":  None,
        "rightHandLandmarks": None,
        "faceLandmarks":      None,
    }

    if pose_res.pose_landmarks:
        payload["poseLandmarks"] = landmarks_to_list(
            pose_res.pose_landmarks, with_visibility=True)
    if pose_res.pose_world_landmarks:
        payload["poseWorldLandmarks"] = landmarks_to_list(
            pose_res.pose_world_landmarks, with_visibility=True)

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for i, hlm in enumerate(hands_res.multi_hand_landmarks):
            label = hands_res.multi_handedness[i].classification[0].label
            key = "leftHandLandmarks" if label == "Left" else "rightHandLandmarks"
            payload[key] = landmarks_to_list(hlm)

    return payload


async def capture_and_broadcast():
    print(f"opening camera index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        print("ERROR: cannot open camera. Is it in use by another app?")
        return

    print("MediaPipe Pose + Hands warming up...")
    pose  = mp.solutions.pose.Pose(**POSE_OPTS)
    hands = mp.solutions.hands.Hands(**HAND_OPTS)
    print("ready. capturing.")

    frame_n = 0
    last_log = 0.0
    import time
    t0 = time.time()

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue

            payload = await asyncio.to_thread(process_frame, frame, pose, hands)

            if clients:
                msg = json.dumps(payload)
                await asyncio.gather(
                    *[c.send(msg) for c in clients],
                    return_exceptions=True,
                )

            frame_n += 1
            now = time.time()
            if now - last_log >= 2.0:
                fps = frame_n / (now - t0) if now > t0 else 0
                print(f"capture fps: {fps:.1f}  ·  clients: {len(clients)}")
                last_log = now

            await asyncio.sleep(0)

    finally:
        cap.release()
        pose.close()
        hands.close()


async def main():
    print("=" * 60)
    print(f"  SignLink landmarks WebSocket server  ws://localhost:{PORT}")
    print("=" * 60)
    print("Run the browser at http://localhost:8000 and it will connect.")
    print()

    # Bind to 0.0.0.0 so it's reachable on IPv4 — "localhost" on Windows
    # often resolves to IPv6 first, which can confuse the browser side.
    async with websockets.serve(register, "0.0.0.0", PORT):
        await capture_and_broadcast()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopping.")
