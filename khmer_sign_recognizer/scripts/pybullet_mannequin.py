"""PyBullet live mannequin — replaces the Godot mannequin.

Listens on UDP port 8888 (same port WSL forwards to), loads a humanoid
in a PyBullet window, and uses inverse kinematics to make the arms
reach where your wrists are in real time.

Run this INSTEAD of Godot:
    python scripts\\pybullet_mannequin.py

The WSL side does not change — start_wsl.sh still forwards to port 8888.
This script just receives and visualizes those packets.

Tuning: see the constants at the top. Change one, rerun the script.

Coordinate notes:
    Mapper output (from src/mapper.py):
        pos.x  positive = your-left direction
        pos.y  positive = down (image Y axis)
        pos.z  ~= 0     (we don't have reliable depth)
        all values are in shoulder-width units
    PyBullet humanoid coordinate frame:
        +X = humanoid's right    (so signer's left maps to +X by default,
                                  which is "mirror image" — flip MIRROR_X
                                  to false if you want same-side)
        +Y = humanoid's forward
        +Z = up
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("ERROR: pybullet not installed. Run:")
    print("  .\\venv\\Scripts\\Activate.ps1")
    print("  pip install pybullet")
    sys.exit(1)


# ─── TUNING KNOBS — change and re-run ────────────────────────────────
PORT             = 8888       # must match start_wsl.sh forwarding
SHOULDER_WIDTH_M = 0.40       # how big a "shoulder width" is in PyBullet metres
TORSO_HEIGHT_Z   = 1.40       # height of shoulder line above ground (metres)
MIRROR_X         = True       # flip left/right so it acts as a mirror
SHOW_TARGETS     = True       # draw red/blue spheres where wrists should be
PRINT_JOINTS     = True       # print joint names on startup (useful first time)
IK_ITERATIONS    = 100        # higher = more accurate, slower
# ─────────────────────────────────────────────────────────────────────


def find_joint_index(robot_id: int, name_substr: str) -> int | None:
    """Find a joint whose name contains the given substring (case-insensitive)."""
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8").lower()
        if name_substr.lower() in joint_name:
            return i
    return None


def map_to_world(point: dict) -> np.ndarray:
    """Convert mapper-output position (shoulder-anchored, image space)
    into PyBullet world coordinates."""
    x = float(point["x"])
    y = float(point["y"])
    # We ignore z (no reliable depth from RTMPose)

    sign_x = -1.0 if MIRROR_X else 1.0
    world_x = sign_x * x * SHOULDER_WIDTH_M
    world_y = 0.0  # forward/back = 0 for now (no depth data)
    world_z = TORSO_HEIGHT_Z - y * SHOULDER_WIDTH_M  # image Y down → world Z down
    return np.array([world_x, world_y, world_z])


def make_marker(color_rgba: tuple[float, float, float, float],
                radius: float = 0.04) -> int:
    """Create a colored debug sphere in PyBullet."""
    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color_rgba,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[0, 0, 0],
    )


def main() -> None:
    print("=" * 60)
    print("  SignLink PyBullet mannequin")
    print("=" * 60)

    # ─── Set up PyBullet GUI ────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)            # no gravity, mannequin won't fall
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # no Bullet UI panels
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=180,
        cameraPitch=-10,
        cameraTargetPosition=[0, 0, 1.0],
    )

    # ground plane
    p.loadURDF("plane.urdf")

    # humanoid model that ships with pybullet_data
    robot = p.loadURDF(
        "humanoid/humanoid.urdf",
        basePosition=[0, 0, 1.0],
        useFixedBase=True,
    )

    # ─── Discover arm joint indices ─────────────────────────────────
    if PRINT_JOINTS:
        print("\nAvailable joints in humanoid:")
        for i in range(p.getNumJoints(robot)):
            print(f"  [{i}]  {p.getJointInfo(robot, i)[1].decode('utf-8')}")
        print()

    # The pybullet_data humanoid uses these joint name fragments
    left_wrist_idx  = (find_joint_index(robot, "left_elbow")
                       or find_joint_index(robot, "left_lower_arm"))
    right_wrist_idx = (find_joint_index(robot, "right_elbow")
                       or find_joint_index(robot, "right_lower_arm"))

    if left_wrist_idx is None or right_wrist_idx is None:
        print("ERROR: could not find arm joints. See list above and adjust "
              "find_joint_index calls in this script.")
        return

    print(f"Left arm end-effector index:  {left_wrist_idx}")
    print(f"Right arm end-effector index: {right_wrist_idx}")

    # ─── Visual markers for wrist targets ───────────────────────────
    left_marker  = make_marker((1.0, 0.2, 0.2, 0.9)) if SHOW_TARGETS else None
    right_marker = make_marker((0.2, 0.4, 1.0, 0.9)) if SHOW_TARGETS else None

    # ─── UDP socket ─────────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)
    print(f"Listening on UDP 0.0.0.0:{PORT}\n")
    print("Move in front of the camera. Ctrl+C to stop.\n")

    last_print = time.time()
    packet_count = 0

    try:
        while True:
            # Drain UDP buffer — only act on the most recent packet
            latest = None
            while True:
                try:
                    data, _ = sock.recvfrom(65536)
                    latest = data
                except BlockingIOError:
                    break

            if latest is not None:
                try:
                    payload = json.loads(latest.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = None

                if payload is not None:
                    bones = payload.get("bone_transforms", {})
                    packet_count += 1

                    l_wrist = bones.get("pos_left_wrist")
                    r_wrist = bones.get("pos_right_wrist")

                    if l_wrist is not None:
                        target = map_to_world(l_wrist)
                        if SHOW_TARGETS:
                            p.resetBasePositionAndOrientation(
                                left_marker, target.tolist(), [0, 0, 0, 1])
                        # Solve IK for left arm
                        joint_targets = p.calculateInverseKinematics(
                            robot, left_wrist_idx, target.tolist(),
                            maxNumIterations=IK_ITERATIONS,
                        )
                        # Apply just the left arm joints (PyBullet returns
                        # full pose; we update everything but it's fine)
                        for j_idx, j_val in enumerate(joint_targets):
                            p.resetJointState(robot, j_idx, j_val)

                    if r_wrist is not None:
                        target = map_to_world(r_wrist)
                        if SHOW_TARGETS:
                            p.resetBasePositionAndOrientation(
                                right_marker, target.tolist(), [0, 0, 0, 1])
                        joint_targets = p.calculateInverseKinematics(
                            robot, right_wrist_idx, target.tolist(),
                            maxNumIterations=IK_ITERATIONS,
                        )
                        for j_idx, j_val in enumerate(joint_targets):
                            p.resetJointState(robot, j_idx, j_val)

            now = time.time()
            if now - last_print >= 2.0:
                print(f"packets received in last 2s: {packet_count}")
                packet_count = 0
                last_print = now

            p.stepSimulation()
            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        sock.close()
        p.disconnect()


if __name__ == "__main__":
    main()
