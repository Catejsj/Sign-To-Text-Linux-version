"""Real-time 3D mannequin visualizer — replaces the Godot mannequin.

We have every joint position from RTMPose + MediaPipe, so we just draw
a humanoid out of primitives (head, torso, limbs) at those positions.
No IK, no rigging, no twist hacks.

Run this INSTEAD of Godot:
    python scripts\\skeleton_viz.py

Pipeline (unchanged):
    camera (Windows) -> port 9999 -> WSL processing -> port 8888 -> this script
"""
from __future__ import annotations

import json
import socket
import sys
import time

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Run:")
    print("  .\\venv\\Scripts\\Activate.ps1")
    print("  pip install open3d")
    sys.exit(1)


# ─── TUNING KNOBS ────────────────────────────────────────────────────
PORT          = 8888
SCALE         = 1.5      # how big the mannequin appears
MIRROR_X      = True     # flip left/right for mirror image

# Body part sizes — original chunky values that worked best
HEAD_RADIUS   = 0.18
JOINT_RADIUS  = 0.09
BONE_RADIUS   = 0.06
TORSO_LENGTH  = 0.65
TORSO_RADIUS  = 0.20

# Colors
SKIN_COLOR    = [0.95, 0.78, 0.62]
TORSO_COLOR   = [0.45, 0.55, 0.75]    # shirt
LEFT_TINT     = [0.95, 0.55, 0.55]    # mark left side red-ish
RIGHT_TINT    = [0.55, 0.65, 0.95]    # mark right side blue-ish

# Robustness against tracker glitches (joints leaving frame, occlusions)
MAX_JUMP      = 1.5      # reject any per-frame move bigger than this
                         # (in world units — set to 0 to disable)
SMOOTH        = 0.7      # lerp factor 0..1, lower = smoother but laggier
                         # 0.7 = snappy, 0.4 = smooth-and-laggy, 1.0 = no smoothing

# Hide a joint when the tracker locks it at literally the same value.
# Detection runs on the RAW incoming data (before smoothing) so that
# natural stillness (which has tiny noise) doesn't get flagged.
STALE_THRESHOLD_FRAMES = 60      # frames of identical raw input before hiding
STALE_MOVE_EPSILON     = 0.0005  # very small — only catches frozen tracker output
# ─────────────────────────────────────────────────────────────────────


JOINTS = [
    "pos_left_shoulder", "pos_right_shoulder",
    "pos_left_elbow",    "pos_right_elbow",
    "pos_left_wrist",    "pos_right_wrist",
]

JOINT_COLORS = {
    "pos_left_shoulder":  LEFT_TINT,
    "pos_right_shoulder": RIGHT_TINT,
    "pos_left_elbow":     LEFT_TINT,
    "pos_right_elbow":    RIGHT_TINT,
    "pos_left_wrist":     LEFT_TINT,
    "pos_right_wrist":    RIGHT_TINT,
}

ARM_BONES = [
    ("pos_left_shoulder",  "pos_left_elbow"),
    ("pos_left_elbow",     "pos_left_wrist"),
    ("pos_right_shoulder", "pos_right_elbow"),
    ("pos_right_elbow",    "pos_right_wrist"),
]


def map_to_world(point: dict) -> np.ndarray:
    x = float(point["x"])
    y = float(point["y"])
    sign_x = -1.0 if MIRROR_X else 1.0
    return np.array([
        sign_x * x * SCALE,
        0.0,
        -y * SCALE + 1.2,
    ])


def make_sphere(radius: float, color: list[float]) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def make_cylinder(radius: float, color: list[float]) -> o3d.geometry.TriangleMesh:
    c = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=1.0, resolution=24)
    c.compute_vertex_normals()
    c.paint_uniform_color(color)
    return c


def cylinder_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """4x4 transform that places a default unit cylinder along p1->p2."""
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < 1e-6:
        return np.eye(4)

    scale = np.eye(4)
    scale[2, 2] = length

    z_axis = np.array([0.0, 0.0, 1.0])
    target = diff / length
    v = np.cross(z_axis, target)
    s_ = float(np.linalg.norm(v))
    c_ = float(np.dot(z_axis, target))
    if s_ < 1e-6:
        rot = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([
            [0.0,   -v[2],  v[1]],
            [v[2],   0.0,  -v[0]],
            [-v[1],  v[0],  0.0],
        ])
        rot = np.eye(3) + vx + vx @ vx * ((1.0 - c_) / (s_ * s_))

    rot4 = np.eye(4)
    rot4[:3, :3] = rot

    trans = np.eye(4)
    trans[:3, 3] = (p1 + p2) / 2.0

    return trans @ rot4 @ scale


def main() -> None:
    print("=" * 60)
    print("  SignLink real-time mannequin")
    print("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)
    print(f"Listening on UDP 0.0.0.0:{PORT}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SignLink Mannequin", width=900, height=800)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.12, 0.16])
    opt.light_on = True

    # Floor grid
    grid_pts = []
    grid_lines = []
    for i in range(-6, 7):
        a = len(grid_pts)
        grid_pts.append([i * 0.25, -1.5, 0])
        grid_pts.append([i * 0.25,  1.5, 0])
        grid_lines.append([a, a + 1])
        a = len(grid_pts)
        grid_pts.append([-1.5, i * 0.25, 0])
        grid_pts.append([ 1.5, i * 0.25, 0])
        grid_lines.append([a, a + 1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(grid_pts)
    grid.lines  = o3d.utility.Vector2iVector(grid_lines)
    grid.paint_uniform_color([0.25, 0.25, 0.30])
    vis.add_geometry(grid)

    # Body parts
    head    = make_sphere(HEAD_RADIUS, SKIN_COLOR)
    torso   = make_cylinder(TORSO_RADIUS, TORSO_COLOR)

    joint_meshes: dict[str, o3d.geometry.TriangleMesh] = {
        j: make_sphere(JOINT_RADIUS, JOINT_COLORS[j]) for j in JOINTS
    }
    arm_bone_meshes = [make_cylinder(BONE_RADIUS, SKIN_COLOR) for _ in ARM_BONES]

    head_last_center = np.zeros(3)
    joint_last_centers = {j: np.zeros(3) for j in JOINTS}
    torso_last_xform = np.eye(4)
    bone_last_xforms = [np.eye(4) for _ in ARM_BONES]

    # Stale-joint state. We track the RAW incoming position so we can
    # detect "tracker stuck on the exact same value" without being fooled
    # by natural stillness (which still has tiny noise).
    stale_counts: dict[str, int] = {j: 0 for j in JOINTS}
    last_raw: dict[str, np.ndarray] = {}
    HIDE_POSITION = np.array([0.0, 0.0, -100.0])

    vis.add_geometry(head)
    vis.add_geometry(torso)
    for m in joint_meshes.values():
        vis.add_geometry(m)
    for b in arm_bone_meshes:
        vis.add_geometry(b)

    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 1.0])
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.7)

    print("Mannequin running. Move in front of the camera. Close window to quit.")
    print("Controls: drag = rotate · scroll = zoom · shift+drag = pan · R = reset view\n")

    last_print = time.time()
    packets = 0
    have_data = False

    try:
        while True:
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
                    bones_dict = payload.get("bone_transforms", {})
                    packets += 1

                    new_positions: dict[str, np.ndarray] = {}
                    for jname in JOINTS:
                        if jname not in bones_dict:
                            continue
                        candidate = map_to_world(bones_dict[jname])

                        last = joint_last_centers[jname]
                        is_first = float(np.linalg.norm(last)) <= 1e-6

                        if (not is_first and MAX_JUMP > 0.0
                                and float(np.linalg.norm(candidate - last)) > MAX_JUMP):
                            new_positions[jname] = last
                        else:
                            if SMOOTH < 1.0 and not is_first:
                                new_positions[jname] = last + (candidate - last) * SMOOTH
                            else:
                                new_positions[jname] = candidate

                        # Stale detection runs on the RAW candidate (pre-smooth)
                        # so natural stillness doesn't get falsely flagged.
                        if jname in last_raw:
                            raw_delta = float(np.linalg.norm(candidate - last_raw[jname]))
                            if raw_delta < STALE_MOVE_EPSILON:
                                stale_counts[jname] += 1
                            else:
                                stale_counts[jname] = 0
                        last_raw[jname] = candidate

                    is_stale: dict[str, bool] = {
                        j: stale_counts[j] > STALE_THRESHOLD_FRAMES for j in JOINTS
                    }

                    def disp(jname: str) -> np.ndarray | None:
                        if jname not in new_positions:
                            return None
                        if is_stale[jname]:
                            return HIDE_POSITION
                        return new_positions[jname]

                    # Move joint spheres
                    for jname, mesh in joint_meshes.items():
                        d = disp(jname)
                        if d is None:
                            continue
                        mesh.translate(d - joint_last_centers[jname], relative=True)
                        joint_last_centers[jname] = d
                        vis.update_geometry(mesh)

                    # Move arm bones. CRITICAL: if either endpoint is stale,
                    # hide the WHOLE bone, otherwise we get a long bone
                    # stretching from a real joint down to (0,0,-100).
                    for i, (a, b) in enumerate(ARM_BONES):
                        if a not in new_positions or b not in new_positions:
                            continue
                        if is_stale[a] or is_stale[b]:
                            # Both endpoints hidden — collapse the bone offscreen
                            da = HIDE_POSITION
                            db = HIDE_POSITION + np.array([0.0, 0.0, 0.001])
                        else:
                            da = new_positions[a]
                            db = new_positions[b]
                        new_x = cylinder_transform(da, db)
                        arm_bone_meshes[i].transform(np.linalg.inv(bone_last_xforms[i]))
                        arm_bone_meshes[i].transform(new_x)
                        bone_last_xforms[i] = new_x
                        vis.update_geometry(arm_bone_meshes[i])

                    # Torso + head from shoulder midpoint (only if both shoulders are real)
                    ls_real = new_positions.get("pos_left_shoulder")
                    rs_real = new_positions.get("pos_right_shoulder")
                    shoulders_visible = (
                        ls_real is not None and rs_real is not None
                        and not is_stale["pos_left_shoulder"]
                        and not is_stale["pos_right_shoulder"]
                    )

                    if shoulders_visible:
                        shoulder_mid = (ls_real + rs_real) / 2.0

                        head_target = shoulder_mid + np.array(
                            [0.0, 0.0, HEAD_RADIUS + 0.10])
                        head.translate(head_target - head_last_center, relative=True)
                        head_last_center = head_target
                        vis.update_geometry(head)

                        torso_top    = shoulder_mid
                        torso_bottom = shoulder_mid + np.array(
                            [0.0, 0.0, -TORSO_LENGTH])
                        new_torso_x = cylinder_transform(torso_top, torso_bottom)
                        torso.transform(np.linalg.inv(torso_last_xform))
                        torso.transform(new_torso_x)
                        torso_last_xform = new_torso_x
                        vis.update_geometry(torso)

                    have_data = True

            still_open = vis.poll_events()
            if not still_open:
                break
            vis.update_renderer()

            now = time.time()
            if now - last_print >= 2.0:
                tag = "" if have_data else "  (no packets yet — is WSL running?)"
                print(f"packets in last 2s: {packets}{tag}")
                packets = 0
                last_print = now

            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        sock.close()
        vis.destroy_window()


if __name__ == "__main__":
    main()
