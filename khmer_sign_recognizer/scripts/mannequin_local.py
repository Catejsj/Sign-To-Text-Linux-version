"""Local 3D mannequin — an articulated humanoid rendered in Open3D, driven
live by RTMPose (body) + MediaPipe (hands) from src/capture.py.

Everything is ONE process on ONE webcam. No browser, no WebSocket, no Godot.
Two windows open:
  - OpenCV window  — the camera feed with the tracked skeleton overlay
  - Open3D window  — the 3D mannequin mirroring you

The mannequin is an artist's-mannequin style figure: tan capsule limbs,
ellipsoid torso, sphere head, line-rigged fingers. It is upper-body only
(head → hips) because that is what Khmer Sign Language uses and what the
RTMPose joints in capture.py cover.

Keys (focus the OpenCV camera window):
  q : quit
  m : show / hide the mannequin

Run:
  cd "D:\\Projects\\Sign to Text\\khmer_sign_recognizer"
  .\\venv\\Scripts\\Activate.ps1
  pip install open3d            # one time
  python scripts/mannequin_local.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.capture import LandmarkCapture            # noqa: E402
from src.utils import load_config, setup_logging   # noqa: E402

# MediaPipe hand bone topology — 21 edges connecting the 21 landmarks.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm edge
]

# Colours (Open3D uses 0..1 RGB)
BODY_COLOR  = (0.82, 0.71, 0.55)   # warm tan — the limbs/torso
JOINT_COLOR = (0.55, 0.45, 0.34)   # darker tan — joint balls
HAND_COLOR  = (0.20, 0.78, 1.00)   # cyan — finger lines

# Scene scale — the figure ends up roughly 1.5 units tall.
SCALE = 2.4


# ─────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────
def rot_from_to(d: np.ndarray) -> np.ndarray:
    """Rotation matrix aligning +Z to unit vector d (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, d))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def cyl_transform(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """4x4 placing a unit cylinder (height 1, axis Z) to span p0..p1."""
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-9:
        L = 1e-9
    R = rot_from_to(v / L)
    T = np.eye(4)
    T[:3, :3] = R @ np.diag([1.0, 1.0, L])
    T[:3, 3] = (p0 + p1) / 2.0
    return T


def blob_transform(center: np.ndarray, sx: float, sy: float, sz: float) -> np.ndarray:
    """4x4 placing a unit sphere as an ellipsoid of half-axes sx,sy,sz."""
    T = np.eye(4)
    T[:3, :3] = np.diag([sx, sy, sz])
    T[:3, 3] = center
    return T


class Part:
    """One solid mesh whose vertices are re-placed every frame from a 4x4."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh, color):
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        self.mesh = mesh
        self._base_v = np.asarray(mesh.vertices).copy()
        self._base_n = np.asarray(mesh.vertex_normals).copy()

    def apply(self, T: np.ndarray) -> None:
        R, t = T[:3, :3], T[:3, 3]
        v = self._base_v @ R.T + t
        self.mesh.vertices = o3d.utility.Vector3dVector(v)
        n = self._base_n @ R.T
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < 1e-9] = 1.0
        self.mesh.vertex_normals = o3d.utility.Vector3dVector(n / ln)


def make_cyl(color) -> Part:
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0,
                                                  resolution=14, split=1)
    return Part(m, color)


def make_sphere(color, res: int = 16) -> Part:
    m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=res)
    return Part(m, color)


# ─────────────────────────────────────────────────────────────────────
#  The mannequin
# ─────────────────────────────────────────────────────────────────────
class Mannequin:
    """An articulated upper-body humanoid. Build once, call update() per frame."""

    def __init__(self):
        # Solid body parts.
        self.head   = make_sphere(BODY_COLOR, res=20)
        self.neck   = make_cyl(BODY_COLOR)
        self.chest  = make_sphere(BODY_COLOR, res=20)
        self.pelvis = make_sphere(BODY_COLOR, res=20)
        self.uarm_l = make_cyl(BODY_COLOR)
        self.uarm_r = make_cyl(BODY_COLOR)
        self.farm_l = make_cyl(BODY_COLOR)
        self.farm_r = make_cyl(BODY_COLOR)
        self.j_sh_l = make_sphere(JOINT_COLOR, res=12)
        self.j_sh_r = make_sphere(JOINT_COLOR, res=12)
        self.j_el_l = make_sphere(JOINT_COLOR, res=12)
        self.j_el_r = make_sphere(JOINT_COLOR, res=12)
        self.j_wr_l = make_sphere(JOINT_COLOR, res=12)
        self.j_wr_r = make_sphere(JOINT_COLOR, res=12)

        self.solids = [
            self.chest, self.pelvis, self.neck, self.head,
            self.uarm_l, self.uarm_r, self.farm_l, self.farm_r,
            self.j_sh_l, self.j_sh_r, self.j_el_l, self.j_el_r,
            self.j_wr_l, self.j_wr_r,
        ]

        # Finger rigs — one LineSet per hand (cheap, updates fast).
        self.hand_l = self._make_hand()
        self.hand_r = self._make_hand()

        # Last-known joint positions, so a momentarily-lost joint freezes
        # instead of snapping to the origin.
        self.joints: dict[str, np.ndarray] = {}
        self.lhand: np.ndarray | None = None
        self.rhand: np.ndarray | None = None

    @staticmethod
    def _make_hand() -> o3d.geometry.LineSet:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
        ls.lines = o3d.utility.Vector2iVector(np.array(HAND_BONES, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(HAND_COLOR, (len(HAND_BONES), 1)))
        return ls

    def geometries(self) -> list:
        return [p.mesh for p in self.solids] + [self.hand_l, self.hand_r]

    def update(self, joints: dict[str, np.ndarray],
               lhand: np.ndarray | None, rhand: np.ndarray | None) -> None:
        # Merge new joints over last-known.
        self.joints.update(joints)
        j = self.joints
        need = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                "l_wrist", "r_wrist", "nose"]
        if not all(k in j for k in need):
            return   # wait until the upper body is tracked

        l_sh, r_sh = j["l_shoulder"], j["r_shoulder"]
        neck = (l_sh + r_sh) / 2.0
        sw = float(np.linalg.norm(l_sh - r_sh)) + 1e-6   # shoulder width

        # Hips: use them if RTMPose found them; otherwise synthesize a
        # pelvis below the shoulders so the torso still renders when the
        # framing is waist-up (the usual sign-language framing).
        if "l_hip" in j and "r_hip" in j:
            l_hip, r_hip = j["l_hip"], j["r_hip"]
        else:
            down = neck - j["nose"]
            down = down / (np.linalg.norm(down) + 1e-9)
            pelv_est = neck + down * sw * 1.5
            side = r_sh - l_sh
            l_hip = pelv_est - side * 0.30
            r_hip = pelv_est + side * 0.30
        pelv = (l_hip + r_hip) / 2.0
        chest_c = neck * 0.62 + pelv * 0.38
        torso_h = float(np.linalg.norm(neck - pelv)) + 1e-6

        limb_r  = sw * 0.085
        joint_r = sw * 0.11

        # Head — sit it above the nose, sized to the body.
        head_r = sw * 0.34
        up = (neck - pelv)
        up = up / (np.linalg.norm(up) + 1e-9)
        head_c = j["nose"] + up * head_r * 0.7

        # The body is a flat z=0 plane (RTMPose gives no depth), but the
        # torso/head are 3D ellipsoids with real thickness. Push each one
        # BACK along z by its own half-thickness so its FRONT surface sits
        # at ~z=0. Combined with the wrists being pulled forward (see
        # main()), the hands then pass cleanly in front instead of
        # phasing through the torso and head.
        zb = np.array([0.0, 0.0, -1.0])
        self.head.apply(blob_transform(head_c + zb * head_r,
                                       head_r, head_r * 1.15, head_r))
        self.neck.apply(self._cyl(neck + zb * head_r * 0.5,
                                  head_c + zb * head_r - up * head_r,
                                  limb_r * 0.9))
        self.chest.apply(blob_transform(chest_c + zb * sw * 0.34,
                                        sw * 0.60, torso_h * 0.40, sw * 0.34))
        self.pelvis.apply(blob_transform(pelv + zb * sw * 0.30,
                                         sw * 0.50, torso_h * 0.26, sw * 0.30))

        # Arms.
        self.uarm_l.apply(self._cyl(l_sh, j["l_elbow"], limb_r))
        self.uarm_r.apply(self._cyl(r_sh, j["r_elbow"], limb_r))
        self.farm_l.apply(self._cyl(j["l_elbow"], j["l_wrist"], limb_r * 0.85))
        self.farm_r.apply(self._cyl(j["r_elbow"], j["r_wrist"], limb_r * 0.85))

        # Joint balls — round out the capsule look.
        for part, key, scl in [
            (self.j_sh_l, "l_shoulder", 1.15), (self.j_sh_r, "r_shoulder", 1.15),
            (self.j_el_l, "l_elbow", 1.0),     (self.j_el_r, "r_elbow", 1.0),
            (self.j_wr_l, "l_wrist", 0.85),    (self.j_wr_r, "r_wrist", 0.85),
        ]:
            r = joint_r * scl
            part.apply(blob_transform(j[key], r, r, r))

        # Hands.
        if lhand is not None:
            self.lhand = lhand
        if rhand is not None:
            self.rhand = rhand
        if self.lhand is not None:
            self.hand_l.points = o3d.utility.Vector3dVector(self.lhand)
        if self.rhand is not None:
            self.hand_r.points = o3d.utility.Vector3dVector(self.rhand)

    @staticmethod
    def _cyl(p0: np.ndarray, p1: np.ndarray, radius: float) -> np.ndarray:
        """cyl_transform but also bakes the radius into the X/Y scale."""
        T = cyl_transform(p0, p1)
        T[:3, :3] = T[:3, :3] @ np.diag([radius, radius, 1.0])
        return T


# ─────────────────────────────────────────────────────────────────────
#  Landmark → 3D scene-space conversion
# ─────────────────────────────────────────────────────────────────────
def body_to_3d(pt: dict, W: int, H: int) -> np.ndarray:
    """RTMPose body point (pixel x,y) → scene 3D. X flipped so it mirrors
    you; Y flipped because image-y points down. Body has no depth (z=0)."""
    x = -(pt["x"] - W / 2.0) / H * SCALE
    y = -(pt["y"] - H / 2.0) / H * SCALE
    return np.array([x, y, 0.0])


def hand_to_3d(hand: dict, W: int, H: int, wrist_anchor: np.ndarray) -> np.ndarray | None:
    """MediaPipe hand dict {'0'..'20': {x,y,z}} → (21,3) array, translated
    so its own wrist (landmark 0) sits on the body wrist from RTMPose."""
    if not hand or "0" not in hand:
        return None
    pts = np.zeros((21, 3), dtype=np.float64)
    for i in range(21):
        lm = hand.get(str(i))
        if lm is None:
            return None
        x = -(lm["x"] * W - W / 2.0) / H * SCALE
        y = -(lm["y"] * H - H / 2.0) / H * SCALE
        z = -lm["z"] * (W / H) * SCALE * 0.5
        pts[i] = (x, y, z)
    offset = wrist_anchor - pts[0]
    return pts + offset


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config(str(ROOT / "config" / "settings.json"))
    setup_logging(cfg)
    W = cfg["capture"]["width"]
    H = cfg["capture"]["height"]

    print("Starting camera + RTMPose + MediaPipe...")
    capture = LandmarkCapture(cfg)
    if not capture.start():
        print("ERROR: camera failed to start")
        return

    mannequin = Mannequin()

    vis = o3d.visualization.Visualizer()
    vis.create_window("SignLink — Local Mannequin", width=900, height=900)
    for g in mannequin.geometries():
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.06, 0.09])
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_front([0.0, 0.0, 1.0])
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_lookat([0.0, 0.0, 0.0])
    vc.set_zoom(0.9)

    mannequin_shown = True
    print("\nRunning. Focus the camera window:  q = quit,  m = show/hide mannequin.\n")

    try:
        while True:
            # ── pull latest landmarks ──
            with capture.result_lock:
                pose = dict(capture.latest_pose)
                lh = dict(capture.latest_left_hand)
                rh = dict(capture.latest_right_hand)

            joints: dict[str, np.ndarray] = {}
            name_map = {
                "left_shoulder": "l_shoulder", "right_shoulder": "r_shoulder",
                "left_elbow": "l_elbow",       "right_elbow": "r_elbow",
                "left_wrist": "l_wrist",       "right_wrist": "r_wrist",
                "left_hip": "l_hip",           "right_hip": "r_hip",
                "nose": "nose",
            }
            for src_name, dst_name in name_map.items():
                if src_name in pose:
                    joints[dst_name] = body_to_3d(pose[src_name], W, H)

            # Pull the wrists toward the camera. The body is a flat z=0
            # plane, so without this the hands sit inside the torso
            # volume. Moving the wrists forward angles the forearms out
            # and puts the hands cleanly in front — like a real signer.
            if "l_shoulder" in joints and "r_shoulder" in joints:
                sw = float(np.linalg.norm(joints["l_shoulder"]
                                          - joints["r_shoulder"]))
                fwd = np.array([0.0, 0.0, sw * 0.32])
                if "l_wrist" in joints:
                    joints["l_wrist"] = joints["l_wrist"] + fwd
                if "r_wrist" in joints:
                    joints["r_wrist"] = joints["r_wrist"] + fwd

            lhand = rhand = None
            if "l_wrist" in joints:
                lhand = hand_to_3d(lh, W, H, joints["l_wrist"])
            if "r_wrist" in joints:
                rhand = hand_to_3d(rh, W, H, joints["r_wrist"])

            if mannequin_shown:
                mannequin.update(joints, lhand, rhand)
                for g in mannequin.geometries():
                    vis.update_geometry(g)

            if not vis.poll_events():
                break          # Open3D window closed
            vis.update_renderer()

            # ── camera window with the tracked skeleton overlay ──
            ret, frame = capture.read_frame()
            if ret and frame is not None:
                cv2.imshow("SignLink — Camera (q quit, m toggle mannequin)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mannequin_shown = not mannequin_shown
                for g in mannequin.geometries():
                    if mannequin_shown:
                        vis.add_geometry(g, reset_bounding_box=False)
                    else:
                        vis.remove_geometry(g, reset_bounding_box=False)
                print(f"mannequin {'shown' if mannequin_shown else 'hidden'}")
    finally:
        capture.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("stopped.")


if __name__ == "__main__":
    main()
