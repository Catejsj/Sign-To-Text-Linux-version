"""Synthetic-signer viewer — live 3D mannequins of synthetic signers.

ONE webcam, ONE process. You sign in front of the camera. Your motion is
captured (RTMPose body + MediaPipe hands) and retargeted live onto several
mannequins, each with a DIFFERENT body — different arm length, shoulder
width, hand size. Same sign, different bodies: that is synthetic data,
shown in real time.

There is no mannequin of you — you are already in the camera window, so a
copy would be redundant. The 3D window shows only the synthetic signers.

Two windows open:
  - OpenCV window  — your camera feed with the tracked skeleton overlay
  - Open3D window  — the synthetic signers, posed live by your motion

Each figure is an artist's-mannequin style humanoid: tan capsule limbs,
ellipsoid torso, sphere head, line-rigged fingers. Upper-body only.

Keys (focus the camera window):
  q : quit
  m : show / hide the synthetic signers

Run:
  python scripts/mannequin_local.py              # 2 synthetic signers
  python scripts/mannequin_local.py --synthetic 4

Playback mode (no camera, animates saved .npy takes):
  python scripts/mannequin_local.py --playback data/sequences_v2/autsl --count 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Open3D renders through GLFW, and GLFW's Wayland path cannot initialize GLEW
# here — create_window() just returns False and the mannequin never appears.
# Steering GLFW to X11 (XWayland) fixes it. GLFW reads the environment when the
# window is created, so this only has to run before the first create_window();
# doing it at import time is simply the easiest place to guarantee that.
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ.setdefault("DISPLAY", ":0")

import open3d as o3d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# cv2 + capture are only needed for the live (camera) mode. Importing them
# inside main() avoids requiring a working camera/MediaPipe setup just to
# run --playback.

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

# How many frames a lost hand is held before it's hidden. At ~12-15 fps
# this is roughly 0.7 s — long enough to ride out a brief detection drop,
# short enough that a genuinely-gone hand doesn't linger as a ghost.
HAND_HIDE_AGE = 10


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

    def __init__(self, body_color=BODY_COLOR, joint_color=JOINT_COLOR):
        # Solid body parts. Colors are parameters so synthetic mannequins
        # can be tinted differently from the real one.
        self.head   = make_sphere(body_color, res=20)
        self.neck   = make_cyl(body_color)
        self.chest  = make_sphere(body_color, res=20)
        self.pelvis = make_sphere(body_color, res=20)
        self.uarm_l = make_cyl(body_color)
        self.uarm_r = make_cyl(body_color)
        self.farm_l = make_cyl(body_color)
        self.farm_r = make_cyl(body_color)
        self.j_sh_l = make_sphere(joint_color, res=12)
        self.j_sh_r = make_sphere(joint_color, res=12)
        self.j_el_l = make_sphere(joint_color, res=12)
        self.j_el_r = make_sphere(joint_color, res=12)
        self.j_wr_l = make_sphere(joint_color, res=12)
        self.j_wr_r = make_sphere(joint_color, res=12)

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
        self.lhand_age = 0    # frames since this hand was last detected
        self.rhand_age = 0

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

        # Hands. A freshly-detected hand resets its age; a lost hand ages.
        # Once a hand is stale past HAND_HIDE_AGE we collapse its LineSet
        # onto the wrist (all lines zero-length = invisible) so it stops
        # lingering as a frozen ghost.
        if lhand is not None:
            self.lhand, self.lhand_age = lhand, 0
        else:
            self.lhand_age += 1
        if rhand is not None:
            self.rhand, self.rhand_age = rhand, 0
        else:
            self.rhand_age += 1

        if self.lhand is not None:
            pts = (self.lhand if self.lhand_age <= HAND_HIDE_AGE
                   else np.tile(j["l_wrist"], (21, 1)))
            self.hand_l.points = o3d.utility.Vector3dVector(pts)
        if self.rhand is not None:
            pts = (self.rhand if self.rhand_age <= HAND_HIDE_AGE
                   else np.tile(j["r_wrist"], (21, 1)))
            self.hand_r.points = o3d.utility.Vector3dVector(pts)

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


def retarget_scene(joints: dict[str, np.ndarray],
                   lhand: np.ndarray | None, rhand: np.ndarray | None,
                   sh: float, ua: float, fa: float, hd: float,
                   xoff: float) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Rebuild a scene-space pose onto a DIFFERENT body and shift it sideways.

    Bone lengths are scaled (shoulder width, upper arm, forearm, hand);
    every joint ANGLE is preserved — so it's the same sign performed by a
    different-bodied signer. This is the synthetic generator, run live,
    per frame. Same math as src/v2/retarget.py, in scene coordinates.
    """
    j = {k: v.copy() for k, v in joints.items()}
    if "l_shoulder" in joints and "r_shoulder" in joints:
        mid = (joints["l_shoulder"] + joints["r_shoulder"]) / 2.0
        for s in ("l", "r"):
            sk, ek, wk = f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"
            if sk not in joints:
                continue
            j[sk] = mid + (joints[sk] - mid) * sh
            if ek in joints:
                j[ek] = j[sk] + (joints[ek] - joints[sk]) * ua
                if wk in joints:
                    j[wk] = j[ek] + (joints[wk] - joints[ek]) * fa

    new_lhand = new_rhand = None
    if lhand is not None and "l_wrist" in joints:
        new_lhand = j["l_wrist"] + (lhand - joints["l_wrist"]) * hd
    if rhand is not None and "r_wrist" in joints:
        new_rhand = j["r_wrist"] + (rhand - joints["r_wrist"]) * hd

    off = np.array([xoff, 0.0, 0.0])
    for k in j:
        j[k] = j[k] + off
    if new_lhand is not None:
        new_lhand = new_lhand + off
    if new_rhand is not None:
        new_rhand = new_rhand + off
    return j, new_lhand, new_rhand


# ─────────────────────────────────────────────────────────────────────
#  Playback mode — animate the mannequin from saved .npy takes
# ─────────────────────────────────────────────────────────────────────
def _frame_to_scene(frame: np.ndarray) -> np.ndarray:
    """A saved (48, 3) frame → mannequin scene coords.

    The saved data uses MediaPipe / image conventions:
      - x: pixel column (0 = image left, larger = image right). A person
        facing the camera has their anatomical LEFT side on the image RIGHT,
        i.e. at HIGH x.
      - y: pixel row (0 = top, larger = bottom). Scene y goes up, so flip.
      - z: smaller = closer to camera. Scene z grows toward the viewer
        (default Open3D camera looks down -z), so flip.

    The live `body_to_3d` and `hand_to_3d` paths flip all three for the
    same reason. We do the same here so playback handedness matches what
    you'd see in live mode.
    """
    out = frame.astype(np.float64).copy()
    out[:, 0] = -out[:, 0]
    out[:, 1] = -out[:, 1]
    out[:, 2] = -out[:, 2]
    return out


def _prepare_scene_frame(scene: np.ndarray) -> np.ndarray:
    """Match the live-mode body/hand depth convention.

    The mannequin renderer assumes the body sits at z=0 (a flat plane) and
    that hands are pushed slightly in front of the torso so they don't
    phase through the chest. Live mode enforces this by setting body z=0
    and shifting wrists forward by ~0.32 * shoulder_width. We do the same
    here. Hand-internal relative structure is preserved by translating
    each whole hand so its own landmark-0 lands on the (now forward-shifted)
    body wrist.
    """
    out = scene.copy()
    # Body joints 0..5 → flat plane (matches live mode body_to_3d).
    out[:6, 2] = 0.0

    sw = float(np.linalg.norm(out[0] - out[1])) + 1e-6
    fwd = sw * 0.32
    out[4, 2] = fwd   # L_WRIST forward
    out[5, 2] = fwd   # R_WRIST forward

    # Realign each hand to its body wrist while preserving the hand's
    # internal shape — translate the whole 21-point cluster so its own
    # wrist (landmark 0) sits on the body wrist.
    out[6:27]  += out[4] - out[6]
    out[27:48] += out[5] - out[27]
    return out


def _label_for(npy_path: Path) -> str:
    """Read the label from the .json sidecar, fall back to the folder name."""
    json_path = npy_path.with_suffix(".json")
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return f"{meta.get('label', npy_path.parent.name)}  " \
               f"[signer={meta.get('signer_id', '?')}]"
    except Exception:
        return npy_path.parent.name


def run_playback(folder: Path, count: int, fps: float = 12.0,
                 prefer_view: str = "clean") -> None:
    """Animate `count` random saved takes from `folder` in the 3D mannequin.

    Picks files matching `*real__<view>__*.npy` so we only play back real
    captures (not synthetic retargeted ones — those would be redundant in
    a viewer whose whole point is to verify the import). Pass
    prefer_view='noisy' to use the raw [0,1] image-space view instead of
    the shoulder-normalized clean view.
    """
    pattern = f"*real__{prefer_view}__*.npy"
    files = sorted(folder.rglob(pattern))
    if not files:
        sys.exit(f"no '{pattern}' files found under {folder}")

    rng = np.random.default_rng()
    pick = min(count, len(files))
    chosen = rng.choice(np.array(files, dtype=object), size=pick, replace=False)

    print(f"playback: {len(files)} files available, showing {pick}")
    print("close the Open3D window to quit.\n")

    mannequin = Mannequin()
    vis = o3d.visualization.Visualizer()
    vis.create_window("SignLink — Playback", width=1100, height=820)
    for g in mannequin.geometries():
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.06, 0.09])
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_front([0.0, 0.0, 1.0])
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_lookat([0.0, 0.0, 0.0])
    vc.set_zoom(0.75)

    dt = 1.0 / max(fps, 1.0)
    try:
        for k, npy_path in enumerate(chosen, start=1):
            clip = np.load(npy_path)
            if clip.shape[1:] != (48, 3):
                print(f"  [skip] {npy_path.name}: unexpected shape {clip.shape}")
                continue

            label = _label_for(Path(npy_path))
            print(f"  [{k}/{pick}] {label}   ({Path(npy_path).name})")

            for t in range(clip.shape[0]):
                scene = _prepare_scene_frame(_frame_to_scene(clip[t]))

                joints = {
                    "l_shoulder": scene[0],
                    "r_shoulder": scene[1],
                    "l_elbow":    scene[2],
                    "r_elbow":    scene[3],
                    "l_wrist":    scene[4],
                    "r_wrist":    scene[5],
                }
                # The saved data has no nose. Synthesize one above the
                # shoulder midpoint so the head/neck render reasonably.
                mid = (joints["l_shoulder"] + joints["r_shoulder"]) / 2.0
                sw = float(np.linalg.norm(
                    joints["l_shoulder"] - joints["r_shoulder"])) + 1e-6
                joints["nose"] = mid + np.array([0.0, sw * 0.45, 0.0])

                lhand = scene[6:27]
                rhand = scene[27:48]
                mannequin.update(joints, lhand, rhand)

                for g in mannequin.geometries():
                    vis.update_geometry(g)
                if not vis.poll_events():
                    return
                vis.update_renderer()
                time.sleep(dt)

            # Brief pause between takes so the eye can register the change.
            for _ in range(int(0.5 / dt)):
                if not vis.poll_events():
                    return
                time.sleep(dt)

        print("\nplayback done — close the window to exit.")
        while vis.poll_events():
            vis.update_renderer()
            time.sleep(dt)
    finally:
        vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=1, metavar="N",
                    help="number of synthetic signers to show — each a "
                         "different body, posed live by your motion. "
                         "Default 1 (= 2 data sources total: your real "
                         "landmarks + 1 synthetic body). More = lower FPS.")
    ap.add_argument("--playback", type=str, default=None, metavar="FOLDER",
                    help="don't open the camera; instead animate saved "
                         "takes from FOLDER (e.g. data/sequences_v2/autsl). "
                         "Useful for visually sanity-checking imported "
                         "datasets.")
    ap.add_argument("--count", type=int, default=5, metavar="N",
                    help="how many random takes to play in --playback "
                         "mode (default 5).")
    ap.add_argument("--view", choices=["clean", "noisy"], default="clean",
                    help="which saved view to play back (default: clean).")
    ap.add_argument("--fps", type=float, default=12.0,
                    help="playback speed in --playback mode "
                         "(default 12; the takes were recorded at 30, "
                         "so 12 plays them slow enough to follow).")
    args = ap.parse_args()

    if args.playback:
        folder = Path(args.playback)
        if not folder.exists():
            sys.exit(f"--playback folder does not exist: {folder}")
        run_playback(folder, count=args.count, prefer_view=args.view,
                     fps=args.fps)
        return

    # Live mode needs the camera + landmark pipeline. Import here so
    # --playback works on a machine with no camera / no MediaPipe.
    import cv2                                              # noqa: F401
    from src.capture import LandmarkCapture                 # noqa: E402
    from src.utils import load_config, setup_logging        # noqa: E402

    cfg = load_config(str(ROOT / "config" / "settings.json"))
    setup_logging(cfg)
    W = cfg["capture"]["width"]
    H = cfg["capture"]["height"]

    print("Starting camera + RTMPose + MediaPipe...")
    capture = LandmarkCapture(cfg)
    if not capture.start():
        print("ERROR: camera failed to start")
        return

    rng = np.random.default_rng()

    # Synthetic signers. Each is a mannequin with a fixed random body
    # (shoulder width, upper-arm, forearm, hand size), posed live by YOUR
    # motion. There is no mannequin of YOU — you are already in the camera
    # window, a copy would be redundant. These figures ARE the synthetic
    # data: the same sign, on bodies that are not yours.
    N = max(1, args.synthetic)
    spacing = 2.6
    xs = (np.arange(N) - (N - 1) / 2.0) * spacing   # centred row of figures
    synths: list[tuple[Mannequin, dict]] = []
    for i in range(N):
        body = dict(
            sh=float(rng.uniform(0.80, 1.20)),
            ua=float(rng.uniform(0.80, 1.20)),
            fa=float(rng.uniform(0.80, 1.20)),
            hd=float(rng.uniform(0.80, 1.20)),
            xoff=float(xs[i]),
        )
        synths.append((Mannequin(), body))

    vis = o3d.visualization.Visualizer()
    vis.create_window("SignLink — Synthetic Signers", width=1200, height=850)
    for mq, _ in synths:
        for g in mq.geometries():
            vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.06, 0.09])
    opt.light_on = True

    vc = vis.get_view_control()
    vc.set_front([0.0, 0.0, 1.0])
    vc.set_up([0.0, 1.0, 0.0])
    vc.set_lookat([0.0, 0.0, 0.0])
    vc.set_zoom(0.75 + 0.14 * N)

    shown = True
    print(f"\n{N} synthetic signer(s) — same motion as you, different bodies.")
    print("Focus the camera window:  q = quit,  m = show/hide.\n")

    try:
        while True:
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

            # Pull wrists toward the camera so hands sit in front of the
            # flat z=0 body plane instead of inside the torso.
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

            if shown:
                for mq, body in synths:
                    sj, sl, sr = retarget_scene(
                        joints, lhand, rhand,
                        body["sh"], body["ua"], body["fa"], body["hd"],
                        body["xoff"])
                    mq.update(sj, sl, sr)
                for mq, _ in synths:
                    for g in mq.geometries():
                        vis.update_geometry(g)

            if not vis.poll_events():
                break          # Open3D window closed
            vis.update_renderer()

            ret, frame = capture.read_frame()
            if ret and frame is not None:
                cv2.imshow("SignLink — Camera (q quit, m toggle)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                shown = not shown
                for mq, _ in synths:
                    for g in mq.geometries():
                        if shown:
                            vis.add_geometry(g, reset_bounding_box=False)
                        else:
                            vis.remove_geometry(g, reset_bounding_box=False)
                print(f"synthetic signers {'shown' if shown else 'hidden'}")
    finally:
        capture.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("stopped.")


if __name__ == "__main__":
    main()
