"""Interchangeable bodies for the 3D view.

The live window has always drawn one thing: an articulated figure fed a dict of
joint positions. This module makes *which* figure a runtime choice.

    classic   the tan capsule mannequin the project has always had
    anime     a rigged VRM / glTF character, skinned to the same joints

Both satisfy the same two-method contract, which is all `webapp/engine.py` and
`scripts/mannequin_local.py` ever call:

    geometries()                    → the Open3D objects to add to a scene
    update(joints, lhand, rhand)    → re-place them for this frame

so a skin can be swapped without either caller knowing what it is holding.

Hands
-----
Both skins draw hands as the 21-point MediaPipe landmark rig, not as mesh
fingers. For `anime` this is deliberate: the capture gives 21 points per hand
but a VRM's fingers are 15 bones with roll axes we have no way to infer, and a
frozen open palm during fingerspelling reads worse than an honest wireframe.
`hand_scale` in `AvatarRig` collapses the model's own hands so the rig shows
through.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Importing mannequin_local also applies the GLFW/Wayland fix it does at import
# time, which has to happen before any Open3D window is created.
from scripts.mannequin_local import (
    HAND_BONES, HAND_COLOR, HAND_HIDE_AGE, Mannequin,
)

import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
AVATAR_DIR = ROOT / "assets" / "avatars"
AVATAR_SUFFIXES = (".vrm", ".glb", ".gltf")

SKINS = ("classic", "anime")
DEFAULT_SKIN = "classic"

# Skinning is linear in vertices and runs on the main thread, next to camera
# capture and pose estimation. Measured on a 200k-vertex character: 24k costs
# 13 ms/frame, 40k costs 18 ms, 100k costs 42 ms. 40k is the point where the
# face and hat edges are crisp and there is still room in a 33 ms frame.
# Raise it with --max-vertices if your machine has the headroom.
DEFAULT_MAX_VERTICES = 40000

# Loading and colour-baking a VRM costs a second or two, and the engine builds
# up to four mannequins at once. Parse each file once and share the geometry;
# every instance still gets its own Open3D buffers to write into.
_MESH_CACHE: dict[tuple[str, int, float], tuple] = {}


class AvatarUnavailable(RuntimeError):
    """No usable avatar file. The message is written to be shown in the UI."""


def find_avatar(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """The avatar file to use, or None if there is not one.

    An explicit path wins. Otherwise the newest file in `assets/avatars/`,
    so dropping in a new model makes it the one that gets used.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    if not AVATAR_DIR.is_dir():
        return None
    found = [p for p in AVATAR_DIR.iterdir()
             if p.suffix.lower() in AVATAR_SUFFIXES and p.is_file()]
    if not found:
        return None
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def avatar_status(explicit: Optional[str | Path] = None) -> dict:
    """What the web UI needs to know about the anime skin, without loading it.

    Deliberately cheap — this runs on every `/api/state` poll, so it stats the
    directory and stops. Whether the file actually *works* is what
    `scripts/check_avatar.py` is for.
    """
    path = find_avatar(explicit)
    if path is None:
        return {
            "available": False,
            "name": None,
            "hint": (f"Drop a .vrm (or .glb) into {AVATAR_DIR.relative_to(ROOT)}"
                     f" and reload."),
        }
    return {"available": True, "name": path.name, "hint": ""}


def _load_shared(path: Path, max_vertices: int, hand_scale: float):
    key = (str(path.resolve()), int(max_vertices), float(hand_scale))
    if key not in _MESH_CACHE:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.gltf_min import GltfError
        from src.avatar_pose import load_rig

        try:
            rig = load_rig(path, max_vertices=max_vertices,
                           hand_scale=hand_scale)
            mesh = rig.mesh
        except (GltfError, ValueError, KeyError, IndexError) as exc:
            raise AvatarUnavailable(
                f"{path.name} could not be used: {exc}  "
                f"Run  ./venv/bin/python scripts/check_avatar.py {path}  "
                f"for the details."
            ) from exc
        _MESH_CACHE[key] = (rig, mesh)
    return _MESH_CACHE[key]


class AnimeMannequin:
    """A rigged character driven by the same joints as the classic mannequin.

    `body_scale` exists because the engine builds several mannequins side by
    side to show one motion on different bodies. The classic skin varies bone
    lengths; a fixed mesh cannot, so it varies overall size instead. The
    motion is identical either way — that is the point of the display.
    """

    def __init__(self, avatar: Optional[str | Path] = None,
                 max_vertices: int = DEFAULT_MAX_VERTICES, hand_scale: float = 0.0,
                 body_scale: float = 1.0):
        path = find_avatar(avatar)
        if path is None:
            raise AvatarUnavailable(avatar_status(avatar)["hint"])

        self.path = path
        self.body_scale = float(body_scale)
        self.rig, shared = _load_shared(path, max_vertices, hand_scale)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(shared.positions)
        self.mesh.triangles = o3d.utility.Vector3iVector(shared.triangles)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(shared.colors)
        if shared.normals.any():
            self.mesh.vertex_normals = o3d.utility.Vector3dVector(
                shared.normals)
        else:
            self.mesh.compute_vertex_normals()

        self.hand_l = _make_hand()
        self.hand_r = _make_hand()
        self.joints: dict[str, np.ndarray] = {}
        self.lhand: np.ndarray | None = None
        self.rhand: np.ndarray | None = None
        self.lhand_age = 0
        self.rhand_age = 0

    def geometries(self) -> list:
        return [self.mesh, self.hand_l, self.hand_r]

    def update(self, joints: dict[str, np.ndarray],
               lhand: np.ndarray | None, rhand: np.ndarray | None) -> None:
        # Same last-known-value policy as the classic skin: a joint that drops
        # out for a frame freezes rather than snapping to the origin.
        self.joints.update(joints)
        j = self.joints
        if not all(k in j for k in ("l_shoulder", "r_shoulder", "l_elbow",
                                    "r_elbow", "l_wrist", "r_wrist", "nose")):
            return

        posed = self.rig.pose(j)
        if posed is not None:
            verts, normals = posed
            if self.body_scale != 1.0:
                centre = (j["l_shoulder"] + j["r_shoulder"]) / 2.0
                verts = (verts - centre) * self.body_scale + centre
            self.mesh.vertices = o3d.utility.Vector3dVector(verts)
            self.mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

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


def _make_hand() -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    ls.lines = o3d.utility.Vector2iVector(np.array(HAND_BONES, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(HAND_COLOR, (len(HAND_BONES), 1)))
    return ls


def make_mannequin(skin: str = DEFAULT_SKIN, *,
                   avatar: Optional[str | Path] = None,
                   body_scale: float = 1.0,
                   max_vertices: int = DEFAULT_MAX_VERTICES):
    """Build one figure of the requested skin.

    Raises `AvatarUnavailable` for `anime` with no usable file — callers are
    expected to catch that and fall back to `classic` rather than crash a
    recording session over a cosmetic setting.
    """
    if skin == "anime":
        return AnimeMannequin(avatar=avatar, body_scale=body_scale,
                              max_vertices=max_vertices)
    return Mannequin()
