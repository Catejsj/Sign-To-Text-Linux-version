"""Drive a rigged humanoid from the six body joints the capture gives us.

The capture produces `l/r_shoulder`, `l/r_elbow`, `l/r_wrist` and `nose` in
scene coordinates. A VRM has a hundred-odd bones. This module bridges the two
without an IK solver, because the problem is easier than general IK: we know
where every joint of the arm *is*, so each bone only has to be aimed.

The method
----------
For every bone we control, take the direction it points in the rest pose,
rotate it by whatever its parent has already accumulated, and then rotate it
again by the minimal rotation that lands it on the direction we want. Walk the
hierarchy parent-first so each bone inherits its parent's motion. Bones we do
not control — fingers, clothing bones, the torso — inherit and follow rigidly.

Why the torso is not driven: aligning the avatar to the scene is done by a
similarity transform built from the shoulder line, which already puts the
chest where the signer's chest is. Rotating the spine on top of that would
apply the same rotation twice.

The skinning matrix
-------------------
For a bone `b` with accumulated rotation `R`, rest world position `p_rest` and
new world position `p_new`, the matrix handed to linear blend skinning is

    T(p_new) @ R @ T(-p_rest) @ node_rest[b] @ inverse_bind[b]

which reads right-to-left as: undo the bind pose, place the bone at its rest
position, rotate about that position, then move it where it now belongs. At
rest this collapses to the identity, which `test_avatar_pose.py` asserts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.gltf_min import SkinnedMesh

# Which bone aims at which, and the scene joints giving the target direction.
# (bone, child bone, from-joint, to-joint)
_ARM_CHAIN = (
    ("leftupperarm",  "leftlowerarm", "l_shoulder", "l_elbow"),
    ("leftlowerarm",  "lefthand",     "l_elbow",    "l_wrist"),
    ("rightupperarm", "rightlowerarm", "r_shoulder", "r_elbow"),
    ("rightlowerarm", "righthand",    "r_elbow",    "r_wrist"),
)

# Bones tried in order when a rig omits the one we asked for. VRM makes
# `upperchest` optional, and some rigs have no separate `chest`.
_TORSO_FALLBACK = {
    "chest": ("chest", "upperchest", "spine"),
    "neck": ("neck", "head"),
}


def rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector `a` onto unit vector `b`."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Exactly opposed: spin a half turn about any perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis) + 1e-12
        K = _skew(axis)
        return np.eye(3) + 2.0 * (K @ K)
    K = _skew(v / s)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _basis(side: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns: side, up, forward) from two rough axes."""
    s = side / (np.linalg.norm(side) + 1e-12)
    u = up - s * float(np.dot(up, s))          # Gram-Schmidt
    u /= np.linalg.norm(u) + 1e-12
    f = np.cross(s, u)
    f /= np.linalg.norm(f) + 1e-12
    return np.column_stack([s, u, f])


class AvatarRig:
    """A loaded humanoid, ready to be posed frame after frame.

    Build once — loading and colour-baking a VRM takes a second or two — then
    call `pose()` per frame. Nothing here imports Open3D; the caller decides
    what to do with the vertex array that comes back.
    """

    def __init__(self, mesh: SkinnedMesh, bones: dict[str, int],
                 hand_scale: float = 0.0):
        self.mesh = mesh
        self.bones = bones
        self.hand_scale = float(hand_scale)

        missing = [b for b in ("leftupperarm", "rightupperarm",
                               "leftlowerarm", "rightlowerarm",
                               "lefthand", "righthand") if b not in bones]
        if missing:
            raise ValueError(
                "the rig is missing arm bones: " + ", ".join(missing) +
                ". Without them there is nothing to drive."
            )

        self.node_index = {j: int(n)
                           for j, n in enumerate(mesh.joint_nodes)}
        self.node_to_joint = {int(n): j
                              for j, n in enumerate(mesh.joint_nodes)}
        self._order = self._topological_order()
        self._rest_pos = mesh.node_rest[:, :3, 3].copy()
        self._collapse = self._hand_descendants() if self.hand_scale < 1.0 \
            else set()

        # The avatar's own rest frame, used to map scene space onto it.
        l_sh = self._rest_pos[bones["leftupperarm"]]
        r_sh = self._rest_pos[bones["rightupperarm"]]
        self.rest_shoulder_width = float(np.linalg.norm(l_sh - r_sh))
        if self.rest_shoulder_width < 1e-9:
            raise ValueError("the rig's shoulders sit on top of each other")
        self.rest_neck = (l_sh + r_sh) / 2.0

        hips = bones.get("hips")
        up = (self.rest_neck - self._rest_pos[hips]) if hips is not None \
            else np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, 1.0, 0.0])
        self._avatar_basis = _basis(l_sh - r_sh, up)

    # ── setup helpers ────────────────────────────────────────────────
    def _topological_order(self) -> list[int]:
        """Node indices with every parent before its children."""
        par = self.mesh.node_parents
        depth = np.zeros(len(par), dtype=np.int64)
        for i in range(len(par)):
            d, j, guard = 0, int(par[i]), 0
            while j != -1 and guard < len(par):
                d += 1
                j = int(par[j])
                guard += 1
            depth[i] = d
        return list(np.argsort(depth, kind="stable"))

    def _hand_descendants(self) -> set[int]:
        """Every node at or below a hand bone.

        These get scaled by `hand_scale`, which defaults to 0 so the model's
        own hands vanish and the MediaPipe landmark rig stands in for them —
        the avatar's fingers cannot be driven from our data, and a frozen
        open palm during fingerspelling reads worse than no palm at all.
        """
        roots = [self.bones[b] for b in ("lefthand", "righthand")
                 if b in self.bones]
        if not roots:
            return set()
        children: dict[int, list[int]] = {}
        for i, p in enumerate(self.mesh.node_parents):
            children.setdefault(int(p), []).append(i)
        out: set[int] = set()
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(children.get(n, []))
        return out

    def _bone(self, name: str) -> Optional[int]:
        for candidate in _TORSO_FALLBACK.get(name, (name,)):
            if candidate in self.bones:
                return self.bones[candidate]
        return None

    # ── per-frame ────────────────────────────────────────────────────
    def scene_transform(self, joints: dict[str, np.ndarray]
                        ) -> Optional[tuple[np.ndarray, float, np.ndarray,
                                            np.ndarray]]:
        """Similarity mapping avatar space ↔ scene space for this frame.

        Returns `(rotation, scale, avatar_origin, scene_origin)` such that
        `scene = rotation @ (avatar - avatar_origin) * scale + scene_origin`.
        Derived from the shoulder line and the torso up-axis in both spaces,
        so handedness and which way the model faces are resolved from the
        geometry rather than assumed — a VRM 0.x model faces the opposite way
        from a VRM 1.0 one, and this does not have to know which it has.
        """
        if not all(k in joints for k in ("l_shoulder", "r_shoulder", "nose")):
            return None
        l_sh, r_sh = joints["l_shoulder"], joints["r_shoulder"]
        neck = (l_sh + r_sh) / 2.0
        width = float(np.linalg.norm(l_sh - r_sh))
        if width < 1e-9:
            return None

        up = neck - joints["nose"]        # nose is above the neck, so negate
        up = -up
        if np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, 1.0, 0.0])
        scene_basis = _basis(l_sh - r_sh, up)

        rotation = scene_basis @ self._avatar_basis.T
        scale = width / self.rest_shoulder_width
        return rotation, scale, self.rest_neck, neck

    def pose(self, joints: dict[str, np.ndarray],
             with_normals: bool = True
             ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Skin the avatar to `joints`. Returns scene-space (verts, normals).

        `None` means the frame did not carry enough of the upper body to place
        the avatar; the caller should hold the previous pose.
        """
        frame = self.scene_transform(joints)
        if frame is None:
            return None
        rotation, scale, avatar_origin, scene_origin = frame

        # Bring the scene targets into avatar space — the rig's own units —
        # so the aiming math never mixes coordinate systems.
        inv_rot = rotation.T
        def to_avatar(p: np.ndarray) -> np.ndarray:
            return inv_rot @ (p - scene_origin) / scale + avatar_origin

        targets = {k: to_avatar(v) for k, v in joints.items()}

        accum = np.tile(np.eye(3), (len(self._rest_pos), 1, 1))
        new_pos = self._rest_pos.copy()
        aims = self._aim_directions(targets)

        par = self.mesh.node_parents
        for node in self._order:
            p = int(par[node])
            if p != -1:
                accum[node] = accum[p]
                new_pos[node] = new_pos[p] + accum[p] @ (
                    self._rest_pos[node] - self._rest_pos[p])
            if node in aims:
                child, direction = aims[node]
                rest_dir = self._rest_pos[child] - self._rest_pos[node]
                if np.linalg.norm(rest_dir) > 1e-9:
                    current = accum[node] @ (
                        rest_dir / np.linalg.norm(rest_dir))
                    accum[node] = rot_from_to(current, direction) @ accum[node]
            if node in self._collapse:
                accum[node] = accum[node] * self.hand_scale

        skin_mats = self._skin_matrices(accum, new_pos)
        verts, normals = self.mesh.skin(skin_mats, with_normals=with_normals)

        verts = (verts - avatar_origin) * scale @ rotation.T + scene_origin
        if with_normals:
            normals = normals @ rotation.T
        return verts, normals

    def _aim_directions(self, targets: dict[str, np.ndarray]
                        ) -> dict[int, tuple[int, np.ndarray]]:
        """`{node: (child node, desired unit direction)}` for this frame."""
        out: dict[int, tuple[int, np.ndarray]] = {}
        for bone, child_bone, a, b in _ARM_CHAIN:
            node, child = self._bone(bone), self._bone(child_bone)
            if node is None or child is None:
                continue
            if a not in targets or b not in targets:
                continue
            d = targets[b] - targets[a]
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                out[node] = (child, d / n)

        # Head: aim neck→head at the nose, which is the only facial point we
        # track. It is a small effect but it stops the head looking pasted on.
        neck, head = self._bone("neck"), self._bone("head")
        if neck is not None and head is not None and "nose" in targets:
            if "l_shoulder" in targets and "r_shoulder" in targets:
                mid = (targets["l_shoulder"] + targets["r_shoulder"]) / 2.0
                d = targets["nose"] - mid
                n = float(np.linalg.norm(d))
                if n > 1e-9:
                    out[neck] = (head, d / n)
        return out

    def _skin_matrices(self, accum: np.ndarray,
                       new_pos: np.ndarray) -> np.ndarray:
        """(J, 4, 4) for the skin's joints, in the skinning frame."""
        nodes = self.mesh.joint_nodes
        r = accum[nodes]                       # (J, 3, 3)
        p_rest = self._rest_pos[nodes]         # (J, 3)
        p_new = new_pos[nodes]

        m = np.tile(np.eye(4), (len(nodes), 1, 1))
        m[:, :3, :3] = r
        m[:, :3, 3] = p_new - np.einsum("jik,jk->ji", r, p_rest)
        return m @ self.mesh.node_rest[nodes] @ self.mesh.inverse_bind


def load_rig(path, max_vertices: int = 24000,
             hand_scale: float = 0.0) -> AvatarRig:
    """Load a `.vrm` / `.glb` / `.gltf` and prepare it for posing.

    Falls back to name-matching when the file carries no VRM humanoid table,
    which is what lets a plain Mixamo or Blender export work too.
    """
    from src.gltf_min import Gltf

    gltf = Gltf.load(path)
    bones = gltf.humanoid_bones() or gltf.guess_humanoid_bones()
    mesh = gltf.skinned_mesh(max_vertices=max_vertices)
    return AvatarRig(mesh, bones, hand_scale=hand_scale)
