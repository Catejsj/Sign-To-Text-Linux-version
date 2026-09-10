"""A minimal glTF 2.0 / GLB / VRM reader, in numpy and nothing else.

Why this exists instead of `pip install pygltflib trimesh`
----------------------------------------------------------
The project adds dependencies reluctantly, and everything here needs is a
narrow slice of glTF: positions, normals, UVs, skin weights, the node
hierarchy, and the VRM humanoid bone map. That is a few hundred lines of
mechanical binary parsing. A full glTF library brings a scene graph, a
material system, exporters and an image stack we would never call.

What is supported
-----------------
* `.glb` / `.vrm` (binary container) and `.gltf` (JSON + external `.bin`)
* accessors: every component type, every element type, `byteStride`
  interleaving, `normalized` integers, and sparse storage
* buffers from a GLB BIN chunk, a `data:` URI, or a file beside the `.gltf`
* skins: joint node lists and inverse bind matrices
* VRM 0.x (`extensions.VRM`) and VRM 1.0 (`extensions.VRMC_vrm`) humanoid
  bone maps, normalised to one lowercase naming scheme

What is not
-----------
* Draco / meshopt compression — detected and reported, not decoded. Re-export
  the model with compression off.
* animations, cameras, morph targets, PBR beyond base colour.

Nothing in here touches Open3D, so it is testable without a display.
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import numpy as np

# ── glTF constants ───────────────────────────────────────────────────
_GLB_MAGIC = 0x46546C67          # 'glTF'
_CHUNK_JSON = 0x4E4F534A         # 'JSON'
_CHUNK_BIN = 0x004E4942          # 'BIN\0'

_COMPONENT = {
    5120: (np.int8, 1), 5121: (np.uint8, 1),
    5122: (np.int16, 2), 5123: (np.uint16, 2),
    5125: (np.uint32, 4), 5126: (np.float32, 4),
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
          "MAT2": 4, "MAT3": 9, "MAT4": 16}

# Integer accessors flagged `normalized` map onto 0..1 by their type max.
_NORM_MAX = {np.int8: 127.0, np.uint8: 255.0,
             np.int16: 32767.0, np.uint16: 65535.0}

_UNSUPPORTED_EXT = ("KHR_draco_mesh_compression", "EXT_meshopt_compression")


def is_outline_material(name: str) -> bool:
    """Is this material an inverted-hull outline shell?

    Toon models ship a second copy of the body, slightly inflated and painted
    near-black, drawn with FRONT faces culled so only its inside surface
    shows — that dark rim is the cartoon outline. Open3D does not cull that
    way, so the shell renders as an opaque black skin over the character; at
    full detail the model disappears entirely inside it.

    It is a rendering trick, not geometry, so we drop it. On a VRoid-style
    export this is typically half the vertices, which is also half the
    skinning cost per frame for something that should never have been drawn.

    Matching on `_Line` and `outline` covers the VRoid / MToon / Blender toon
    conventions. The `_line` test requires the underscore so that a material
    legitimately called `eyeliner` survives.
    """
    n = name.lower()
    return n.endswith("_line") or "outline" in n


class GltfError(RuntimeError):
    """The file is not glTF we can read. The message says what to do."""


# ── the humanoid bone vocabulary ─────────────────────────────────────
# VRM 0.x spells bones in camelCase ("leftUpperArm"); VRM 1.0 uses the same
# names as object keys. We lowercase both so callers match on one spelling.
HUMANOID_BONES = (
    "hips", "spine", "chest", "upperchest", "neck", "head",
    "leftshoulder", "leftupperarm", "leftlowerarm", "lefthand",
    "rightshoulder", "rightupperarm", "rightlowerarm", "righthand",
    "leftupperleg", "leftlowerleg", "leftfoot",
    "rightupperleg", "rightlowerleg", "rightfoot",
)


def _quat_to_mat(q) -> np.ndarray:
    """glTF quaternion [x, y, z, w] → 3x3 rotation."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


class Gltf:
    """A parsed glTF document. Construct with `Gltf.load(path)`."""

    def __init__(self, doc: dict, buffers: list[bytes], base: Path):
        self.doc = doc
        self.buffers = buffers
        self.base = base
        self._accessor_cache: dict[int, np.ndarray] = {}
        self._image_cache: dict[int, Any] = {}
        self.outline_prims = 0     # how many outline shells were skipped
        self.stray_prims = 0       # primitives too far from the skeleton

    # ── loading ──────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "Gltf":
        path = Path(path)
        if not path.exists():
            raise GltfError(f"no such file: {path}")
        raw = path.read_bytes()

        if raw[:4] == b"glTF":
            doc, bin_chunk = cls._parse_glb(raw, path)
        else:
            try:
                doc = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GltfError(
                    f"{path.name} is neither a GLB container nor valid glTF "
                    f"JSON ({exc})"
                ) from exc
            bin_chunk = None

        used = set(doc.get("extensionsRequired", []) or [])
        used |= set(doc.get("extensionsUsed", []) or [])
        blocked = [e for e in _UNSUPPORTED_EXT if e in used]
        if blocked:
            raise GltfError(
                f"{path.name} uses {', '.join(blocked)}, which this reader "
                f"does not decode. Re-export the model with mesh compression "
                f"turned off."
            )

        buffers = [cls._resolve_buffer(b, i, bin_chunk, path.parent)
                   for i, b in enumerate(doc.get("buffers", []))]
        return cls(doc, buffers, path.parent)

    @staticmethod
    def _parse_glb(raw: bytes, path: Path) -> tuple[dict, Optional[bytes]]:
        if len(raw) < 12:
            raise GltfError(f"{path.name} is too short to be a GLB")
        magic, version, _length = struct.unpack_from("<III", raw, 0)
        if magic != _GLB_MAGIC:
            raise GltfError(f"{path.name}: bad GLB magic")
        if version != 2:
            raise GltfError(
                f"{path.name} is glTF version {version}; only 2 is supported"
            )
        doc: Optional[dict] = None
        bin_chunk: Optional[bytes] = None
        off = 12
        while off + 8 <= len(raw):
            clen, ctype = struct.unpack_from("<II", raw, off)
            body = raw[off + 8: off + 8 + clen]
            if ctype == _CHUNK_JSON and doc is None:
                doc = json.loads(body.decode("utf-8"))
            elif ctype == _CHUNK_BIN and bin_chunk is None:
                bin_chunk = body
            off += 8 + clen + (-clen % 4)   # chunks are 4-byte aligned
        if doc is None:
            raise GltfError(f"{path.name}: GLB has no JSON chunk")
        return doc, bin_chunk

    @staticmethod
    def _resolve_buffer(spec: dict, index: int, bin_chunk: Optional[bytes],
                        folder: Path) -> bytes:
        uri = spec.get("uri")
        if uri is None:
            if bin_chunk is None:
                raise GltfError(
                    f"buffer {index} expects the GLB binary chunk, but the "
                    f"file has none"
                )
            return bin_chunk
        if uri.startswith("data:"):
            head, _, payload = uri.partition(",")
            if "base64" not in head:
                raise GltfError(f"buffer {index}: only base64 data URIs read")
            return base64.b64decode(payload)
        side = folder / unquote(uri)
        if not side.exists():
            raise GltfError(
                f"buffer {index} points at '{uri}', which is missing next to "
                f"the .gltf. A single-file .glb or .vrm avoids this."
            )
        return side.read_bytes()

    # ── accessors ────────────────────────────────────────────────────
    def accessor(self, index: int) -> np.ndarray:
        """Accessor `index` as an (count, ncomp) array. Cached."""
        if index in self._accessor_cache:
            return self._accessor_cache[index]
        acc = self.doc["accessors"][index]
        dtype, size = _COMPONENT[acc["componentType"]]
        ncomp = _NCOMP[acc["type"]]
        count = acc["count"]
        elem = size * ncomp

        if "bufferView" in acc:
            out = self._read_view(acc["bufferView"], acc.get("byteOffset", 0),
                                  count, ncomp, dtype, elem)
        else:
            # No bufferView: the accessor is all zeros, then sparse patches it.
            out = np.zeros((count, ncomp), dtype=dtype)

        if "sparse" in acc:
            out = self._apply_sparse(out.copy(), acc["sparse"], ncomp, dtype)

        if acc.get("normalized") and dtype in _NORM_MAX:
            out = np.clip(out.astype(np.float32) / _NORM_MAX[dtype], -1.0, 1.0)

        self._accessor_cache[index] = out
        return out

    def _read_view(self, view_index: int, offset: int, count: int,
                   ncomp: int, dtype, elem: int) -> np.ndarray:
        view = self.doc["bufferViews"][view_index]
        buf = self.buffers[view["buffer"]]
        start = view.get("byteOffset", 0) + offset
        stride = view.get("byteStride") or elem

        if stride == elem:
            need = count * elem
            chunk = buf[start:start + need]
            if len(chunk) < need:
                raise GltfError(
                    f"bufferView {view_index} is truncated: wanted {need} "
                    f"bytes at {start}, file has {len(chunk)}"
                )
            return np.frombuffer(chunk, dtype=dtype).reshape(count, ncomp)

        # Interleaved: gather each element out of the stride grid.
        need = (count - 1) * stride + elem
        flat = np.frombuffer(buf[start:start + need], dtype=np.uint8)
        if flat.size < need:
            raise GltfError(
                f"bufferView {view_index} is truncated: wanted {need} "
                f"interleaved bytes at {start}, file has {flat.size}"
            )
        idx = (np.arange(count) * stride)[:, None] + np.arange(elem)[None, :]
        return flat[idx].copy().view(dtype).reshape(count, ncomp)

    def _apply_sparse(self, base: np.ndarray, sparse: dict, ncomp: int,
                      dtype) -> np.ndarray:
        n = sparse["count"]
        if n == 0:
            return base
        idx_spec = sparse["indices"]
        i_dtype, i_size = _COMPONENT[idx_spec["componentType"]]
        indices = self._read_view(idx_spec["bufferView"],
                                  idx_spec.get("byteOffset", 0),
                                  n, 1, i_dtype, i_size).reshape(-1)
        # Sparse values share the accessor's component type, so one element
        # is itemsize * ncomp bytes.
        val_spec = sparse["values"]
        elem = np.dtype(dtype).itemsize * ncomp
        values = self._read_view(val_spec["bufferView"],
                                 val_spec.get("byteOffset", 0),
                                 n, ncomp, dtype, elem)
        base[indices.astype(np.int64)] = values
        return base

    # ── node hierarchy ───────────────────────────────────────────────
    def local_matrix(self, node_index: int) -> np.ndarray:
        node = self.doc["nodes"][node_index]
        if "matrix" in node:
            # glTF stores matrices column-major.
            return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        r = _quat_to_mat(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
        s = np.diag(np.array(node.get("scale", (1.0, 1.0, 1.0)),
                             dtype=np.float64))
        m[:3, :3] = r @ s
        m[:3, 3] = np.array(node.get("translation", (0.0, 0.0, 0.0)),
                            dtype=np.float64)
        return m

    def parents(self) -> np.ndarray:
        """`parents[i]` = parent node index of node i, or -1 for a root."""
        nodes = self.doc.get("nodes", [])
        out = np.full(len(nodes), -1, dtype=np.int64)
        for i, node in enumerate(nodes):
            for c in node.get("children", []) or []:
                out[c] = i
        return out

    def global_matrices(self) -> np.ndarray:
        """(N, 4, 4) world matrix per node, in the file's rest pose."""
        nodes = self.doc.get("nodes", [])
        par = self.parents()
        out = np.zeros((len(nodes), 4, 4))
        done = np.zeros(len(nodes), dtype=bool)

        def resolve(i: int) -> np.ndarray:
            # Iterative, so a deep rig cannot blow the Python stack.
            chain = []
            j = i
            while j != -1 and not done[j]:
                chain.append(j)
                j = int(par[j])
            acc = out[j] if j != -1 else np.eye(4)
            for k in reversed(chain):
                acc = acc @ self.local_matrix(k)
                out[k] = acc
                done[k] = True
            return out[i]

        for i in range(len(nodes)):
            resolve(i)
        return out

    # ── VRM humanoid map ─────────────────────────────────────────────
    def humanoid_bones(self) -> dict[str, int]:
        """`{lowercase bone name: node index}`. Empty if the file is not VRM.

        Reads VRM 1.0 first, then VRM 0.x. Both describe the same skeleton;
        they disagree only on where the table lives and how it is shaped.
        """
        ext = self.doc.get("extensions", {}) or {}
        out: dict[str, int] = {}

        vrm1 = ext.get("VRMC_vrm", {}).get("humanoid", {}).get("humanBones")
        if isinstance(vrm1, dict):
            for name, spec in vrm1.items():
                if isinstance(spec, dict) and isinstance(spec.get("node"), int):
                    out[name.lower()] = spec["node"]
            if out:
                return out

        vrm0 = ext.get("VRM", {}).get("humanoid", {}).get("humanBones")
        if isinstance(vrm0, list):
            for spec in vrm0:
                if not isinstance(spec, dict):
                    continue
                name, node = spec.get("bone"), spec.get("node")
                if isinstance(name, str) and isinstance(node, int):
                    out[name.lower()] = node
        return out

    def guess_humanoid_bones(self) -> dict[str, int]:
        """Fallback for plain glTF: match humanoid bones by node name.

        Mixamo, Blender and VRoid exports all name bones recognisably
        ("mixamorig:LeftForeArm", "upper_arm.L", "J_Bip_L_UpperArm"). This
        strips punctuation and looks for the distinguishing substrings. It is
        a guess — `check_avatar.py` prints what it matched so you can see.
        """
        aliases = {
            "hips": ("hips", "pelvis"),
            "spine": ("spine",),
            "chest": ("chest", "spine1", "spine2"),
            "neck": ("neck",),
            "head": ("head",),
            "leftshoulder": ("leftshoulder", "shoulderl", "clavicle_l"),
            "rightshoulder": ("rightshoulder", "shoulderr", "clavicle_r"),
            "leftupperarm": ("leftarm", "leftupperarm", "upperarml", "upper_arml"),
            "rightupperarm": ("rightarm", "rightupperarm", "upperarmr", "upper_armr"),
            "leftlowerarm": ("leftforearm", "leftlowerarm", "forearml", "lower_arml"),
            "rightlowerarm": ("rightforearm", "rightlowerarm", "forearmr", "lower_armr"),
            # "wrist" is as common as "hand" for this bone — Blender and
            # several VRChat rigs name it that way, with the fingers hanging
            # off it exactly as they would off a hand.
            "lefthand": ("lefthand", "handl", "hand_l", "wristl", "wrist_l"),
            "righthand": ("righthand", "handr", "hand_r", "wristr", "wrist_r"),
        }

        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        named = [(norm(n.get("name", "")), i)
                 for i, n in enumerate(self.doc.get("nodes", []))]
        out: dict[str, int] = {}
        for bone, keys in aliases.items():
            for key in keys:
                hits = [i for name, i in named if norm(key) in name]
                if hits:
                    # Shortest name wins: "LeftArm" over "LeftArmTwist01".
                    hits.sort(key=lambda i: len(
                        self.doc["nodes"][i].get("name", "")))
                    out[bone] = hits[0]
                    break
        return out

    # ── geometry ─────────────────────────────────────────────────────
    def skinned_mesh(self, max_vertices: int = 0) -> "SkinnedMesh":
        """Flatten every skinned primitive in the file into one mesh.

        A VRM is typically split into a handful of primitives (body, face,
        hair, clothes) that share one skin. We concatenate them, re-basing
        each primitive's triangle indices, so the caller deals with a single
        vertex array — which is what a renderer wants anyway.
        """
        doc = self.doc
        if not doc.get("skins"):
            raise GltfError(
                "this file has no skin, so it cannot be posed. Export the "
                "model with its armature (a VRM always has one)."
            )
        globals_ = self.global_matrices()

        pos_l, nrm_l, uv_l, tri_l = [], [], [], []
        jnt_l, wgt_l, col_l = [], [], []
        base = 0
        skin_index: Optional[int] = None

        # Where the skeleton actually is, for the sanity check below. Taken
        # from the first skin, which is the one we use.
        first_skin = doc["skins"][0]
        bone_pos = globals_[np.array(first_skin["joints"], dtype=np.int64)][:, :3, 3]
        bone_centre = (bone_pos.min(axis=0) + bone_pos.max(axis=0)) / 2.0
        bone_reach = float(np.linalg.norm(
            bone_pos.max(axis=0) - bone_pos.min(axis=0))) + 1e-9

        for node_index, node in enumerate(doc.get("nodes", [])):
            if "mesh" not in node or "skin" not in node:
                continue
            if skin_index is None:
                skin_index = node["skin"]
            elif node["skin"] != skin_index:
                continue   # a second armature; the first one wins
            node_world = globals_[node_index]

            for prim in doc["meshes"][node["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4:
                    continue   # not triangles
                attrs = prim.get("attributes", {})
                if "POSITION" not in attrs or "JOINTS_0" not in attrs:
                    continue
                if "material" in prim and is_outline_material(
                        doc.get("materials", [])[prim["material"]]
                        .get("name", "")):
                    self.outline_prims += 1
                    continue

                # A skinned primitive's POSITION is already in the skinning
                # frame — glTF says the mesh node's own transform is ignored
                # for skinned meshes, because the inverse bind matrices
                # already account for it. Baking `node_world` in here would
                # apply it twice.
                pos = self.accessor(attrs["POSITION"]).astype(np.float64)
                n = len(pos)
                nrm = (self.accessor(attrs["NORMAL"]).astype(np.float64)
                       if "NORMAL" in attrs else np.zeros((n, 3)))

                uv = (self.accessor(attrs["TEXCOORD_0"]).astype(np.float64)
                      if "TEXCOORD_0" in attrs else np.zeros((n, 2)))
                jnt = self.accessor(attrs["JOINTS_0"]).astype(np.int64)
                wgt = self.accessor(attrs["WEIGHTS_0"]).astype(np.float64)

                # Geometry that cannot belong to this skeleton.
                #
                # Some exports carry primitives whose POSITION is hundreds of
                # units from every bone that drives them — bound to ordinary
                # joints, but sitting nowhere near them, so at rest they fly
                # off into space. One VRChat export puts three coat meshes
                # 450 units below a figure 12 units tall. Their mesh node
                # transform does not account for it either; the geometry is
                # simply wrong in the file.
                #
                # Keeping them wrecks the whole view: they stretch the scene
                # bounds by a factor of forty, which throws off framing and
                # the decimation grid alike. The test is deliberately loose,
                # at twice the skeleton's own diagonal, so it can only ever
                # catch geometry that is unambiguously astray.
                centre = np.median(pos, axis=0)
                if np.linalg.norm(centre - bone_centre) > 2.0 * bone_reach:
                    self.stray_prims += 1
                    continue

                if "indices" in prim:
                    tri = self.accessor(prim["indices"]).reshape(-1)
                else:
                    tri = np.arange(n)
                tri = tri.astype(np.int64).reshape(-1, 3) + base

                pos_l.append(pos)
                nrm_l.append(nrm)
                uv_l.append(uv)
                jnt_l.append(jnt)
                wgt_l.append(wgt)
                tri_l.append(tri)
                col_l.append(self._primitive_colors(prim, uv, n))
                base += n

        if not pos_l:
            raise GltfError(
                "found a skin but no skinned triangle geometry. The model may "
                "use compression, or store the body as a non-skinned mesh."
            )

        skin = doc["skins"][skin_index]
        joint_nodes = np.array(skin["joints"], dtype=np.int64)

        # Inverse bind matrices are optional; the spec's default is identity.
        if isinstance(skin.get("inverseBindMatrices"), int):
            ibm = self.accessor(skin["inverseBindMatrices"]).astype(np.float64)
            # glTF matrices are column-major, so transpose each one.
            inverse_bind = ibm.reshape(-1, 4, 4).transpose(0, 2, 1).copy()
        else:
            inverse_bind = np.tile(np.eye(4), (len(joint_nodes), 1, 1))

        mesh = SkinnedMesh(
            positions=np.concatenate(pos_l),
            normals=np.concatenate(nrm_l),
            uvs=np.concatenate(uv_l),
            colors=np.concatenate(col_l),
            triangles=np.concatenate(tri_l),
            joints=np.concatenate(jnt_l),
            weights=np.concatenate(wgt_l),
            joint_nodes=joint_nodes,
            joint_rest=globals_[joint_nodes][:, :3, 3].copy(),
            inverse_bind=inverse_bind,
            node_parents=self.parents(),
            node_rest=globals_,
        )
        mesh.normalise_weights()
        if max_vertices:
            mesh.decimate(max_vertices)
        return mesh

    def _primitive_colors(self, prim: dict, uv: np.ndarray,
                          n: int) -> np.ndarray:
        """Per-vertex RGB for one primitive.

        Open3D's legacy Visualizer will not texture a mesh whose vertices we
        rewrite every frame, so we bake the look into vertex colours instead:
        sample the base-colour texture once at each vertex's UV and multiply
        by the material's base-colour factor. For an anime model — flat cel
        shading, large uniform regions — this reproduces hair, skin, eyes and
        clothing colours closely. It loses only fine texture detail.
        """
        attrs = prim.get("attributes", {})
        if "COLOR_0" in attrs:
            col = self.accessor(attrs["COLOR_0"]).astype(np.float64)
            if col.max() > 1.5:      # unnormalised ints
                col = col / 255.0
            return np.clip(col[:, :3], 0.0, 1.0)

        if "material" not in prim:
            return np.tile(np.array([0.8, 0.8, 0.8]), (n, 1))

        mat = self.doc.get("materials", [])[prim["material"]]
        pbr = mat.get("pbrMetallicRoughness", {}) or {}
        base = self._sample(
            pbr.get("baseColorTexture"),
            np.array(pbr.get("baseColorFactor", (0.8, 0.8, 0.8))[:3],
                     dtype=np.float64),
            uv, n)

        # Toon and unlit models — this includes every MToon/VRM export and
        # most anime characters — set the base colour to pure black and put
        # the artwork in the emissive channel, so a lit renderer cannot
        # darken it. Read as base colour we would get a black silhouette.
        if base.max() < 0.02:
            emissive = self._sample(
                mat.get("emissiveTexture"),
                np.array(mat.get("emissiveFactor", (0.0, 0.0, 0.0))[:3],
                         dtype=np.float64),
                uv, n)
            if emissive.max() >= 0.02:
                base = emissive

        return np.clip(base, 0.0, 1.0)

    def _sample(self, tex_spec: Optional[dict], factor: np.ndarray,
                uv: np.ndarray, n: int) -> np.ndarray:
        """`factor`, modulated by a texture point-sampled at each vertex UV."""
        source: Optional[int] = None
        if isinstance((tex_spec or {}).get("index"), int):
            texture = self.doc.get("textures", [])[tex_spec["index"]]
            if isinstance(texture.get("source"), int):
                source = texture["source"]

        img = self._image(source) if source is not None else None
        if img is None:
            return np.tile(factor, (n, 1))

        h, w = img.shape[:2]
        # glTF UV origin is top-left; wrap so out-of-range UVs still land.
        u = np.clip((np.mod(uv[:, 0], 1.0) * w).astype(np.int64), 0, w - 1)
        v = np.clip((np.mod(uv[:, 1], 1.0) * h).astype(np.int64), 0, h - 1)
        return img[v, u] * factor

    def _image(self, index: int):
        """Decode image `index` to an (h, w, 3) float array, or None."""
        if index in self._image_cache:
            return self._image_cache[index]
        result = None
        try:
            from PIL import Image
            import io

            spec = self.doc["images"][index]
            if "bufferView" in spec:
                view = self.doc["bufferViews"][spec["bufferView"]]
                buf = self.buffers[view["buffer"]]
                off = view.get("byteOffset", 0)
                data = buf[off: off + view["byteLength"]]
            elif "uri" in spec and spec["uri"].startswith("data:"):
                data = base64.b64decode(spec["uri"].partition(",")[2])
            elif "uri" in spec:
                side = self.base / unquote(spec["uri"])
                data = side.read_bytes() if side.exists() else None
            else:
                data = None

            if data:
                with Image.open(io.BytesIO(data)) as im:
                    im = im.convert("RGB")
                    # Downscale big atlases: we only point-sample them.
                    if max(im.size) > 1024:
                        im.thumbnail((1024, 1024))
                    result = np.asarray(im, dtype=np.float64) / 255.0
        except Exception:
            # A texture we cannot decode is cosmetic — fall back to the
            # material's flat colour rather than failing the whole load.
            result = None
        self._image_cache[index] = result
        return result


class SkinnedMesh:
    """Rest-pose geometry plus the skin binding needed to re-pose it."""

    def __init__(self, positions, normals, uvs, colors, triangles,
                 joints, weights, joint_nodes, joint_rest, inverse_bind,
                 node_parents, node_rest):
        self.positions = positions        # (V, 3) rest, skinning space
        self.normals = normals            # (V, 3)
        self.uvs = uvs                    # (V, 2)
        self.colors = colors              # (V, 3) in 0..1
        self.triangles = triangles        # (F, 3)
        self.joints = joints              # (V, 4) indices into joint_nodes
        self.weights = weights            # (V, 4)
        self.joint_nodes = joint_nodes    # (J,) node index per skin joint
        self.joint_rest = joint_rest      # (J, 3) rest world position
        self.inverse_bind = inverse_bind  # (J, 4, 4) row-major
        self.node_parents = node_parents  # (N,) parent node index or -1
        self.node_rest = node_rest        # (N, 4, 4) rest world matrices

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    def normalise_weights(self) -> None:
        s = self.weights.sum(axis=1, keepdims=True)
        s[s < 1e-9] = 1.0
        self.weights = self.weights / s

    def decimate(self, max_vertices: int) -> None:
        """Merge vertices onto a uniform grid until the budget is met.

        Skinning cost is linear in vertices and we re-skin every frame, so a
        200k-vertex character is worth thinning for a live preview.

        Vertex *clustering*, not triangle dropping. Dropping every n-th
        triangle barely helps: the survivors are spread across the whole mesh
        and still reference nearly every vertex. Snapping to a grid and
        merging each cell to one vertex reduces the count by construction,
        and collapses the triangles that become degenerate.

        The grid resolution is found by bisection — the finest grid whose
        occupied-cell count still fits the budget, so we keep as much detail
        as the budget allows. Attributes are averaged within a cell; skin
        bindings are taken from one representative vertex, since averaging
        joint *indices* is meaningless.
        """
        if self.n_vertices <= max_vertices or len(self.triangles) == 0:
            return

        p = self.positions
        lo = p.min(axis=0)
        extent = np.maximum(p.max(axis=0) - lo, 1e-9)

        best: Optional[np.ndarray] = None
        low, high = 2, 1024
        while low <= high:
            n = (low + high) // 2
            key = np.minimum(((p - lo) / extent * n).astype(np.int64), n - 1)
            flat = (key[:, 0] * n + key[:, 1]) * n + key[:, 2]
            _, inverse = np.unique(flat, return_inverse=True)
            if inverse.max() + 1 <= max_vertices:
                best = inverse
                low = n + 1
            else:
                high = n - 1
        if best is None:
            return                      # even a 2^3 grid overflows: give up

        inverse = best
        count = int(inverse.max()) + 1
        occupancy = np.bincount(inverse, minlength=count).astype(np.float64)

        def cluster_mean(arr: np.ndarray) -> np.ndarray:
            out = np.empty((count, arr.shape[1]))
            for c in range(arr.shape[1]):
                out[:, c] = np.bincount(inverse, weights=arr[:, c],
                                        minlength=count)
            return out / occupancy[:, None]

        # One representative vertex per cell, for the attributes that cannot
        # be averaged. Writing indices in reverse leaves the lowest per cell.
        representative = np.empty(count, dtype=np.int64)
        order = np.arange(len(inverse))[::-1]
        representative[inverse[::-1]] = order

        positions = cluster_mean(self.positions)
        normals = cluster_mean(self.normals)
        ln = np.linalg.norm(normals, axis=1, keepdims=True)
        ln[ln < 1e-9] = 1.0

        tri = inverse[self.triangles]
        keep = ((tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2])
                & (tri[:, 0] != tri[:, 2]))

        self.positions = positions
        self.normals = normals / ln
        self.uvs = cluster_mean(self.uvs)
        self.colors = cluster_mean(self.colors)
        self.joints = self.joints[representative]
        self.weights = self.weights[representative]
        self.triangles = tri[keep]

    def skin(self, skin_matrices: np.ndarray,
             with_normals: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Linear blend skinning. `skin_matrices` is (J, 4, 4).

        Returns `(positions, normals)`, both (V, 3). Normals are rotated by
        the same blend rather than recomputed from the triangles — a fraction
        of the cost, and accurate enough for a preview.
        """
        v = self.positions
        out = np.zeros_like(v)
        out_n = np.zeros_like(self.normals) if with_normals else None

        rot = skin_matrices[:, :3, :3]
        trans = skin_matrices[:, :3, 3]
        for i in range(self.joints.shape[1]):
            w = self.weights[:, i]
            active = w > 1e-6
            if not active.any():
                continue
            j = self.joints[active, i]
            wa = w[active][:, None]
            out[active] += wa * (
                np.einsum("vij,vj->vi", rot[j], v[active]) + trans[j])
            if with_normals:
                out_n[active] += wa * np.einsum(
                    "vij,vj->vi", rot[j], self.normals[active])

        if with_normals:
            ln = np.linalg.norm(out_n, axis=1, keepdims=True)
            ln[ln < 1e-9] = 1.0
            out_n = out_n / ln
        return out, out_n
