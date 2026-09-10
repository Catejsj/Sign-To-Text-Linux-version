#!/usr/bin/env python3
"""Inspect an avatar file, or self-test the retargeting math.

Two jobs, both of which run without a camera and without a display:

    python scripts/check_avatar.py --selftest
        Builds a small rigged figure in memory, poses it, and asserts the
        skinning is correct. Run this after touching src/gltf_min.py or
        src/avatar_pose.py — it catches a broken matrix convention in a
        second, where the live window would just look subtly wrong.

    python scripts/check_avatar.py assets/avatars/my_model.vrm
        Reports what the loader found in a real file: bone map, vertex count,
        whether the arms can be driven, and how fast a frame skins. Use this
        before wondering why the model looks odd in the live window.

Modelled on check_camera.py: say what is wrong in words, not a traceback.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.gltf_min import Gltf, GltfError, HUMANOID_BONES
    from src.avatar_pose import (
        AvatarRig, load_rig, load_rig_spec, rig_sidecar_path,
    )
except ImportError as exc:                                   # pragma: no cover
    print(f"cannot import the avatar modules: {exc}")
    print("Run this with the project's interpreter:  ./venv/bin/python "
          "scripts/check_avatar.py --selftest")
    sys.exit(2)


# ─────────────────────────────────────────────────────────────────────
#  A synthetic rig, built in memory, for the self-test
# ─────────────────────────────────────────────────────────────────────
# A T-posed stick figure. Positions are chosen so every bone has a
# different length — a bug that assumes uniform bones then shows up.
_REST = {
    "hips":          (0.00, 0.00, 0.0),
    "spine":         (0.00, 0.45, 0.0),
    "chest":         (0.00, 0.80, 0.0),
    "neck":          (0.00, 1.00, 0.0),
    "head":          (0.00, 1.18, 0.0),
    "leftupperarm":  (0.20, 1.00, 0.0),
    "leftlowerarm":  (0.55, 1.00, 0.0),
    "lefthand":      (0.83, 1.00, 0.0),
    "rightupperarm": (-0.20, 1.00, 0.0),
    "rightlowerarm": (-0.55, 1.00, 0.0),
    "righthand":     (-0.83, 1.00, 0.0),
}
_PARENT = {
    "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "leftupperarm": "chest", "leftlowerarm": "leftupperarm",
    "lefthand": "leftlowerarm",
    "rightupperarm": "chest", "rightlowerarm": "rightupperarm",
    "righthand": "rightlowerarm",
}


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def build_test_glb() -> bytes:
    """A valid single-file GLB holding the stick figure above.

    One vertex is pinned to each bone with weight 1, which makes the
    self-test's assertions exact: the posed vertex must land exactly where
    that bone lands.
    """
    names = list(_REST)
    index = {n: i for i, n in enumerate(names)}
    rest = {n: np.array(p, dtype=np.float64) for n, p in _REST.items()}

    nodes = []
    for n in names:
        parent = _PARENT.get(n)
        local = rest[n] - (rest[parent] if parent else np.zeros(3))
        node = {"name": n, "translation": [float(x) for x in local]}
        kids = [index[c] for c, p in _PARENT.items() if p == n]
        if kids:
            node["children"] = kids
        nodes.append(node)

    # Geometry: one vertex per bone, sitting exactly on it.
    positions = np.array([rest[n] for n in names], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], np.float32), (len(names), 1))
    joints = np.zeros((len(names), 4), dtype=np.uint16)
    joints[:, 0] = np.arange(len(names), dtype=np.uint16)
    weights = np.zeros((len(names), 4), dtype=np.float32)
    weights[:, 0] = 1.0
    # Triangles are only there to make it a legal mesh.
    tri = np.array([[i, (i + 1) % len(names), (i + 2) % len(names)]
                    for i in range(len(names))], dtype=np.uint32)

    inverse_bind = np.zeros((len(names), 4, 4), dtype=np.float32)
    for n in names:
        m = np.eye(4)
        m[:3, 3] = -rest[n]          # inverse of a pure translation
        inverse_bind[index[n]] = m.T  # glTF wants column-major

    blobs = [positions, normals, joints, weights, tri, inverse_bind]
    buffer = b""
    views, offset = [], 0
    for arr in blobs:
        raw = _pad4(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": offset,
                      "byteLength": int(arr.nbytes)})
        buffer += raw
        offset += len(raw)

    n = len(names)
    doc = {
        "asset": {"version": "2.0", "generator": "SignLink self-test"},
        "scene": 0,
        "scenes": [{"nodes": [index["hips"]]}],
        "nodes": nodes,
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n,
             "type": "VEC3",
             "min": positions.min(0).tolist(),
             "max": positions.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n,
             "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": n,
             "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": n,
             "type": "VEC4"},
            {"bufferView": 4, "componentType": 5125, "count": int(tri.size),
             "type": "SCALAR"},
            {"bufferView": 5, "componentType": 5126, "count": n,
             "type": "MAT4"},
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1,
                           "JOINTS_0": 2, "WEIGHTS_0": 3},
            "indices": 4, "mode": 4,
        }]}],
        "skins": [{"inverseBindMatrices": 5,
                   "joints": list(range(n)),
                   "skeleton": index["hips"]}],
        "extensions": {"VRM": {"humanoid": {"humanBones": [
            {"bone": name, "node": i} for name, i in index.items()
        ]}}},
        "extensionsUsed": ["VRM"],
    }
    # The mesh has to hang off a node that also carries the skin.
    doc["nodes"].append({"name": "body", "mesh": 0, "skin": 0})
    doc["scenes"][0]["nodes"].append(len(doc["nodes"]) - 1)

    # GLB chunks are 4-byte aligned; JSON pads with spaces, BIN with zeros.
    json_chunk = json.dumps(doc).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = buffer + b"\x00" * (-len(buffer) % 4)

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return out


# ─────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────
def selftest() -> int:
    import tempfile

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("\nSelf-test — synthetic rig, no external files\n")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selftest.glb"
        path.write_bytes(build_test_glb())

        # ── the parser ───────────────────────────────────────────────
        gltf = Gltf.load(path)
        bones = gltf.humanoid_bones()
        check("GLB container parses", len(gltf.doc["nodes"]) == len(_REST) + 1)
        check("VRM humanoid bone map read",
              set(bones) == set(_REST),
              f"{len(bones)} bones")

        mesh = gltf.skinned_mesh()
        check("skinned geometry extracted",
              mesh.n_vertices == len(_REST),
              f"{mesh.n_vertices} verts, {len(mesh.triangles)} tris")

        rest_ok = np.allclose(
            mesh.joint_rest,
            np.array([_REST[n] for n in _REST]), atol=1e-5)
        check("joint rest positions match the hierarchy", rest_ok)

        # ── the retargeting ──────────────────────────────────────────
        # hand_scale=1 keeps the hand bones, so the hand vertex is testable.
        # mirror=False feeds the rig scene coordinates in the SAME handedness
        # as the model; the mirrored capture convention is checked separately
        # below, because it is a property of body_to_3d and not of the math.
        rig = AvatarRig(mesh, bones, hand_scale=1.0, mirror=False)

        # 1. Rest in, rest out. Feed the rig its own rest pose as the scene
        #    and every vertex must come back untouched.
        scene_rest = {
            "l_shoulder": np.array(_REST["leftupperarm"]),
            "r_shoulder": np.array(_REST["rightupperarm"]),
            "l_elbow": np.array(_REST["leftlowerarm"]),
            "r_elbow": np.array(_REST["rightlowerarm"]),
            "l_wrist": np.array(_REST["lefthand"]),
            "r_wrist": np.array(_REST["righthand"]),
            "nose": np.array(_REST["head"]),
        }
        posed, _ = rig.pose(scene_rest)
        err = float(np.abs(posed - mesh.positions).max())
        check("rest pose is the identity", err < 1e-6, f"max error {err:.2e}")

        # 2. Bend the left arm. Bone lengths are preserved by construction,
        #    so if we place the target elbow and wrist exactly one bone
        #    length apart, the posed hand vertex must land on the wrist.
        ua = np.linalg.norm(np.array(_REST["leftlowerarm"])
                            - np.array(_REST["leftupperarm"]))
        fa = np.linalg.norm(np.array(_REST["lefthand"])
                            - np.array(_REST["leftlowerarm"]))
        shoulder = np.array(_REST["leftupperarm"])
        elbow = shoulder + np.array([0.0, -1.0, 0.0]) * ua      # arm hangs down
        wrist = elbow + np.array([0.0, 0.0, 1.0]) * fa          # forearm forward

        bent = dict(scene_rest)
        bent["l_elbow"], bent["l_wrist"] = elbow, wrist
        posed, _ = rig.pose(bent)

        hand_vertex = posed[list(_REST).index("lefthand")]
        d_hand = float(np.linalg.norm(hand_vertex - wrist))
        check("bent arm puts the hand on the target wrist",
              d_hand < 1e-6, f"off by {d_hand:.2e}")

        elbow_vertex = posed[list(_REST).index("leftlowerarm")]
        d_elbow = float(np.linalg.norm(elbow_vertex - elbow))
        check("bent arm puts the elbow on the target elbow",
              d_elbow < 1e-6, f"off by {d_elbow:.2e}")

        # 3. The untouched right arm must not have moved.
        r_hand = posed[list(_REST).index("righthand")]
        d_right = float(np.linalg.norm(r_hand - np.array(_REST["righthand"])))
        check("the other arm stayed put", d_right < 1e-6,
              f"moved {d_right:.2e}")

        # 4. Scale invariance. Double the signer's size and the avatar must
        #    track it — this is what lets one model fit every signer.
        big = {k: v * 2.0 for k, v in scene_rest.items()}
        posed_big, _ = rig.pose(big)
        got = float(np.linalg.norm(
            posed_big[list(_REST).index("lefthand")]
            - np.array(_REST["lefthand"]) * 2.0))
        check("avatar rescales to the signer", got < 1e-6, f"off by {got:.2e}")

        # 5. Hands collapse when asked, so the landmark rig can stand in.
        # 4b. Facing. This is the one that a symmetric mannequin cannot show.
        #     Real capture puts the signer's anatomical LEFT at NEGATIVE x
        #     (body_to_3d negates x "so it mirrors you") while they still face
        #     +z. Build that frame and check the avatar ends up facing the
        #     camera rather than turning its back.
        capture = {
            "l_shoulder": np.array([-0.20, 1.00, 0.0]),   # left, image-right
            "r_shoulder": np.array([0.20, 1.00, 0.0]),
            "l_elbow": np.array([-0.36, 0.78, 0.05]),
            "r_elbow": np.array([0.36, 0.78, 0.05]),
            "l_wrist": np.array([-0.30, 0.55, 0.25]),
            "r_wrist": np.array([0.30, 0.55, 0.25]),
            "nose": np.array([0.0, 1.20, 0.05]),
        }
        rig_mirror = AvatarRig(mesh, bones, hand_scale=1.0, mirror=True)
        frame = rig_mirror.scene_transform(capture)
        forward = frame[0] @ np.array([0.0, 0.0, 1.0])   # avatar +z in scene
        check("avatar faces the camera under capture handedness",
              forward[2] > 0.9,
              f"model forward maps to z={forward[2]:+.2f} (want +1)")

        turned = AvatarRig(mesh, bones, hand_scale=1.0, mirror=False)
        back = turned.scene_transform(capture)[0] @ np.array([0.0, 0.0, 1.0])
        check("mirror=False is what turned it around", back[2] < -0.9,
              f"z={back[2]:+.2f} — this was the bug")

        # 4c. Stability. A mis-detected nose used to swing the whole-body
        #     basis, and a nose below the shoulders flipped the avatar upside
        #     down. Up is now the world axis, so no facial landmark can roll
        #     the body at all. Sweep the nose everywhere, including below the
        #     shoulders, and the up axis must not budge.
        # Compare the axis directly rather than via an angle: arccos near 1
        # loses half its precision to a square root and cannot resolve a
        # genuinely-zero tilt.
        worst = 0.0
        for nose in ([0.0, 1.20, 0.05], [0.5, 1.20, 0.05], [0.0, 0.80, 0.05],
                     [0.0, 0.95, -0.4], [-0.6, 1.6, 0.3]):
            frame = rig_mirror.scene_transform({**capture,
                                                "nose": np.array(nose)})
            world_up = frame[0] @ np.array([0.0, 1.0, 0.0])
            worst = max(worst, float(np.linalg.norm(
                world_up - np.array([0.0, 1.0, 0.0]))))
        check("a bad nose cannot roll the body", worst < 1e-9,
              f"up axis moved {worst:.2e} over 5 nose positions, "
              f"including one below the shoulders")

        # 4d. Smoothing must not change where a held pose settles, only how
        #     fast it gets there — otherwise it would bias every pose.
        damped = AvatarRig(mesh, bones, hand_scale=1.0, smoothing=0.4)
        for _ in range(60):
            damped.pose(bent)
        settled = damped.pose(bent)[0]
        exact = AvatarRig(mesh, bones, hand_scale=1.0,
                          smoothing=1.0).pose(bent)[0]
        drift = float(np.abs(settled - exact).max())
        check("smoothing converges on the exact pose", drift < 1e-6,
              f"drift {drift:.2e}")

        rig_nohands = AvatarRig(mesh, bones, hand_scale=0.0, mirror=False)
        posed_nh, _ = rig_nohands.pose(scene_rest)
        moved = float(np.linalg.norm(
            posed_nh[list(_REST).index("lefthand")]
            - np.array(_REST["lefthand"])))
        check("hand_scale=0 hides the model's hands", moved < 1e-6,
              "hand vertex sits on the wrist joint")

        # 6. A frame missing the upper body must be refused, not guessed.
        check("incomplete frame returns None",
              rig.pose({"nose": np.zeros(3)}) is None)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("All checks passed — the retargeting math is sound.")
    return 0


# ─────────────────────────────────────────────────────────────────────
#  Real-file report
# ─────────────────────────────────────────────────────────────────────
def inspect(path: Path, max_vertices: int) -> int:
    print(f"\nInspecting {path.name}  ({path.stat().st_size / 1e6:.1f} MB)\n")
    try:
        gltf = Gltf.load(path)
    except GltfError as exc:
        print(f"  Cannot read it: {exc}")
        return 1

    asset = gltf.doc.get("asset", {})
    print(f"  generator     {asset.get('generator', 'unknown')}")
    print(f"  nodes         {len(gltf.doc.get('nodes', []))}")
    print(f"  meshes        {len(gltf.doc.get('meshes', []))}")
    print(f"  skins         {len(gltf.doc.get('skins', []))}")

    vrm_bones = gltf.humanoid_bones()
    if vrm_bones:
        source = "VRM humanoid table"
        bones = vrm_bones
    else:
        bones = gltf.guess_humanoid_bones()
        source = "guessed from node names (no VRM extension present)"

    spec = load_rig_spec(path)
    by_name = {n.get("name", ""): i
               for i, n in enumerate(gltf.doc.get("nodes", []))}
    overridden: set[str] = set()
    if spec:
        for bone, value in (spec.get("bones") or {}).items():
            if value in by_name:
                bones[bone.lower()] = by_name[value]
                overridden.add(bone.lower())
        source = f"{rig_sidecar_path(path).name} over {source}"
    print(f"  bone map      {len(bones)} bones, {source}")

    required = ("leftupperarm", "leftlowerarm", "lefthand",
                "rightupperarm", "rightlowerarm", "righthand")
    missing = [b for b in required if b not in bones]
    for b in HUMANOID_BONES:
        if b in bones:
            name = gltf.doc["nodes"][bones[b]].get("name", "?")
            mark = " (sidecar)" if b in overridden else ""
            print(f"      {b:<16} node {bones[b]:<4} {name}{mark}")

    if missing:
        print(f"\n  UNUSABLE — missing arm bones: {', '.join(missing)}")
        print("  The arms are the whole point; without them there is nothing")
        print("  to drive.")
        print(f"\n  If the bones exist under names the matcher did not")
        print(f"  recognise, describe them yourself:")
        print(f"      python scripts/check_avatar.py {path} --write-rig-template")
        print("  then fill in the names and re-run this check. Otherwise")
        print("  re-export with a standard humanoid rig, or use a VRM from")
        print("  VRoid Studio, which always has one.")
        return 1

    n_chain = len(spec.get("chain_parents") or {})
    if n_chain:
        print(f"  re-parented   {n_chain} bones, to restore a chain the "
              f"export flattened")

    try:
        mesh = gltf.skinned_mesh(max_vertices=max_vertices)
    except GltfError as exc:
        print(f"\n  Cannot build geometry: {exc}")
        return 1

    if gltf.outline_prims:
        print(f"  outline shells {gltf.outline_prims} primitives skipped — "
              f"toon outlines, not geometry")

    print(f"\n  vertices      {mesh.n_vertices}"
          f"  (cap {max_vertices})")
    print(f"  triangles     {len(mesh.triangles)}")
    print(f"  skin joints   {len(mesh.joint_nodes)}")
    flat = mesh.colors.std(axis=0).mean()
    print(f"  colours       {'baked from textures' if flat > 0.01 else 'flat / untextured'}")

    try:
        rig = load_rig(path, max_vertices=max_vertices, hand_scale=0.0)
    except ValueError as exc:
        print(f"\n  Cannot build the rig: {exc}")
        return 1
    print(f"  shoulder span {rig.rest_shoulder_width:.3f} model units")

    # A signing-ish pose: both arms bent forward and up in front of the chest.
    scene = {
        "l_shoulder": np.array([0.2, 1.0, 0.0]),
        "r_shoulder": np.array([-0.2, 1.0, 0.0]),
        "l_elbow": np.array([0.4, 0.75, 0.1]),
        "r_elbow": np.array([-0.4, 0.75, 0.1]),
        "l_wrist": np.array([0.3, 0.5, 0.3]),
        "r_wrist": np.array([-0.3, 0.5, 0.3]),
        "nose": np.array([0.0, 1.25, 0.05]),
    }

    # Does the rig actually REACH? A flattened hierarchy — bones parented to
    # a common root rather than to each other — poses without error and
    # leaves the forearm behind, which is invisible until you watch it move.
    # Compare the angle the posed arm makes against the angle it was aimed at.
    print()
    placed = rig.posed_bone_positions(scene)
    worst = 0.0
    for side, sh, el, wr in (("left", "l_shoulder", "l_elbow", "l_wrist"),
                             ("right", "r_shoulder", "r_elbow", "r_wrist")):
        for seg, a, b in (("upper arm", sh, el), ("forearm", el, wr)):
            bone_a = f"{side}{'upperarm' if seg == 'upper arm' else 'lowerarm'}"
            bone_b = f"{side}{'lowerarm' if seg == 'upper arm' else 'hand'}"
            got = placed[bone_b] - placed[bone_a]
            want = scene[b] - scene[a]
            gn, wn = np.linalg.norm(got), np.linalg.norm(want)
            if gn < 1e-9 or wn < 1e-9:
                continue
            deg = float(np.degrees(np.arccos(
                np.clip(np.dot(got / gn, want / wn), -1.0, 1.0))))
            worst = max(worst, deg)
            flag = "ok" if deg < 1.0 else "OFF"
            print(f"  reach         {side:<5} {seg:<9} {deg:6.2f}° error  {flag}")

    if worst >= 1.0:
        print(f"\n  BROKEN — the arms do not follow. The bones were found but")
        print("  they are not connected to each other, so rotating one does")
        print("  not carry the next. Add a `chain_parents` map to")
        print(f"  {rig_sidecar_path(path).name} joining upper arm → forearm →")
        print("  hand. Run with --write-rig-template for a starting point.")
        return 1

    rig.pose(scene)                       # warm up
    t0 = time.perf_counter()
    for _ in range(20):
        rig.pose(scene)
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  pose cost     {ms:.1f} ms/frame  "
          f"({'fine' if ms < 25 else 'slow — lower --max-vertices'})")

    print("\n  READY — this file will work. Point the live window at it with")
    print(f"      ./run_web.sh          then pick Anime in Configuration")
    print(f"  or  ./venv/bin/python scripts/mannequin_local.py "
          f"--skin anime --avatar {path}")
    return 0


def write_template(path: Path) -> int:
    """Emit a starter `<model>.rig.json` with the automatic guesses filled in.

    Refuses to clobber an existing sidecar — that file is hand-written and
    losing it means working the bone names out a second time.
    """
    out = rig_sidecar_path(path)
    if out.exists():
        print(f"\n  {out.name} already exists — not overwriting it.")
        print("  Edit it directly, or move it aside and re-run.")
        return 1

    gltf = Gltf.load(path)
    bones = gltf.humanoid_bones() or gltf.guess_humanoid_bones()
    nodes = gltf.doc.get("nodes", [])

    required = ("leftshoulder", "leftupperarm", "leftlowerarm", "lefthand",
                "rightshoulder", "rightupperarm", "rightlowerarm",
                "righthand", "neck", "head", "hips", "spine", "chest")
    filled = {b: nodes[bones[b]].get("name", "") if b in bones
              else "FIXME — node name from the listing above"
              for b in required}

    spec = {
        "_comment": [
            "Bone names for this model. Values are node names exactly as",
            "check_avatar.py prints them. Replace every FIXME.",
            "",
            "chain_parents is only needed when the export flattened the",
            "skeleton — if check_avatar.py reports the arms do not reach,",
            "join upper arm -> forearm -> hand here.",
        ],
        "bones": filled,
        "chain_parents": {},
    }
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote {out.name}")
    missing = [b for b in required if b not in bones]
    if missing:
        print(f"  {len(missing)} bone(s) need filling in: "
              f"{', '.join(missing)}")
    print(f"  Then re-run:  python scripts/check_avatar.py {path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("avatar", nargs="?", type=Path,
                    help="a .vrm / .glb / .gltf file to inspect")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the retargeting math on a synthetic rig")
    ap.add_argument("--max-vertices", type=int, default=40000,
                    help="skinning budget. Higher is crisper and slower; the\n"
                         "report prints the per-frame cost.")
    ap.add_argument("--write-rig-template", action="store_true",
                    help="emit a starter <model>.rig.json to describe bones "
                         "the automatic matcher could not identify")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.avatar is None:
        ap.error("give an avatar file, or pass --selftest")
    if not args.avatar.exists():
        print(f"no such file: {args.avatar}")
        sys.exit(2)
    if args.write_rig_template:
        sys.exit(write_template(args.avatar))
    sys.exit(inspect(args.avatar, args.max_vertices))


if __name__ == "__main__":
    main()
