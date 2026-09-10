"""Flask control panel for SignLink recording.

The browser is a control surface only — no video. The live camera + mannequin is
the native OpenCV window driven by RecorderEngine on the MAIN thread. Flask runs
in a daemon thread and only flips shared state:

    - mode:    "record" | "recognize"  (isolated; switching tears the other down)
    - engine:  thread-safe setters (queue_label, set_config, ...)
    - library: pure filesystem ops (create/scan/delete) — safe from any thread

The main-thread supervisor loop (see __main__.py) reads `state.mode` and
starts/stops/ticks the engine accordingly, so every cv2/o3d call stays on the
main thread.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_file, send_from_directory

from webapp import library
from webapp.engine import RecorderEngine, HAS_OPEN3D, list_models

try:
    from scripts.mannequin_skins import find_avatar as avatar_file
except ImportError:                       # no Open3D installed
    def avatar_file(explicit=None):
        """Locate an avatar without importing the Open3D-backed skins module.

        The browser viewer does not need Open3D at all, so a machine with no
        Open3D should still be able to serve a model to the page.
        """
        folder = Path(__file__).resolve().parents[1] / "assets" / "avatars"
        if not folder.is_dir():
            return None
        found = [p for p in folder.iterdir()
                 if p.suffix.lower() in (".vrm", ".glb", ".gltf") and p.is_file()]
        if not found:
            return None
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found[0]

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AppState:
    """Shared between the Flask thread and the main supervisor thread."""
    def __init__(self, engine: RecorderEngine):
        self.engine = engine
        self.lock = Lock()
        self.mode = "record"          # "record" | "recognize"
        self.shutdown = False

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.mode = mode


def create_app(state: AppState) -> Flask:
    app = Flask(__name__, static_folder=None)
    engine = state.engine

    # ── UI ──
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:fname>")
    def static_files(fname):
        return send_from_directory(STATIC_DIR, fname)

    # ── state ──
    @app.get("/api/state")
    def api_state():
        with state.lock:
            mode = state.mode
        return jsonify({
            "mode": mode,
            "has_open3d": HAS_OPEN3D,
            "engine": engine.snapshot(),
            "languages": library.list_languages(),
        })

    @app.post("/api/mode")
    def api_mode():
        mode = (request.json or {}).get("mode")
        if mode not in ("record", "recognize"):
            return jsonify(error="mode must be 'record' or 'recognize'"), 400
        state.set_mode(mode)
        return jsonify(ok=True, mode=mode)

    # ── languages ──
    @app.get("/api/languages")
    def api_languages():
        return jsonify(library.list_languages())

    @app.post("/api/languages")
    def api_create_language():
        name = (request.json or {}).get("name", "")
        try:
            library.create_language(name)
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, name=name)

    # ── labels + takes (record mode only) ──
    @app.get("/api/labels")
    def api_labels():
        lang = request.args.get("lang", "")
        if not (library.SEQUENCES / lang).exists():
            return jsonify(error=f"language '{lang}' not found"), 404
        return jsonify(library.scan_language(lang))

    @app.post("/api/labels")
    def api_add_label():
        if not _require_record():
            return _wrong_mode()
        data = request.json or {}
        lang, text = data.get("lang", ""), data.get("text", "")
        try:
            slug = library.add_label(lang, text)
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, slug=slug)

    @app.delete("/api/labels")
    def api_delete_label():
        if not _require_record():
            return _wrong_mode()
        d = request.json or {}
        try:
            library.delete_label(d.get("lang", ""), d.get("slug", ""))
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True)

    @app.post("/api/record")
    def api_record():
        if not _require_record():
            return _wrong_mode()
        data = request.json or {}
        if data.get("lang"):
            engine.set_language(data["lang"])
        if data.get("signer"):
            engine.set_signer(data["signer"])
        label = data.get("label")
        if not label:
            return jsonify(error="label required"), 400
        engine.queue_label(label)
        return jsonify(ok=True)

    @app.post("/api/stop")
    def api_stop():
        if not _require_record():
            return _wrong_mode()
        engine.stop_take()
        return jsonify(ok=True)

    @app.post("/api/config")
    def api_config():
        if not _require_record():
            return _wrong_mode()
        data = request.json or {}
        engine.set_config(
            mannequin=data.get("mannequin"),
            synthetic=data.get("synthetic"),
            duration=data.get("duration"),
            view=data.get("view"),
            skin=data.get("skin"),
        )
        if data.get("lang"):
            engine.set_language(data["lang"])
        if data.get("signer"):
            engine.set_signer(data["signer"])
        return jsonify(ok=True, config=engine.snapshot()["config"])

    # ── browser 3D view ──
    @app.get("/api/landmarks")
    def api_landmarks():
        """The current frame's scene coordinates.

        Polled rather than streamed: the page asks about as often as the
        camera produces frames, and `seq` tells it when nothing is new. A
        Server-Sent Events stream would shave a few milliseconds and add a
        connection lifecycle to get wrong on a dev server.
        """
        return jsonify(engine.landmark_snapshot())

    @app.get("/api/avatar/model")
    def api_avatar_model():
        """The avatar file itself, for three.js to load in the browser."""
        path = avatar_file()
        if path is None:
            return jsonify(error="no avatar file"), 404
        return send_file(path, mimetype="model/gltf-binary",
                         conditional=True)

    _rig_cache: dict = {}

    @app.get("/api/avatar/rig")
    def api_avatar_rig():
        """The rig description for the browser viewer.

        Returns the `<model>.rig.json` sidecar plus `resolved_bones`: the
        humanoid bone map as Python worked it out, by node name.

        Resolving bones in one place is the point. Both viewers used to match
        bone names independently and they drifted twice — once on an
        Auto-Rig Pro rig calling the upper arm `arm.l`, once on a VRChat rig
        calling the forearm `Lower_Arm_L`, each time working in Python and
        failing silently in the browser. The Python matcher is the one
        `check_avatar.py` exercises, so it is the one that decides; the
        browser keeps its own table only as a fallback for when this fails.
        """
        path = avatar_file()
        if path is None:
            return jsonify({})

        spec: dict = {}
        sidecar = Path(str(path) + ".rig.json")
        if sidecar.exists():
            try:
                spec = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return jsonify(
                    error=f"{sidecar.name} is not valid JSON: {exc}"), 400

        key = (str(path), path.stat().st_mtime_ns)
        if key not in _rig_cache:
            _rig_cache.clear()          # only ever one avatar at a time
            try:
                from src.gltf_min import Gltf
                gltf = Gltf.load(path)
                bones = gltf.humanoid_bones() or gltf.guess_humanoid_bones()
                nodes = gltf.doc.get("nodes", [])
                _rig_cache[key] = {
                    b: nodes[i].get("name", "")
                    for b, i in bones.items()
                    if 0 <= i < len(nodes) and nodes[i].get("name")
                }
            except Exception:
                # Never fail the page over this — the browser can still fall
                # back to its own name matching.
                _rig_cache[key] = {}
        spec["resolved_bones"] = _rig_cache[key]
        return jsonify(spec)

    # ── deletion (record mode only) ──
    @app.delete("/api/takes")
    def api_delete_take():
        if not _require_record():
            return _wrong_mode()
        d = request.json or {}
        try:
            n = library.delete_take(
                d["lang"], d["slug"], d["signer"], int(d["variant"]),
                d.get("source", "real"))
        except (KeyError, ValueError):
            return jsonify(error="lang, slug, signer, variant required"), 400
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, removed=n)

    @app.post("/api/delete_all")
    def api_delete_all():
        if not _require_record():
            return _wrong_mode()
        d = request.json or {}
        try:
            n = library.delete_all(d.get("lang", ""), d.get("slug"))
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, removed=n)

    @app.post("/api/synthetic/clear")
    def api_clear_synthetic():
        if not _require_record():
            return _wrong_mode()
        d = request.json or {}
        try:
            n = library.clear_synthetic(d.get("lang", ""), d.get("slug"))
        except library.LibraryError as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, removed=n)

    # ── recognize ──
    @app.get("/api/models")
    def api_models():
        return jsonify(models=list_models())

    @app.post("/api/recognize/start")
    def api_recognize_start():
        if not _require_recognize():
            return jsonify(error="switch to Recognize mode first"), 409
        name = (request.json or {}).get("model")
        if not name:
            return jsonify(error="model required"), 400
        try:
            engine.start_recognition(name)
        except (ValueError, OSError) as e:
            return jsonify(error=str(e)), 400
        return jsonify(ok=True, model=name)

    @app.post("/api/recognize/stop")
    def api_recognize_stop():
        engine.stop_recognition()
        return jsonify(ok=True)

    @app.get("/api/recognize/state")
    def api_recognize_state():
        return jsonify(engine.recognition_snapshot())

    @app.post("/api/quit")
    def api_quit():
        """Stop the server for good. The page cannot recover from this — the
        UI asks for explicit confirmation before calling it."""
        with state.lock:
            state.shutdown = True
        return jsonify(ok=True)

    # ── helpers ──
    def _require_record() -> bool:
        with state.lock:
            return state.mode == "record"

    def _require_recognize() -> bool:
        with state.lock:
            return state.mode == "recognize"

    def _wrong_mode():
        return jsonify(error="not in record mode"), 409

    return app
