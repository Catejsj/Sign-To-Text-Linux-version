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

from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory

from webapp import library
from webapp.engine import RecorderEngine, HAS_OPEN3D, list_models

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
        )
        if data.get("lang"):
            engine.set_language(data["lang"])
        if data.get("signer"):
            engine.set_signer(data["signer"])
        return jsonify(ok=True, config=engine.snapshot()["config"])

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
