"""Entry point:  python -m webapp

Starts the Flask control panel in a daemon thread and runs the RecorderEngine
supervisor on the MAIN thread (OpenCV/Open3D require it). The browser drives
everything; the command line is just this one launch.
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path
from threading import Thread

from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config                       # noqa: E402
from webapp.app import AppState, create_app             # noqa: E402
from webapp.engine import RecorderEngine                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SignLink recording control panel")
    ap.add_argument("--signer", default="me",
                    help="default signer tag for new recordings")
    ap.add_argument("--lang", default="khmer",
                    help="language selected on startup")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=str(ROOT / "config" / "settings.json"))
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open the browser")
    args = ap.parse_args()

    cfg = load_config(args.config)
    engine = RecorderEngine(cfg, signer=args.signer, language=args.lang,
                            config_path=args.config)
    state = AppState(engine)
    app = create_app(state)

    server = make_server(args.host, args.port, app, threaded=True)
    Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://{args.host}:{args.port}/"
    print("=" * 60)
    print("  SignLink Control Center")
    print(f"  open:  {url}")
    print("  (the live camera + mannequin is a separate desktop window)")
    print("  Ctrl+C here, or Quit in the browser, to stop.")
    print("=" * 60)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    _supervise(state, engine)
    server.shutdown()
    print("\n[shutdown] bye.")


def _supervise(state: AppState, engine: RecorderEngine) -> None:
    """Main-thread loop: reconcile the engine with the requested mode, and pump
    the native window when recording. All cv2/o3d calls happen here."""
    warned_no_cam = False
    try:
        while True:
            with state.lock:
                if state.shutdown:
                    break
                mode = state.mode

            recognizing = mode == "recognize" and engine.recognizer is not None

            if mode == "record" or recognizing:
                if not engine.running:
                    # Recognize mode wants the camera but no mannequin scene.
                    if engine.start(mannequins=(mode == "record")):
                        warned_no_cam = False
                    else:
                        if not warned_no_cam:
                            print("[camera] failed to open — retrying...")
                            warned_no_cam = True
                        time.sleep(1.0)
                        continue
                if mode == "record":
                    engine.tick()
                else:
                    engine.tick_recognize()
                time.sleep(0.01)
            else:
                # Recognize mode with nothing loaded yet, or idle: release the
                # camera so the other mode (and other apps) can have it.
                if engine.running:
                    engine.stop()
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if engine.running:
            engine.stop()


if __name__ == "__main__":
    main()
