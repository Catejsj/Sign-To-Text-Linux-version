"""RecorderEngine — the native camera + mannequin window, driven by web state.

This is a refactor of scripts/record_session.py's `Recorder`. The behaviour of
the capture/record/mannequin pipeline is unchanged; what changes is where the
inputs come from. Instead of the old in-process `BrowserUI` label dialog, the
engine exposes thread-safe setters that the Flask layer calls, and a `snapshot()`
the UI polls.

THREADING CONTRACT (important):
    OpenCV windows and the Open3D visualizer are NOT thread-safe and must be
    created/pumped from ONE thread. So `start()`, `stop()` and `tick()` are all
    called from the **main thread** (by the supervisor loop in __main__.py).
    Flask runs in another thread and only ever calls the setters below, which
    touch plain fields under `self.lock` — never cv2/o3d directly. Mannequin
    rebuilds (which touch o3d) are deferred: a setter records the *desired*
    count and `tick()` applies it on the main thread.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture import LandmarkCapture                                  # noqa: E402
from src.v2.normalize import (                                          # noqa: E402
    frame_from_landmarks, clean_clip_from_frames, noisy_clip_from_frames,
)
from src.v2.schema import save_pair                                     # noqa: E402
from src.v2.retarget import generate_variants                          # noqa: E402
from src.v2.recognizer import (                                        # noqa: E402
    LiveRecognizer, bundle_path, list_models, load_bundle,
)
from scripts.record_session import (                                   # noqa: E402
    OverlayFont, COUNTDOWN_S, MAX_RECORD_S, MANNEQUIN_W, MANNEQUIN_H,
)
from webapp import library                                             # noqa: E402
from webapp.media import (                                             # noqa: E402
    KIND_MEDIA_ONLY, KIND_WITH_LANDMARKS, MediaTake, next_media_only_variant,
)

# What a take can capture. Any combination is allowed except none at all.
# Landmarks-only is the default because it is what every existing recording,
# result and privacy statement in this project assumes.
CAPTURE_MODES = ("landmarks", "video", "frames")

# A series records the same sign again and again from one click. Between takes
# the signer gets REST_S to drop their hands, then the usual countdown. After
# MAX_MISSES discarded takes in a row (nobody in frame) the series pauses
# itself rather than spin forever.
REST_S = 1.5
MAX_MISSES = 3
DEFAULT_MODES = {"landmarks": True, "video": False, "frames": False}

try:
    import open3d as o3d                                               # noqa: E402
    from scripts.mannequin_local import (                              # noqa: E402
        LINE_WIDTH, Mannequin, retarget_scene, body_to_3d, hand_to_3d,
    )
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

CAM_WINDOW = "SignLink — Camera + Mannequin"


class RecorderEngine:
    def __init__(self, cfg, signer: str = "me", language: str = "khmer",
                 config_path: str | None = None):
        self.cfg = cfg
        self.config_path = config_path or str(ROOT / "config" / "settings.json")
        self.img_w = cfg["capture"]["width"]
        self.img_h = cfg["capture"]["height"]
        self.fps = cfg["capture"].get("fps", 30)

        self.lock = Lock()
        self.rng = np.random.default_rng()

        # ── config (mutated by Flask, read by tick) ──
        self.language = language
        self.signer = signer
        self.synthetic = 0
        self.duration = 3.0
        self.desired_mannequin = 1 if HAS_OPEN3D else 0
        # which panes the native window shows: "both" | "camera" | "mannequin"
        self.view = "both"
        # what each take saves: landmarks, a video, still frames, or a mix
        self.capture_modes = dict(DEFAULT_MODES)

        # ── recording state (owned by tick / main thread) ──
        self.state = "IDLE"
        self.current_label: str | None = None
        self.pending_label: str | None = None
        self.stop_now = False
        self.cancel_countdown = False
        self.countdown_start = 0.0
        self.record_start = 0.0
        self.phase_end_at: float | None = None
        self.take_buffer: list[np.ndarray] = []
        self.media_take: MediaTake | None = None
        self.take_modes = dict(DEFAULT_MODES)   # frozen when a take starts
        self.session_take_count = 0
        self.last_saved: str | None = None

        # ── take series (one click, many takes) ──
        self.series_label: str | None = None
        self.series_target = 0          # takes to save; 0 = no series
        self.series_done = 0
        self.series_paused = False
        self.series_misses = 0
        self.series_next_at = 0.0

        # ── recognition (Recognize mode) ──
        self.recognizer: LiveRecognizer | None = None
        self.recognizer_name: str | None = None
        self.last_prediction: dict | None = None
        self.history: list[dict] = []      # recent stable predictions

        # ── screen recording (main thread writes, Flask reads) ──
        self.writer: cv2.VideoWriter | None = None
        self.video_path: Path | None = None
        self.video_size: tuple[int, int] | None = None
        self.video_started_at = 0.0
        self.video_frames = 0

        # ── runtime handles (main thread only) ──
        self.capture: LandmarkCapture | None = None
        self.overlay: OverlayFont | None = None
        self.vis = None
        self.synths: list[tuple] = []
        self.current_mannequin = 0
        self.current_view = "both"
        self.running = False

    # ── lifecycle (main thread) ──────────────────────────────────────
    def start(self, mannequins: bool = True) -> bool:
        """Open the camera and the native window.

        `mannequins=False` is used by Recognize mode: it only needs the camera
        feed, so skipping the Open3D scene saves GPU work and start-up time.
        """
        if self.running:
            return True
        self.capture = LandmarkCapture(self.cfg)
        if not self.capture.start():
            self.capture = None
            return False
        self.overlay = OverlayFont()
        cv2.namedWindow(CAM_WINDOW, cv2.WINDOW_NORMAL)
        if mannequins:
            with self.lock:
                view, desired = self.view, self.desired_mannequin
            self._build_mannequins(self._needed_mannequin(view, desired))
            self.current_view = view
            cv2.resizeWindow(CAM_WINDOW, self._compose_width(view), self.img_h)
        else:
            self.current_view = "camera"
            cv2.resizeWindow(CAM_WINDOW, self.img_w, self.img_h)
        self.running = True
        return True

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        # close the file first — a writer left open across a mode switch or a
        # quit produces an .mp4 with no moov atom, which plays nowhere
        self.stop_video()
        self._teardown_mannequins()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        try:
            cv2.destroyWindow(CAM_WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass
        # abandon any half-recorded take
        self.state = "IDLE"
        self.current_label = None
        self.pending_label = None
        self.take_buffer = []
        self.media_take = None
        self._end_series()

    def _mann_pane_width(self) -> int:
        return int(MANNEQUIN_W * self.img_h / MANNEQUIN_H)

    def _compose_width(self, view: str) -> int:
        """Width of the native window for the current view + mannequin count."""
        if view == "mannequin":
            return self._mann_pane_width()
        if view == "camera":
            return self.img_w
        # both
        return self.img_w + (self._mann_pane_width() if self.current_mannequin > 0 else 0)

    def _needed_mannequin(self, view: str, desired: int) -> int:
        """How many mannequins to actually build, given the view. Camera-only
        needs none; mannequin-only needs at least one even if the slider is 0."""
        if view == "camera":
            return 0
        if view == "mannequin":
            return desired if desired > 0 else 1
        return desired

    # ── mannequin scene (main thread) ────────────────────────────────
    def _build_mannequins(self, n: int) -> None:
        if not HAS_OPEN3D or n <= 0:
            self.current_mannequin = 0
            return
        spacing = 2.6
        xs = (np.arange(n) - (n - 1) / 2.0) * spacing
        self.synths = []
        for i in range(n):
            body = dict(
                sh=float(self.rng.uniform(0.80, 1.20)),
                ua=float(self.rng.uniform(0.80, 1.20)),
                fa=float(self.rng.uniform(0.80, 1.20)),
                hd=float(self.rng.uniform(0.80, 1.20)),
                xoff=float(xs[i]),
            )
            self.synths.append((Mannequin(), body))
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("SignLink mannequin (offscreen)",
                               width=MANNEQUIN_W, height=MANNEQUIN_H,
                               visible=False)
        for mq, _ in self.synths:
            for g in mq.geometries():
                self.vis.add_geometry(g)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.06, 0.09])
        opt.light_on = True
        opt.line_width = LINE_WIDTH
        vc = self.vis.get_view_control()
        vc.set_front([0.0, 0.0, 1.0]); vc.set_up([0.0, 1.0, 0.0])
        vc.set_lookat([0.0, 0.0, 0.0]); vc.set_zoom(0.75 + 0.14 * n)
        self.current_mannequin = n

    def _teardown_mannequins(self) -> None:
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
        self.vis = None
        self.synths = []
        self.current_mannequin = 0

    # ── setters (Flask thread — plain fields under lock only) ─────────
    def queue_label(self, text: str) -> None:
        with self.lock:
            if self.series_target:      # a series owns the queue until it ends
                return
            self.pending_label = text

    def stop_take(self) -> None:
        with self.lock:
            if self.state == "RECORDING":
                self.stop_now = True

    def start_series(self, text: str, count: int) -> None:
        """Record `text` `count` times in a row from a single click."""
        with self.lock:
            if self.series_target or self.state != "IDLE":
                raise RuntimeError("a take or series is already running")
            self.series_label = text
            self.series_target = max(1, int(count))
            self.series_done = 0
            self.series_paused = False
            self.series_misses = 0
            self.pending_label = text

    def pause_series(self, paused: bool) -> None:
        """Pause after the current take finishes, or resume. A take that is
        already recording is never cut short by a pause."""
        with self.lock:
            if not self.series_target:
                return
            self.series_paused = bool(paused)
            if paused:
                # nothing recorded yet, so a countdown in progress is dropped
                self.pending_label = None
                if self.state == "COUNTDOWN":
                    self.cancel_countdown = True
            else:
                self.series_misses = 0
                self.series_next_at = 0.0     # resume straight into a countdown

    def end_series(self) -> None:
        """Stop the series. A countdown is cancelled; a take already recording
        finishes and is saved, since it is a complete take."""
        with self.lock:
            if not self.series_target:
                return
            self._end_series()
            self.pending_label = None
            if self.state == "COUNTDOWN":
                self.cancel_countdown = True

    def _end_series(self) -> None:
        self.series_label = None
        self.series_target = 0
        self.series_done = 0
        self.series_paused = False
        self.series_misses = 0

    def set_signer(self, signer: str) -> None:
        with self.lock:
            self.signer = signer

    def set_language(self, language: str) -> None:
        with self.lock:
            # changing language mid-take would misfile it; only when idle
            if self.state == "IDLE" and not self.series_target:
                self.language = language

    def set_config(self, mannequin=None, synthetic=None, duration=None,
                   view=None, modes=None) -> None:
        with self.lock:
            if mannequin is not None:
                self.desired_mannequin = max(0, int(mannequin))
            if synthetic is not None:
                self.synthetic = max(0, int(synthetic))
            if duration is not None:
                self.duration = float(np.clip(duration, 0.5, MAX_RECORD_S))
            if view in ("both", "camera", "mannequin"):
                self.view = view
            if isinstance(modes, dict):
                new = {m: bool(modes.get(m, self.capture_modes[m]))
                       for m in CAPTURE_MODES}
                # A take that saves nothing is never what anyone meant.
                if not any(new.values()):
                    new["landmarks"] = True
                self.capture_modes = new

    # ── recognition ──────────────────────────────────────────────────
    def start_recognition(self, model_name: str) -> None:
        """Load a saved model and begin classifying the live camera.

        Called from the Flask thread — it only builds objects and sets state;
        the camera itself is started by the main-thread supervisor.
        """
        models = {m["name"]: m for m in list_models()}
        if model_name not in models:
            raise ValueError(f"no saved model named {model_name!r}")
        info = models[model_name]
        bundle = load_bundle(bundle_path(info["language"], info["algo"]))
        with self.lock:
            self.recognizer = LiveRecognizer(bundle, fps=self.fps)
            self.recognizer_name = model_name
            self.last_prediction = None
            self.history = []

    def stop_recognition(self) -> None:
        with self.lock:
            self.recognizer = None
            self.recognizer_name = None
            self.last_prediction = None

    def recognition_snapshot(self) -> dict:
        with self.lock:
            return {
                "active": self.recognizer is not None,
                "model": self.recognizer_name,
                "prediction": self.last_prediction,
                "history": list(self.history[-8:]),
            }

    def tick_recognize(self) -> None:
        """One recognition frame. MAIN THREAD ONLY (drives the cv2 window)."""
        if self.capture is None:
            return

        view = self._sync_view()

        ret, frame = self.capture.read_frame()
        if not ret or frame is None:
            cv2.waitKey(1)
            return

        with self.capture.result_lock:
            pose = dict(self.capture.latest_pose)
            lh = dict(self.capture.latest_left_hand)
            rh = dict(self.capture.latest_right_hand)

        with self.lock:
            rec = self.recognizer

        pred = None
        if rec is not None and pose:
            rec.push(frame_from_landmarks(pose, lh, rh, self.img_w, self.img_h))
            pred = rec.predict()

        if pred is not None:
            entry = {"label": pred.label, "text": pred.text,
                     "confidence": round(float(pred.confidence), 3),
                     "stable": pred.stable, "moving": pred.moving,
                     "committed": pred.committed}
            with self.lock:
                self.last_prediction = entry
                # log only committed answers (end of a sign), never the same twice
                if pred.committed and pred.text and (
                        not self.history or self.history[-1]["text"] != pred.text):
                    self.history.append({"text": pred.text,
                                         "confidence": entry["confidence"],
                                         "at": time.strftime("%H:%M:%S")})

        # The prediction is drawn on the camera frame before composing, so the
        # text stays over the camera pane rather than straddling both.
        self._draw_recognition(frame, pred)
        out = self._compose(frame, view, pose, lh, rh)
        self._write_video(out)

        cv2.imshow(CAM_WINDOW, out)
        cv2.waitKey(1)

    def _draw_recognition(self, frame: np.ndarray, pred) -> None:
        cv2.putText(frame, "RECOGNIZING", (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 200, 255), 2, cv2.LINE_AA)
        if pred is None:
            self.overlay.draw(frame, "warming up...", (15, 100), 28, (180, 180, 180))
            return
        if not pred.moving:
            self.overlay.draw(frame, "waiting for a sign...", (15, 100), 28,
                              (180, 180, 180))
            return
        colour = (0, 255, 0) if pred.stable else (200, 200, 200)
        self.overlay.draw(frame, pred.text or "?", (15, 100), 44, colour)
        cv2.putText(frame, f"{pred.confidence*100:.0f}%", (15, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    def snapshot(self) -> dict:
        with self.lock:
            remaining = (max(0.0, self.phase_end_at - time.time())
                         if self.phase_end_at else 0.0)
            return {
                "running": self.running,
                "language": self.language,
                "signer": self.signer,
                "state": self.state,
                "current_label": self.current_label,
                "pending_label": self.pending_label,
                "remaining": round(remaining, 2),
                "session_take_count": self.session_take_count,
                "last_saved": self.last_saved,
                "series": ({
                    "label": self.series_label,
                    "target": self.series_target,
                    "done": self.series_done,
                    "paused": self.series_paused,
                } if self.series_target else None),
                "config": {
                    "mannequin": self.desired_mannequin,
                    "synthetic": self.synthetic,
                    "duration": self.duration,
                    "view": self.view,
                    "modes": dict(self.capture_modes),
                },
            }

    # ── per-frame update (main thread) ───────────────────────────────
    # ── screen recording ─────────────────────────────────────────────
    def start_video(self) -> Path:
        """Begin writing the native window to an .mp4.

        Called from Flask, but the writer is only *used* on the main thread in
        `_write_video`, and a half-built writer is never visible because the
        attribute is assigned last.
        """
        if self.writer is not None:
            raise RuntimeError("already recording video")
        out_dir = ROOT / "demo_videos"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"signlink_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        # The frame size has to be fixed for the whole file, so it is locked in
        # here from the current view. Switching camera/model mid-take changes
        # the composed width, and _write_video letterboxes rather than fail.
        w, h = self._compose_width(self.current_view), self.img_h
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(self.fps), (w, h))
        if not writer.isOpened():
            raise OSError("could not open the video writer (mp4v unavailable?)")

        self.video_path, self.video_size = path, (w, h)
        self.video_started_at, self.video_frames = time.time(), 0
        self.writer = writer
        return path

    def stop_video(self) -> dict | None:
        w = self.writer
        if w is None:
            return None
        self.writer = None          # stop _write_video before releasing
        w.release()
        info = {"path": str(self.video_path), "name": self.video_path.name,
                "seconds": round(time.time() - self.video_started_at, 1),
                "frames": self.video_frames}
        self.video_path = self.video_size = None
        return info

    def video_snapshot(self) -> dict:
        return {
            "recording": self.writer is not None,
            "name": self.video_path.name if self.video_path else None,
            "seconds": (round(time.time() - self.video_started_at, 1)
                        if self.writer is not None else 0.0),
        }

    def _write_video(self, out: np.ndarray) -> None:
        """Append one composed frame. MAIN THREAD ONLY."""
        if self.writer is None or self.video_size is None:
            return
        w, h = self.video_size
        if (out.shape[1], out.shape[0]) != (w, h):
            # the view was switched mid-recording — fit the frame into the size
            # the file was opened with instead of dropping it
            scale = min(w / out.shape[1], h / out.shape[0])
            small = cv2.resize(out, (int(out.shape[1] * scale),
                                     int(out.shape[0] * scale)))
            canvas = np.zeros((h, w, 3), dtype=out.dtype)
            y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
            canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
            out = canvas
        self.writer.write(out)
        self.video_frames += 1

    def _sync_view(self) -> str:
        """Apply pending view / mannequin-count changes. MAIN THREAD ONLY —
        Open3D must build and tear down its window on the thread that owns it.

        Returns the view now in force. Shared by record and recognize so both
        modes offer the same camera / model / both switch.
        """
        with self.lock:
            view = self.view
            desired = self.desired_mannequin
        needed = self._needed_mannequin(view, desired)
        if needed != self.current_mannequin or view != self.current_view:
            if needed != self.current_mannequin:
                self._teardown_mannequins()
                self._build_mannequins(needed)
            self.current_view = view
            try:
                cv2.resizeWindow(CAM_WINDOW, self._compose_width(view), self.img_h)
            except cv2.error:
                pass
        return view

    def _compose(self, frame: np.ndarray, view: str,
                 pose: dict, lh: dict, rh: dict) -> np.ndarray:
        """Camera frame, mannequin pane, or both side by side.

        The mannequin is driven by the live pose even when the camera pane is
        hidden — hiding it is a display choice, not a capture one.
        """
        mann = None
        if self.vis is not None and self.current_mannequin > 0:
            if pose:
                self._update_mannequins(pose, lh, rh)
            self.vis.poll_events()
            self.vis.update_renderer()
            mann = self._mannequin_image(frame.shape[0])

        show_camera = view != "mannequin"
        if show_camera and mann is not None:
            return np.hstack([frame, mann])
        if show_camera:
            return frame
        if mann is not None:
            return mann
        return frame          # mannequin-only but none built yet — show camera

    def tick(self) -> None:
        if not self.running or self.capture is None:
            return

        view = self._sync_view()

        ret, frame = self.capture.read_frame()
        if not ret or frame is None:
            cv2.waitKey(1)
            return

        with self.capture.result_lock:
            pose = dict(self.capture.latest_pose)
            lh = dict(self.capture.latest_left_hand)
            rh = dict(self.capture.latest_right_hand)
        now = time.time()

        self._advance_state(pose, lh, rh, now)
        out = self._compose(frame, view, pose, lh, rh)
        # overlay the status text last, so it is visible whichever pane shows
        self._draw_overlay(out, now)
        self._write_video(out)

        cv2.imshow(CAM_WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            self.stop_take()
        elif key == ord("p"):
            with self.lock:
                paused = self.series_paused
            self.pause_series(not paused)

    def _advance_state(self, pose, lh, rh, now: float) -> None:
        with self.lock:
            # a series queues its next take once the rest gap is over
            if (self.state == "IDLE" and self.series_target
                    and not self.series_paused and self.pending_label is None
                    and now >= self.series_next_at):
                self.pending_label = self.series_label
            pending = self.pending_label
            stop_now = self.stop_now
            duration = self.duration
            cancel = self.cancel_countdown
            self.cancel_countdown = False

        if cancel and self.state == "COUNTDOWN":
            self.state = "IDLE"
            self.phase_end_at = None
            return

        if self.state == "IDLE" and pending is not None:
            self.current_label = pending
            with self.lock:
                self.pending_label = None
            self.state = "COUNTDOWN"
            self.countdown_start = now
            self.phase_end_at = now + COUNTDOWN_S

        if self.state == "COUNTDOWN" and now - self.countdown_start >= COUNTDOWN_S:
            self.state = "RECORDING"
            self.record_start = now
            self.take_buffer = []
            # Freeze the modes for this take, so flipping a switch mid-take
            # cannot produce a take that is half one thing and half another.
            with self.lock:
                self.take_modes = dict(self.capture_modes)
            m = self.take_modes
            self.media_take = (MediaTake(video=m["video"], frames=m["frames"])
                               if m["video"] or m["frames"] else None)
            self.phase_end_at = now + duration

        if self.state == "RECORDING":
            if pose:
                self.take_buffer.append(
                    frame_from_landmarks(pose, lh, rh, self.img_w, self.img_h))
            if self.media_take is not None:
                # RAW frame: read_frame() has the skeleton drawn on it, and a
                # model trained on that would learn the drawing, not the hand.
                raw, seq = self.capture.read_raw_frame()
                if raw is not None:
                    # the landmark frame added on this same tick, if any
                    lm = len(self.take_buffer) - 1 if pose else None
                    self.media_take.add(raw, seq, lm, now)
            elapsed = now - self.record_start
            if stop_now or elapsed >= duration or elapsed >= MAX_RECORD_S:
                saved = self._finalize_take()
                self.state = "IDLE"
                self.phase_end_at = None
                with self.lock:
                    self.stop_now = False
                    self._series_after_take(saved, now)

    def _series_after_take(self, saved: bool, now: float) -> None:
        """Count a finished take against the series. Caller holds self.lock."""
        if not self.series_target:
            return
        if saved:
            self.series_done += 1
            self.series_misses = 0
        else:
            self.series_misses += 1
            if self.series_misses >= MAX_MISSES:
                self.series_paused = True
                print(f"series paused: {MAX_MISSES} takes in a row discarded "
                      "(is anyone in frame?)")
        if self.series_done >= self.series_target:
            print(f"series done: {self.series_done} takes of {self.series_label}")
            self._end_series()
        else:
            self.series_next_at = now + REST_S

    def _finalize_take(self) -> bool:
        """Save the take. Returns False if it was discarded."""
        modes = self.take_modes
        media, self.media_take = self.media_take, None

        # Landmarks need 5 frames with a body pose found. A take without
        # landmarks needs 5 camera frames. Failing the check saves nothing,
        # including any video, so a take is never stored half-paired.
        if modes["landmarks"]:
            n = len(self.take_buffer)
        else:
            n = len(media) if media is not None else 0
        if n < 5 or not self.current_label:
            print(f"WARN: only {n} frames — discarded.")
            return False

        with self.lock:
            language, signer, synthetic = self.language, self.signer, self.synthetic
        # resolve (or create) the slug for this label text in this language
        slug = library.add_label(language, self.current_label)
        saved: list[str] = []

        if modes["landmarks"]:
            clean = clean_clip_from_frames(self.take_buffer)
            noisy = noisy_clip_from_frames(self.take_buffer)
            p_clean, p_noisy = save_pair(
                library.SEQUENCES, clean, noisy,
                label=slug, signer_id=signer, language=language, fps=self.fps)
            saved.append(p_clean.name)
            # The media shares this exact number, so the take joins across
            # modalities on (signer, label, variant).
            variant = int(p_clean.stem.split("__")[-1])
            kind = KIND_WITH_LANDMARKS
            if synthetic > 0:
                generate_variants(
                    library.SEQUENCES, noisy, label=slug, signer_id=signer,
                    language=language, fps=self.fps, n=synthetic, jitter=0.20,
                    rng=self.rng)
        else:
            # Own name space: must not consume a landmark take number, or the
            # synthetic `variant // ratio` parent lookup breaks.
            variant = next_media_only_variant(library.MEDIA, language, slug,
                                              signer)
            kind = KIND_MEDIA_ONLY
            synthetic = 0                  # synthetic is landmark-only

        if media is not None:
            written = media.save(
                library.MEDIA, language=language, label=slug,
                label_text=self.current_label, signer=signer, variant=variant,
                kind=kind, nominal_fps=self.fps)
            saved.extend(written.values())

        with self.lock:
            self.session_take_count += 1
            self.last_saved = f"{self.current_label} → {slug}"
        print(f"saved [{self.session_take_count}] {self.current_label} → {slug}: "
              f"{', '.join(saved)} (+{synthetic} synthetic)")
        return True

    # ── drawing (ported from Recorder) ───────────────────────────────
    def _draw_overlay(self, frame: np.ndarray, now: float) -> None:
        color = {"IDLE": (200, 200, 200), "COUNTDOWN": (0, 200, 255),
                 "RECORDING": (0, 0, 255)}[self.state]
        cv2.putText(frame, self.state, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"signer: {self.signer}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"lang: {self.language}", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"saved: {self.session_take_count}", (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        with self.lock:
            target, done, paused = (self.series_target, self.series_done,
                                    self.series_paused)
        if target:
            # Where the signer is looking, so the series is readable without
            # turning to the browser. Take number = the one in progress.
            n = min(done + (0 if self.state == "IDLE" else 1), target)
            msg = (f"PAUSED {done}/{target} - press p to resume" if paused
                   else f"take {n}/{target}  (p = pause)")
            cv2.putText(frame, msg, (15, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 255) if paused else (255, 255, 255), 2,
                        cv2.LINE_AA)
        if self.current_label and self.state == "COUNTDOWN":
            rem = max(0.0, COUNTDOWN_S - (now - self.countdown_start))
            self.overlay.draw(frame, f"{self.current_label}  ({rem:.1f}s)",
                              (15, 160), 36, color)
        elif self.current_label and self.state == "RECORDING":
            rem = max(0.0, self.duration - (now - self.record_start))
            self.overlay.draw(frame, f"REC: {self.current_label}  ({rem:.1f}s)",
                              (15, 160), 36, color)
        elif self.current_label:
            self.overlay.draw(frame, f"last: {self.current_label}",
                              (15, 160), 28, (180, 180, 180))

    def _mannequin_image(self, height: int) -> np.ndarray | None:
        try:
            buf = self.vis.capture_screen_float_buffer(do_render=True)
        except Exception:
            return None
        img = np.asarray(buf)
        if img.size == 0:
            return None
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        width = max(1, int(img.shape[1] * height / img.shape[0]))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _update_mannequins(self, pose: dict, lh: dict, rh: dict) -> None:
        scene_joints: dict[str, np.ndarray] = {}
        name_map = {
            "left_shoulder": "l_shoulder", "right_shoulder": "r_shoulder",
            "left_elbow": "l_elbow", "right_elbow": "r_elbow",
            "left_wrist": "l_wrist", "right_wrist": "r_wrist",
            "left_hip": "l_hip", "right_hip": "r_hip", "nose": "nose",
        }
        for src_name, dst_name in name_map.items():
            if src_name in pose:
                scene_joints[dst_name] = body_to_3d(
                    pose[src_name], self.img_w, self.img_h)
        if "l_shoulder" in scene_joints and "r_shoulder" in scene_joints:
            sw = float(np.linalg.norm(
                scene_joints["l_shoulder"] - scene_joints["r_shoulder"]))
            fwd = np.array([0.0, 0.0, sw * 0.32])
            if "l_wrist" in scene_joints:
                scene_joints["l_wrist"] = scene_joints["l_wrist"] + fwd
            if "r_wrist" in scene_joints:
                scene_joints["r_wrist"] = scene_joints["r_wrist"] + fwd
        scene_lhand = scene_rhand = None
        if "l_wrist" in scene_joints:
            scene_lhand = hand_to_3d(lh, self.img_w, self.img_h,
                                     scene_joints["l_wrist"])
        if "r_wrist" in scene_joints:
            scene_rhand = hand_to_3d(rh, self.img_w, self.img_h,
                                     scene_joints["r_wrist"])
        for mq, body in self.synths:
            sj, sl, sr = retarget_scene(
                scene_joints, scene_lhand, scene_rhand,
                body["sh"], body["ua"], body["fa"], body["hd"], body["xoff"])
            mq.update(sj, sl, sr)
        for mq, _ in self.synths:
            for g in mq.geometries():
                self.vis.update_geometry(g)
