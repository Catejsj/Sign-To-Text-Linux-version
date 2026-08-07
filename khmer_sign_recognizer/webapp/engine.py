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

try:
    import open3d as o3d                                               # noqa: E402
    from scripts.mannequin_local import (                              # noqa: E402
        Mannequin, retarget_scene, body_to_3d, hand_to_3d,
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

        # ── recording state (owned by tick / main thread) ──
        self.state = "IDLE"
        self.current_label: str | None = None
        self.pending_label: str | None = None
        self.stop_now = False
        self.countdown_start = 0.0
        self.record_start = 0.0
        self.phase_end_at: float | None = None
        self.take_buffer: list[np.ndarray] = []
        self.session_take_count = 0
        self.last_saved: str | None = None
        self.quit_requested = False

        # ── recognition (Recognize mode) ──
        self.recognizer: LiveRecognizer | None = None
        self.recognizer_name: str | None = None
        self.last_prediction: dict | None = None
        self.history: list[dict] = []      # recent stable predictions

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
        self.quit_requested = False
        return True

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
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
            self.pending_label = text

    def stop_take(self) -> None:
        with self.lock:
            if self.state == "RECORDING":
                self.stop_now = True

    def set_signer(self, signer: str) -> None:
        with self.lock:
            self.signer = signer

    def set_language(self, language: str) -> None:
        with self.lock:
            # changing language mid-take would misfile it; only when idle
            if self.state == "IDLE":
                self.language = language

    def set_config(self, mannequin=None, synthetic=None, duration=None,
                   view=None) -> None:
        with self.lock:
            if mannequin is not None:
                self.desired_mannequin = max(0, int(mannequin))
            if synthetic is not None:
                self.synthetic = max(0, int(synthetic))
            if duration is not None:
                self.duration = float(np.clip(duration, 0.5, MAX_RECORD_S))
            if view in ("both", "camera", "mannequin"):
                self.view = view

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

        self._draw_recognition(frame, pred)
        cv2.imshow(CAM_WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            self.request_quit()

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

    def request_quit(self) -> None:
        with self.lock:
            self.quit_requested = True

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
                "config": {
                    "mannequin": self.desired_mannequin,
                    "synthetic": self.synthetic,
                    "duration": self.duration,
                    "view": self.view,
                },
            }

    # ── per-frame update (main thread) ───────────────────────────────
    def tick(self) -> None:
        if not self.running or self.capture is None:
            return

        # apply pending view / mannequin-count changes (o3d must run here)
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

        ret, frame = self.capture.read_frame()
        if not ret or frame is None:
            cv2.waitKey(1)
            return

        with self.capture.result_lock:
            pose = dict(self.capture.latest_pose)
            lh = dict(self.capture.latest_left_hand)
            rh = dict(self.capture.latest_right_hand)
        has_pose = bool(pose)
        now = time.time()

        self._advance_state(pose, lh, rh, now)

        # render the mannequin pane (still driven by the camera pose even when
        # the camera itself is hidden)
        mann = None
        if self.vis is not None and self.current_mannequin > 0:
            if has_pose:
                self._update_mannequins(pose, lh, rh)
            self.vis.poll_events()
            self.vis.update_renderer()
            mann = self._mannequin_image(frame.shape[0])

        show_camera = view != "mannequin"
        if show_camera and mann is not None:
            out = np.hstack([frame, mann])
        elif show_camera:
            out = frame
        elif mann is not None:
            out = mann
        else:
            out = frame   # mannequin-only but none built yet — show camera
        # overlay the status text last, so it is visible whichever pane shows
        self._draw_overlay(out, now)

        cv2.imshow(CAM_WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.request_quit()
        elif key == ord(" "):
            self.stop_take()

    def _advance_state(self, pose, lh, rh, now: float) -> None:
        with self.lock:
            pending = self.pending_label
            stop_now = self.stop_now
            duration = self.duration

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
            self.phase_end_at = now + duration

        if self.state == "RECORDING":
            if pose:
                self.take_buffer.append(
                    frame_from_landmarks(pose, lh, rh, self.img_w, self.img_h))
            elapsed = now - self.record_start
            if stop_now or elapsed >= duration or elapsed >= MAX_RECORD_S:
                self._finalize_take()
                self.state = "IDLE"
                self.phase_end_at = None
                with self.lock:
                    self.stop_now = False

    def _finalize_take(self) -> None:
        if len(self.take_buffer) < 5 or not self.current_label:
            print(f"WARN: only {len(self.take_buffer)} frames — discarded.")
            return
        with self.lock:
            language, signer, synthetic = self.language, self.signer, self.synthetic
        # resolve (or create) the slug for this label text in this language
        slug = library.add_label(language, self.current_label)
        clean = clean_clip_from_frames(self.take_buffer)
        noisy = noisy_clip_from_frames(self.take_buffer)
        p_clean, p_noisy = save_pair(
            library.SEQUENCES, clean, noisy,
            label=slug, signer_id=signer, language=language, fps=self.fps)
        if synthetic > 0:
            generate_variants(
                library.SEQUENCES, noisy, label=slug, signer_id=signer,
                language=language, fps=self.fps, n=synthetic, jitter=0.20,
                rng=self.rng)
        with self.lock:
            self.session_take_count += 1
            self.last_saved = f"{self.current_label} → {slug}"
        print(f"saved [{self.session_take_count}] {self.current_label} → {slug}: "
              f"{p_clean.name} (+{synthetic} synthetic)")

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
