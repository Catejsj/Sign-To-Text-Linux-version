"""
Windows Layer - Dual Pipeline
RTMPose: body tracking (GPU)
MediaPipe: hands tracking (original working version)
"""

import cv2
import mediapipe as mp
import numpy as np
from rtmlib import Wholebody, draw_skeleton
from typing import Dict, Optional, Tuple
import logging
import time
import os
import sys
from threading import Thread, Lock
from queue import Queue, Empty

logger = logging.getLogger(__name__)


def _add_nvidia_dlls():
    venv_path = os.path.dirname(sys.executable)
    site_packages = os.path.join(os.path.dirname(venv_path), 'Lib', 'site-packages')
    nvidia_bins = [
        os.path.join(site_packages, 'nvidia', 'cublas', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cudnn', 'bin'),
        os.path.join(site_packages, 'nvidia', 'cuda_runtime', 'bin'),
    ]
    for path in nvidia_bins:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

_add_nvidia_dlls()


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.min_cutoff, dt)
        dx_hat = edx * dx + (1 - edx) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        ex = self._alpha(cutoff, dt)
        x_hat = ex * x + (1 - ex) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _alpha(self, cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)


class LandmarkCapture:

    # RTMPose COCO wholebody body indices
    BODY_INDICES = {
        'nose':           0,
        'left_shoulder':  5,
        'right_shoulder': 6,
        'left_elbow':     7,
        'right_elbow':    8,
        'left_wrist':     9,
        'right_wrist':    10,
        'left_hip':       11,
        'right_hip':      12,
    }

    LEFT_HAND_START  = 91
    RIGHT_HAND_START = 112

    def __init__(self, config: Dict):
        self.config     = config
        self.cap_config = config['capture']
        self.mp_config  = config['mediapipe']

        # ── Shared camera state ──
        self.frame       = None
        self.frame_count = 0
        self.is_running  = False
        self.cap         = None
        self.lock        = Lock()

        # ── RTMPose (Pipeline A - body/arms on GPU) ──
        logger.info("Loading RTMPose Wholebody model...")
        self.wholebody = Wholebody(
            to_openpose=False,
            mode='balanced',
            backend='onnxruntime',
            device='cuda'
        )
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.wholebody(dummy)
        logger.info("RTMPose ready")

        # RTMPose smoothing
        self.rtm_filters: Dict[str, OneEuroFilter] = {}
        for name in self.BODY_INDICES:
            for axis in ('x', 'y'):
                self.rtm_filters[f'{name}_{axis}'] = OneEuroFilter(
                    min_cutoff=1.0, beta=0.05
                )

        # RTMPose overlay canvas (skeleton lines on black)
        self.rtm_canvas     = None
        self.rtm_canvas_lock = Lock()

        # ── MediaPipe (Pipeline B - hands, ORIGINAL working version) ──
        self.mp_holistic_mod = mp.solutions.holistic
        self.holistic = self.mp_holistic_mod.Holistic(
            model_complexity=self.mp_config['model_complexity'],
            min_detection_confidence=self.mp_config['min_detection_confidence'],
            min_tracking_confidence=self.mp_config['min_tracking_confidence'],
            enable_segmentation=self.mp_config['enable_segmentation']
        )

        # Standalone hands as fallback (same as original)
        self.mp_hands_mod = mp.solutions.hands
        self.hands = self.mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # MediaPipe overlay canvas
        self.mp_canvas      = None
        self.mp_canvas_lock = Lock()
        self.mp_canvas_ttl  = 0
        self.MP_CANVAS_MAX_TTL = 5  # freeze for 5 frames then clear

        # MediaPipe smoothing filters
        self.mp_filters: Dict[str, OneEuroFilter] = {}
        for hand in ('left', 'right'):
            for idx in range(21):
                for axis in ('x', 'y', 'z'):
                    self.mp_filters[f'{hand}_{idx}_{axis}'] = OneEuroFilter(
                        min_cutoff=1.5, beta=0.07
                    )

        # ── Merged results ──
        self.latest_pose       = {}
        self.latest_left_hand  = {}
        self.latest_right_hand = {}
        self.result_lock       = Lock()

        # Queues for merger
        self.rtm_queue: Queue = Queue(maxsize=2)
        self.mp_queue:  Queue = Queue(maxsize=2)

        self.last_detection_time = None

        logger.info("Dual pipeline initialized")

    # ─────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────

    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(
                self.cap_config['camera_id'], cv2.CAP_DSHOW
            )
            if not self.cap.isOpened():
                logger.warning("DSHOW failed, trying default...")
                self.cap = cv2.VideoCapture(self.cap_config['camera_id'])

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cap_config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_config['height'])
            self.cap.set(cv2.CAP_PROP_FPS,          self.cap_config['fps'])

            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False

            ret, self.frame = self.cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                return False

            self.is_running = True

            Thread(target=self._camera_thread,    daemon=True).start()
            Thread(target=self._rtmpose_thread,   daemon=True).start()
            Thread(target=self._mediapipe_thread, daemon=True).start()

            logger.info("All threads started")
            return True

        except Exception as e:
            logger.error(f"Error starting: {e}")
            return False

    def _camera_thread(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                if self.cap_config.get('flip_horizontal', False):
                    frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    # ─────────────────────────────────────────
    #  Pipeline A - RTMPose body thread
    # ─────────────────────────────────────────

    def _rtmpose_thread(self):
        while self.is_running:
            with self.lock:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.frame.copy()

            t = time.time()
            keypoints, scores = self.wholebody(frame)

            if keypoints is not None and len(keypoints) > 0:
                kp = keypoints[0]
                sc = scores[0]

                # Draw skeleton on BLACK canvas so it composites cleanly
                canvas = np.zeros_like(frame)
                draw_skeleton(canvas, keypoints, scores, kpt_thr=0.3)

                with self.rtm_canvas_lock:
                    self.rtm_canvas = canvas

                # Extract body landmarks with smoothing
                pose = {}
                for name, idx in self.BODY_INDICES.items():
                    if idx >= len(kp) or sc[idx] < 0.3:
                        continue
                    x = self.rtm_filters[f'{name}_x'](float(kp[idx][0]), t)
                    y = self.rtm_filters[f'{name}_y'](float(kp[idx][1]), t)
                    pose[name] = {
                        'x': x, 'y': y, 'z': 0.0,
                        'visibility': float(sc[idx])
                    }

                with self.result_lock:
                    self.latest_pose = pose

                try:
                    self.rtm_queue.put_nowait(
                        {'pose': pose, 'timestamp': t}
                    )
                except Exception:
                    pass

    # ─────────────────────────────────────────
    #  Pipeline B - MediaPipe hands thread
    #  (original working logic preserved exactly)
    # ─────────────────────────────────────────

    def _mediapipe_thread(self):
        while self.is_running:
            with self.lock:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.holistic.process(rgb)
            rgb.flags.writeable = True

            t = time.time()

            # Drawing specs (same as original)
            hand_spec = self.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2
            )
            hand_conn_spec = self.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            )
            fallback_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 255), thickness=2, circle_radius=2
            )
            fallback_conn_spec = self.mp_drawing.DrawingSpec(
                color=(0, 255, 255), thickness=2
            )

            # Get hands from holistic
            left_lm  = results.left_hand_landmarks
            right_lm = results.right_hand_landmarks
            left_is_fallback  = False
            right_is_fallback = False

            # Fallback to standalone hands if holistic missed either
            if not left_lm or not right_lm:
                hands_results = self.hands.process(rgb)
                if hands_results.multi_hand_landmarks and \
                   hands_results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        hands_results.multi_hand_landmarks,
                        hands_results.multi_handedness
                    ):
                        label = handedness.classification[0].label
                        if label == 'Left' and not left_lm:
                            left_lm = hand_landmarks
                            left_is_fallback = True
                        elif label == 'Right' and not right_lm:
                            right_lm = hand_landmarks
                            right_is_fallback = True

            # Draw on BLACK canvas (not on camera frame)
            canvas = np.zeros_like(frame)

            if left_lm:
                spec  = fallback_spec      if left_is_fallback  else hand_spec
                cspec = fallback_conn_spec if left_is_fallback  else hand_conn_spec
                self.mp_drawing.draw_landmarks(
                    canvas, left_lm,
                    self.mp_hands_mod.HAND_CONNECTIONS,
                    landmark_drawing_spec=spec,
                    connection_drawing_spec=cspec
                )

            if right_lm:
                spec  = fallback_spec      if right_is_fallback else hand_spec
                cspec = fallback_conn_spec if right_is_fallback else hand_conn_spec
                self.mp_drawing.draw_landmarks(
                    canvas, right_lm,
                    self.mp_hands_mod.HAND_CONNECTIONS,
                    landmark_drawing_spec=spec,
                    connection_drawing_spec=cspec
                )

            # Only update canvas when we actually detected something
            if left_lm or right_lm:
                with self.mp_canvas_lock:
                    self.mp_canvas     = canvas
                    self.mp_canvas_ttl = self.MP_CANVAS_MAX_TTL
            else:
                # Count down - clear canvas after a few frames so no ghost
                with self.mp_canvas_lock:
                    self.mp_canvas_ttl -= 1
                    if self.mp_canvas_ttl <= 0:
                        self.mp_canvas     = None
                        self.mp_canvas_ttl = 0

            # Extract hand data (no visibility filter - original working version)
            left_hand  = self._extract_hand(left_lm)
            right_hand = self._extract_hand(right_lm)

            with self.result_lock:
                self.latest_left_hand  = left_hand
                self.latest_right_hand = right_hand
            self.last_detection_time = t

            try:
                self.mp_queue.put_nowait({
                    'left_hand':  left_hand,
                    'right_hand': right_hand,
                    'timestamp':  t
                })
            except Exception:
                pass

            # Cap MediaPipe to ~20fps — it runs on CPU so uncapped it
            # starves RTMPose and the camera thread
            time.sleep(0.05)

    # ─────────────────────────────────────────
    #  Display - composites both canvases
    # ─────────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            base = self.frame.copy()

        # Add RTMPose skeleton layer
        with self.rtm_canvas_lock:
            if self.rtm_canvas is not None:
                base = cv2.add(base, self.rtm_canvas)

        # Add MediaPipe hand layer on top
        with self.mp_canvas_lock:
            if self.mp_canvas is not None:
                base = cv2.add(base, self.mp_canvas)

        return True, base

    # ─────────────────────────────────────────
    #  Data extraction (original working versions)
    # ─────────────────────────────────────────

    def process_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """Wait for both pipelines then merge"""
        try:
            rtm_data = self.rtm_queue.get(timeout=0.1)
        except Empty:
            return None

        try:
            mp_data    = self.mp_queue.get(timeout=0.05)
            left_hand  = mp_data['left_hand']
            right_hand = mp_data['right_hand']
        except Empty:
            left_hand  = self.latest_left_hand
            right_hand = self.latest_right_hand

        pose = rtm_data['pose']
        if len(pose) < 4:
            return None

        self.frame_count += 1
        h = self.cap_config['height']
        w = self.cap_config['width']

        return {
            'frame_id':     self.frame_count,
            'timestamp':    rtm_data['timestamp'],
            'pose':         pose,
            'left_hand':    left_hand,
            'right_hand':   right_hand,
            'image_width':  w,
            'image_height': h,
            'source':       'dual_pipeline'
        }

    def _extract_hand(self, hand_landmarks) -> Dict:
        """Original working extraction - no visibility filter"""
        if not hand_landmarks:
            return {}
        landmarks = {}
        for idx, lm in enumerate(hand_landmarks.landmark):
            landmarks[str(idx)] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            }
        return landmarks

    def _extract_pose(self, pose_landmarks) -> Dict:
        if not pose_landmarks:
            return {}
        landmarks = {}
        for idx in range(25):
            if idx < len(pose_landmarks.landmark):
                lm = pose_landmarks.landmark[idx]
                if lm.visibility < 0.5:
                    continue
                landmarks[idx] = {
                    'x': lm.x, 'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
        return landmarks

    def get_stats(self) -> Dict:
        return {
            'total_frames':   self.frame_count,
            'is_running':     self.is_running,
            'last_detection': self.last_detection_time
        }

    def stop(self):
        self.is_running = False
        time.sleep(0.3)
        if self.cap:
            self.cap.release()
        self.holistic.close()
        self.hands.close()
        logger.info("Dual pipeline stopped")