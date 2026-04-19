"""
Sign Recorder — Buffers frames during recording and saves to .npy files.

Saves two versions per recording (domain randomization):
  {label}_{NNN}_clean.npy  — shoulder-normalized + smoothed
  {label}_{NNN}_raw.npy    — pixel-normalized only
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional
from src.normalizer import FrameNormalizer, TOTAL_JOINTS

logger = logging.getLogger(__name__)


class SignRecorder:
    """
    State machine for recording sign sequences.

    States:  IDLE → RECORDING → IDLE
    """

    def __init__(self, config: Dict):
        rec_config      = config.get('recording', {})
        self.seq_length = rec_config.get('sequence_length', 60)
        self.data_dir   = Path(rec_config.get('data_dir', 'data/sequences'))
        self.save_raw   = rec_config.get('save_raw', True)
        self.save_clean = rec_config.get('save_clean', True)

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Normalizer for clean frames
        self.normalizer = FrameNormalizer(smoothing=True, min_cutoff=1.0, beta=0.05)

        # Raw normalizer (no smoothing)
        self.raw_normalizer = FrameNormalizer(smoothing=False)

        # Recording state
        self.is_recording   = False
        self.current_label  = ""
        self.clean_buffer   = []
        self.raw_buffer     = []
        self.record_start   = 0.0

        # Stats
        self.total_saved = 0

        logger.info(f"SignRecorder ready — saving to {self.data_dir}")
        logger.info(f"Sequence length: {self.seq_length} frames")

    def start_recording(self, label: str):
        """Begin recording frames for the given label."""
        if self.is_recording:
            logger.warning("Already recording! Stop first.")
            return

        self.current_label = label.strip().lower().replace(' ', '_')
        self.clean_buffer  = []
        self.raw_buffer    = []
        self.record_start  = time.time()

        # Reset smoothing between recordings
        self.normalizer.reset()

        self.is_recording = True
        logger.info(f"▶ Recording started: '{self.current_label}'")

    def add_frame(self, pose: Dict, left_hand: Dict, right_hand: Dict,
                  img_w: int = 640, img_h: int = 480):
        """Add one frame of skeleton data to the recording buffer."""
        if not self.is_recording:
            return

        t = time.time()

        # Get both versions from normalizer
        clean, raw = self.normalizer.normalize(
            pose, left_hand, right_hand, img_w, img_h, t
        )

        # Raw version uses separate normalizer (no smoothing state)
        _, raw_frame = self.raw_normalizer.normalize(
            pose, left_hand, right_hand, img_w, img_h, t
        )

        self.clean_buffer.append(clean)
        self.raw_buffer.append(raw_frame)

    def stop_recording(self) -> Optional[Dict]:
        """
        Stop recording and save the sequence.

        Returns dict with saved file paths, or None if nothing to save.
        """
        if not self.is_recording:
            return None

        self.is_recording = False
        n_frames = len(self.clean_buffer)

        if n_frames == 0:
            logger.warning("⚠ No frames recorded — nothing saved.")
            return None

        logger.info(f"⏹ Recording stopped: {n_frames} frames captured")

        # Pad or trim to fixed length
        clean_seq = self._pad_or_trim(self.clean_buffer)
        raw_seq   = self._pad_or_trim(self.raw_buffer)

        # Save
        label_dir = self.data_dir / self.current_label
        label_dir.mkdir(parents=True, exist_ok=True)

        seq_num = self._next_sequence_number(label_dir)
        saved = {}

        if self.save_clean:
            clean_path = label_dir / f"{self.current_label}_{seq_num:03d}_clean.npy"
            np.save(str(clean_path), clean_seq)
            saved['clean'] = str(clean_path)
            logger.info(f"  💾 Saved: {clean_path.name}  shape={clean_seq.shape}")

        if self.save_raw:
            raw_path = label_dir / f"{self.current_label}_{seq_num:03d}_raw.npy"
            np.save(str(raw_path), raw_seq)
            saved['raw'] = str(raw_path)
            logger.info(f"  💾 Saved: {raw_path.name}  shape={raw_seq.shape}")

        self.total_saved += 1
        self.clean_buffer = []
        self.raw_buffer   = []

        return saved

    def get_status(self) -> Dict:
        """Current recorder state."""
        return {
            'is_recording':  self.is_recording,
            'label':         self.current_label,
            'frames':        len(self.clean_buffer),
            'max_frames':    self.seq_length,
            'total_saved':   self.total_saved,
        }

    # ─────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────

    def _pad_or_trim(self, buffer: list) -> np.ndarray:
        """
        Ensure sequence is exactly self.seq_length frames.
        Trim from start (keep latest), pad by repeating last frame.
        """
        frames = np.array(buffer, dtype=np.float32)  # (N, 51, 3)

        if len(frames) > self.seq_length:
            # Keep the last seq_length frames
            frames = frames[-self.seq_length:]
        elif len(frames) < self.seq_length:
            # Pad by repeating last frame
            pad_count = self.seq_length - len(frames)
            last_frame = frames[-1:]  # (1, 51, 3)
            padding = np.repeat(last_frame, pad_count, axis=0)
            frames = np.concatenate([frames, padding], axis=0)

        return frames  # (seq_length, 51, 3)

    def _next_sequence_number(self, label_dir: Path) -> int:
        """Find the next available sequence number for a label."""
        existing = list(label_dir.glob('*.npy'))
        if not existing:
            return 1

        # Extract numbers from filenames like hello_003_clean.npy
        numbers = []
        for f in existing:
            parts = f.stem.split('_')
            for part in parts:
                if part.isdigit():
                    numbers.append(int(part))
                    break

        return max(numbers, default=0) + 1
