"""Video and still-frame capture for a take, alongside (or instead of) landmarks.

Landmarks were the only modality recorded until now. To test whether pixels
add anything, a take can also keep its raw camera frames as a video, as still
images, or both. Pixels matter here because they survive exactly where the
hand tracker fails: in the stored corpus the left hand is missing in about 40%
of frames, and a landmark model gets nothing there while the image still does.

Three rules this module exists to enforce:

1. **Raw frames only.** `LandmarkCapture.read_frame()` returns the camera image
   with the RTMPose skeleton and MediaPipe hands already drawn on it. A model
   trained on that learns to read the drawing, so the "image" stream would be
   the landmark stream in disguise and any multimodal gain would be fake. The
   engine feeds this module `read_raw_frame()` instead.

2. **Never disturb landmark take numbering.** Synthetic copies find their
   parent by `variant // ratio`, which only works while real landmark takes
   are numbered 0, 1, 2 ... with no gaps. So:
     - a take WITH landmarks saves its media under the SAME number as its
       landmark files, `<signer>__real__<NNNN>`, and the two join on
       (signer, label, variant);
     - a take WITHOUT landmarks uses its own name space,
       `<signer>__mediaonly__<NNNN>`, and never consumes a landmark number.
   Getting this wrong reattaches synthetic data to the wrong parent, which is
   the take-split leak this project has already been caught by once.

3. **Nothing reaches git.** Everything lands under `data/media/`, and `data/`,
   `*.mp4` and `*.jpg` are all gitignored.

Only new camera frames are kept. The engine loop runs about twice as fast as
the camera, so without the sequence-number check about half of a take's frames
were exact copies of the one before (measured: 83 of 174 on the first real
take, giving a false 57 fps).

Frames are held as JPEG bytes in memory while recording (roughly 60 KB each,
so even the 8-second cap at 30 fps is about 15 MB) and written once the take
ends. That way the video plays at the frame rate actually achieved, which on a
busy laptop is often below the camera's nominal 30.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

JPEG_QUALITY = 95
KIND_WITH_LANDMARKS = "real"
KIND_MEDIA_ONLY = "mediaonly"


def _variants_in(folder: Path, signer: str, kind: str) -> list[int]:
    nums = []
    if not folder.exists():
        return nums
    for p in folder.glob(f"{signer}__{kind}__*"):
        # 0003.mp4 / 0003.json / 0003_frames  ->  3
        head = p.name.split("__")[-1].split(".")[0].split("_")[0]
        if head.isdigit():
            nums.append(int(head))
    return nums


def next_media_only_variant(media_root: Path, language: str, label: str,
                            signer: str) -> int:
    nums = _variants_in(media_root / language / label, signer, KIND_MEDIA_ONLY)
    return max(nums) + 1 if nums else 0


def delete_take_media(media_root: Path, language: str, label: str,
                      signer: str, variant: int,
                      kind: str = KIND_WITH_LANDMARKS) -> int:
    """Remove the video, frames and sidecar of one take. Returns items removed."""
    folder = media_root / language / label
    stem = f"{signer}__{kind}__{variant:04d}"
    n = 0
    for ext in (".mp4", ".json"):
        f = folder / f"{stem}{ext}"
        if f.exists():
            f.unlink()
            n += 1
    frames = folder / f"{stem}_frames"
    if frames.is_dir():
        shutil.rmtree(frames)
        n += 1
    return n


class MediaTake:
    """Collects one take's raw frames while the engine is RECORDING."""

    def __init__(self, video: bool, frames: bool):
        self.video = video
        self.frames = frames
        self.jpegs: list[bytes] = []
        self.times: list[float] = []
        self.pose: list[bool] = []
        self.lm: list[int | None] = []
        self.last_seq = -1
        self.size: tuple[int, int] | None = None
        self.t0 = time.time()

    def __len__(self) -> int:
        return len(self.jpegs)

    def add(self, frame: np.ndarray, seq: int, lm_index: int | None,
            now: float) -> None:
        """Keep `frame` if it is a new camera frame.

        `seq` is the camera's frame number; a repeat of the last one is
        skipped. `lm_index` is the index of the landmark frame recorded on the
        same tick (None if no pose was found then).
        """
        if seq == self.last_seq:
            return
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        if self.size is None:
            self.size = (frame.shape[1], frame.shape[0])
        self.last_seq = seq
        self.jpegs.append(buf.tobytes())
        self.times.append(now - self.t0)
        self.pose.append(lm_index is not None)
        self.lm.append(lm_index)

    def achieved_fps(self, nominal: float) -> float:
        span = self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0
        if span <= 0:
            return float(nominal)
        return (len(self.times) - 1) / span

    def save(self, media_root: Path, *, language: str, label: str,
             label_text: str, signer: str, variant: int, kind: str,
             nominal_fps: float) -> dict:
        folder = media_root / language / label
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{signer}__{kind}__{variant:04d}"
        fps = self.achieved_fps(nominal_fps)
        written: dict[str, str] = {}

        if self.video and self.jpegs:
            path = folder / f"{stem}.mp4"
            w, h = self.size
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
            for b in self.jpegs:
                vw.write(cv2.imdecode(np.frombuffer(b, np.uint8),
                                      cv2.IMREAD_COLOR))
            vw.release()
            written["video"] = path.name

        if self.frames and self.jpegs:
            fdir = folder / f"{stem}_frames"
            if fdir.exists():          # a reused number must not keep old frames
                shutil.rmtree(fdir)
            fdir.mkdir()
            for i, b in enumerate(self.jpegs):
                (fdir / f"{i:04d}.jpg").write_bytes(b)
            written["frames"] = fdir.name

        meta = {
            "label": label,
            "label_text": label_text,
            "signer_id": signer,
            "language": language,
            "variant": variant,
            "kind": kind,
            "has_landmarks": kind == KIND_WITH_LANDMARKS,
            "n_frames": len(self.jpegs),
            "fps_nominal": nominal_fps,
            "fps_achieved": round(fps, 2),
            "width": self.size[0] if self.size else None,
            "height": self.size[1] if self.size else None,
            "files": written,
            # One entry per saved camera frame: seconds since recording began,
            # whether a body pose was found, and `lm`, the index of the
            # landmark frame recorded at the same moment (null if none).
            # Indices are into the landmark clip BEFORE it was resampled to
            # the stored length. Pose runs on a background thread, so the
            # pairing may be off by about one camera frame in time.
            "t": [round(t, 4) for t in self.times],
            "pose": self.pose,
            "lm": self.lm,
        }
        (folder / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return written
