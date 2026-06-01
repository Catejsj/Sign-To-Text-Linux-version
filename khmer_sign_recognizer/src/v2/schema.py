"""Data contract for v2. Every file in v2 imports from here."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import json
import numpy as np


SEQ_LEN = 60
NUM_BODY = 6
NUM_HAND = 21
NUM_JOINTS = NUM_BODY + 2 * NUM_HAND  # 48
NUM_COORDS = 3
FEATURE_DIM = NUM_JOINTS * NUM_COORDS  # 144

# Body joint indices (0..5)
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW, R_ELBOW = 2, 3
L_WRIST, R_WRIST = 4, 5

# Hand slices
LEFT_HAND = slice(6, 27)
RIGHT_HAND = slice(27, 48)


class Source(str, Enum):
    """Where a sample's landmarks came from.

    REAL       — MediaPipe + RTMPose on a webcam frame of an actual signer.
    SYNTHETIC  — generated from a 3D animation rig (no live input).
    MANNEQUIN  — MediaPipe re-detection of the rendered VRM avatar that was
                 itself driven by a real webcam in the same session. Paired
                 with a REAL take from the same recording window for the
                 ablation experiment (real-only vs. real+mannequin).
    """
    REAL = "real"
    SYNTHETIC = "synthetic"
    MANNEQUIN = "mannequin"


class View(str, Enum):
    """Two parallel renderings of the same recording.

    CLEAN: shoulder-anchored, scaled by shoulder width — signer-invariant.
    NOISY: raw [0, 1] image-space coords — preserves position/scale variation.

    Both are saved per take so the model sees the same motion under two
    different distributions. Cheap built-in domain randomization.
    """
    CLEAN = "clean"
    NOISY = "noisy"


@dataclass
class SampleMeta:
    label: str
    signer_id: str
    source: Source
    language: str = "khmer"      # ASCII language code: khmer, english, asl, ...
    view: View = View.CLEAN
    fps: int = 30
    variant: int = 0
    notes: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["source"] = self.source.value
        d["view"] = self.view.value
        return json.dumps(d, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "SampleMeta":
        d = json.loads(text)
        d["source"] = Source(d["source"])
        d["view"] = View(d.get("view", "clean"))
        d.setdefault("language", "khmer")    # back-compat: pre-language data
        return cls(**d)


def sample_paths(root: Path, label: str, signer_id: str, source: Source,
                 view: View, variant: int,
                 language: str = "khmer") -> tuple[Path, Path]:
    stem = f"{signer_id}__{source.value}__{view.value}__{variant:04d}"
    folder = root / language / label
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}.npy", folder / f"{stem}.json"


def next_variant(root: Path, label: str, signer_id: str,
                 source: Source, view: View,
                 language: str = "khmer") -> int:
    """Find the next free variant index for (label, signer, source, view)
    within a language folder."""
    folder = root / language / label
    if not folder.exists():
        return 0
    pattern = f"{signer_id}__{source.value}__{view.value}__*.npy"
    nums = []
    for p in folder.glob(pattern):
        try:
            nums.append(int(p.stem.split("__")[-1]))
        except (ValueError, IndexError):
            pass
    return max(nums) + 1 if nums else 0


def save_sample(root: Path, clip: np.ndarray, meta: SampleMeta,
                variant: Optional[int] = None) -> Path:
    assert clip.shape == (SEQ_LEN, NUM_JOINTS, NUM_COORDS), (
        f"expected {(SEQ_LEN, NUM_JOINTS, NUM_COORDS)}, got {clip.shape}"
    )
    assert clip.dtype == np.float32, f"expected float32, got {clip.dtype}"
    if variant is not None:
        meta.variant = variant
    npy_path, json_path = sample_paths(
        root, meta.label, meta.signer_id, meta.source, meta.view, meta.variant,
        language=meta.language,
    )
    np.save(npy_path, clip)
    json_path.write_text(meta.to_json(), encoding="utf-8")
    return npy_path


def save_pair(root: Path, clean_clip: np.ndarray, noisy_clip: np.ndarray,
              label: str, signer_id: str, fps: int = 30,
              notes: str = "", language: str = "khmer") -> tuple[Path, Path]:
    """Save the same take in both views, sharing a variant index."""
    variant = max(
        next_variant(root, label, signer_id, Source.REAL, View.CLEAN, language),
        next_variant(root, label, signer_id, Source.REAL, View.NOISY, language),
    )
    clean_meta = SampleMeta(label=label, signer_id=signer_id,
                            source=Source.REAL, view=View.CLEAN,
                            language=language, fps=fps,
                            variant=variant, notes=notes)
    noisy_meta = SampleMeta(label=label, signer_id=signer_id,
                            source=Source.REAL, view=View.NOISY,
                            language=language, fps=fps,
                            variant=variant, notes=notes)
    p1 = save_sample(root, clean_clip, clean_meta)
    p2 = save_sample(root, noisy_clip, noisy_meta)
    return p1, p2


def load_sample(npy_path: Path) -> tuple[np.ndarray, SampleMeta]:
    clip = np.load(npy_path).astype(np.float32)
    json_path = npy_path.with_suffix(".json")
    meta = SampleMeta.from_json(json_path.read_text(encoding="utf-8"))
    return clip, meta
