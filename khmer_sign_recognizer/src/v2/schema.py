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
    REAL = "real"
    SYNTHETIC = "synthetic"


@dataclass
class SampleMeta:
    """One metadata record per .npy sample. Saved as sibling .json."""
    label: str
    signer_id: str
    source: Source
    fps: int = 30
    variant: int = 0
    notes: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["source"] = self.source.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "SampleMeta":
        d = json.loads(text)
        d["source"] = Source(d["source"])
        return cls(**d)


def sample_paths(root: Path, label: str, signer_id: str, source: Source,
                 variant: int) -> tuple[Path, Path]:
    """Canonical (npy, meta.json) paths for a sample."""
    stem = f"{signer_id}__{source.value}__{variant:04d}"
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{stem}.npy", folder / f"{stem}.json"


def save_sample(root: Path, clip: np.ndarray, meta: SampleMeta,
                variant: Optional[int] = None) -> Path:
    """Save (clip, meta). Returns the .npy path."""
    assert clip.shape == (SEQ_LEN, NUM_JOINTS, NUM_COORDS), (
        f"expected {(SEQ_LEN, NUM_JOINTS, NUM_COORDS)}, got {clip.shape}"
    )
    assert clip.dtype == np.float32, f"expected float32, got {clip.dtype}"
    if variant is not None:
        meta.variant = variant
    npy_path, json_path = sample_paths(root, meta.label, meta.signer_id,
                                       meta.source, meta.variant)
    np.save(npy_path, clip)
    json_path.write_text(meta.to_json())
    return npy_path


def load_sample(npy_path: Path) -> tuple[np.ndarray, SampleMeta]:
    clip = np.load(npy_path).astype(np.float32)
    json_path = npy_path.with_suffix(".json")
    meta = SampleMeta.from_json(json_path.read_text())
    return clip, meta
