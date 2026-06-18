"""Load v2 samples as flat feature vectors for classical-ML baselines.

No PyTorch. Just numpy + scikit-learn-friendly arrays. Every teammate's
algorithm loads data through here so the 8-way comparison is apples-to-apples.

The AUTSL importer encodes the official train/val/test split in each
signer id (``autsl_p000_train``, ``autsl_p012_val``, ...). We read the
split from that suffix. Locally-recorded KSL data has no suffix, so it
falls into "train" by default.

THE ONE SAFETY RULE
-------------------
Evaluation splits (val / test) are ALWAYS loaded as REAL samples only,
no matter what ``source_mode`` you ask for. You never grade the model on
synthetic cards — that would measure performance on fake data, not real
signers. ``source_mode`` only ever affects the TRAIN split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from .schema import Source, View, SEQ_LEN, NUM_JOINTS, NUM_COORDS
from .dataset import discover_samples

ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "sequences_v2"

VALID_SPLITS = ("train", "val", "test")
VALID_SOURCE_MODES = ("real", "synthetic", "both")
VALID_FEATURES = ("summary", "flat")


def split_of(signer_id: str) -> str:
    """Read train/val/test from the signer-id suffix; default 'train'."""
    for s in VALID_SPLITS:
        if signer_id.endswith("_" + s):
            return s
    return "train"


def _featurize(clip: np.ndarray, feature_mode: str) -> np.ndarray:
    """(60, 48, 3) clip -> 1-D feature vector.

    flat    : every number, 60*48*3 = 8640 features. Faithful but high-dim.
    summary : per-(joint,coord) mean/std/min/max over time, 48*3*4 = 576
              features. Lower-dim, usually better for classical models.
    """
    if feature_mode == "flat":
        return clip.reshape(-1).astype(np.float32)
    return np.concatenate([
        clip.mean(axis=0).reshape(-1),
        clip.std(axis=0).reshape(-1),
        clip.min(axis=0).reshape(-1),
        clip.max(axis=0).reshape(-1),
    ]).astype(np.float32)


def load_split(
    split: str,
    language: str,
    source_mode: str = "real",
    root: Optional[Path] = None,
    view: View = View.CLEAN,
    feature_mode: str = "summary",
    label_to_idx: Optional[dict[str, int]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Return (X, y, signer_ids, label_to_idx) for one split.

    X            : (N, D) float32 feature matrix
    y            : (N,)   int class indices
    signer_ids   : list[str], len N — for per-signer scoring
    label_to_idx : the label->index map (pass it back in for other splits
                   so val/test use the same indexing as train)
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if source_mode not in VALID_SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {VALID_SOURCE_MODES}")
    if feature_mode not in VALID_FEATURES:
        raise ValueError(f"feature_mode must be one of {VALID_FEATURES}")

    root = Path(root) if root is not None else ROOT_DEFAULT

    # SAFETY: only the train split may include synthetic. Eval = real only.
    effective_mode = source_mode if split == "train" else "real"
    allowed_sources = {
        "real": {Source.REAL},
        "synthetic": {Source.SYNTHETIC},
        "both": {Source.REAL, Source.SYNTHETIC},
    }[effective_mode]

    all_samples = discover_samples(root, language=language)

    # Build a stable label index from ALL real labels if not provided, so
    # train/val/test agree even if a rare label is missing from one split.
    if label_to_idx is None:
        labels = sorted({m.label for _, m in all_samples})
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

    X, y, signers = [], [], []
    for npy, meta in all_samples:
        if split_of(meta.signer_id) != split:
            continue
        if meta.source not in allowed_sources:
            continue
        if meta.view != view:
            continue
        clip = np.load(npy).astype(np.float32)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            continue
        X.append(_featurize(clip, feature_mode))
        y.append(label_to_idx[meta.label])
        signers.append(meta.signer_id)

    if not X:
        raise RuntimeError(
            f"no samples for split={split!r} language={language!r} "
            f"source={effective_mode!r} view={view.value!r} under {root}"
        )
    return (np.stack(X), np.asarray(y, dtype=np.int64), signers, label_to_idx)
