"""v2 dataset. Reads (.npy, .json) pairs; supports leave-one-signer-out splits."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH = True
except ImportError:
    _TORCH = False
    Dataset = object  # type: ignore

from .schema import SampleMeta, Source, SEQ_LEN, NUM_JOINTS, NUM_COORDS
from .augment import augment_clip


def discover_samples(root: Path) -> list[tuple[Path, SampleMeta]]:
    """Walk data/sequences_v2 and return every (.npy, meta) pair."""
    out = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for npy in sorted(label_dir.glob("*.npy")):
            meta_path = npy.with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = SampleMeta.from_json(meta_path.read_text())
            out.append((npy, meta))
    return out


def build_label_index(samples: list[tuple[Path, SampleMeta]]) -> dict[str, int]:
    labels = sorted({m.label for _, m in samples})
    return {lab: i for i, lab in enumerate(labels)}


def split_leave_one_signer_out(
    samples: list[tuple[Path, SampleMeta]],
    held_out_signer: str,
) -> tuple[list, list]:
    train, val = [], []
    for pair in samples:
        _, meta = pair
        (val if meta.signer_id == held_out_signer else train).append(pair)
    return train, val


def split_random(
    samples: list[tuple[Path, SampleMeta]],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(len(samples) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


class SignDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, SampleMeta]],
        label_to_idx: dict[str, int],
        augment: bool = False,
        flip_labels: Optional[set[str]] = None,
        seed: int = 0,
    ):
        if not _TORCH:
            raise ImportError("PyTorch required to use SignDataset")
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.flip_labels = flip_labels or set()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        npy, meta = self.samples[i]
        clip = np.load(npy).astype(np.float32)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            raise ValueError(f"{npy}: shape {clip.shape} != expected")
        if self.augment:
            flip_prob = 0.5 if meta.label in self.flip_labels else 0.0
            clip = augment_clip(clip, flip_prob=flip_prob, rng=self.rng)
        feat = clip.reshape(SEQ_LEN, -1)  # (60, 144)
        y = self.label_to_idx[meta.label]
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long)


def source_stats(samples: list[tuple[Path, SampleMeta]]) -> dict:
    # Dynamic dict so new Source values (e.g. MANNEQUIN) get counted
    # without having to edit this function each time.
    by_source: dict[str, int] = {}
    by_signer: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for _, m in samples:
        by_source[m.source.value] = by_source.get(m.source.value, 0) + 1
        by_signer[m.signer_id] = by_signer.get(m.signer_id, 0) + 1
        by_label[m.label] = by_label.get(m.label, 0) + 1
    return {"total": len(samples), "by_source": by_source,
            "by_signer": by_signer, "by_label": by_label}
