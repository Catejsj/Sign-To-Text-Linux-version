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

from .schema import SampleMeta, Source, View, SEQ_LEN, NUM_JOINTS, NUM_COORDS
from .augment import augment_clip


def discover_samples(root: Path,
                     language: Optional[str] = None) -> list[tuple[Path, SampleMeta]]:
    """Walk data/sequences_v2/<lang>/<label>/ and return every (.npy, meta).

    If `language` is given, only samples from that language folder are returned.
    None = all languages."""
    out = []
    if not root.exists():
        return out
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if language is not None and lang_dir.name != language:
            continue
        for label_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            for npy in sorted(label_dir.glob("*.npy")):
                meta_path = npy.with_suffix(".json")
                if not meta_path.exists():
                    continue
                meta = SampleMeta.from_json(meta_path.read_text(encoding="utf-8"))
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


def synthetic_ratios(
    samples: list[tuple[Path, SampleMeta]],
) -> dict[tuple[str, str], int]:
    """How many synthetic copies exist per real take, per (signer, label).

    Measured rather than assumed: it differs between people, and hard-coding it
    is what let a double `generate_synthetic.py` run group children under the
    wrong parent (see docs/project/PROBLEM_LOG.md §C2).
    """
    real: dict = {}
    synth: dict = {}
    for _p, m in samples:
        if m.view is not View.CLEAN:
            continue
        key = (m.signer_id, m.label)
        d = real if m.source is Source.REAL else synth
        d[key] = d.get(key, 0) + 1
    return {k: max(1, synth.get(k, 0) // n) for k, n in real.items() if n}


def take_id(meta: SampleMeta, ratios: dict[tuple[str, str], int]) -> tuple:
    """The recording a sample came from.

    A take produces several files: a clean view and a noisy view of the same
    movement, plus a synthetic copy of each per generated body. They are all
    the same two seconds of one person signing, so they must never be split
    across train and eval.
    """
    if meta.source is Source.REAL:
        variant = meta.variant
    else:
        variant = meta.variant // ratios.get((meta.signer_id, meta.label), 1)
    return (meta.signer_id, meta.label, variant)


def split_random(
    samples: list[tuple[Path, SampleMeta]],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list]:
    """Hold out a fraction of TAKES — not of samples.

    Splitting by sample leaks completely. Every take contributes a clean view,
    a noisy view and several synthetic copies; shuffling the flat list puts
    some of those in train and the rest in val, so the model is scored on
    warped twins of clips it has just memorised. Measured on the real corpora
    before this was fixed: 100% of validation samples had their own take in
    training (docs/project/PROBLEM_LOG.md §C4).

    Grouping by take makes the number mean what it says. It also drops it
    sharply, which is the point.
    """
    ratios = synthetic_ratios(samples)
    takes = sorted({take_id(m, ratios) for _p, m in samples})
    rng = np.random.default_rng(seed)
    order = np.arange(len(takes))
    rng.shuffle(order)
    n_val = max(1, int(len(takes) * val_frac))
    held = {takes[i] for i in order[:n_val].tolist()}

    train, val = [], []
    for pair in samples:
        _p, meta = pair
        (val if take_id(meta, ratios) in held else train).append(pair)
    return train, val


class SignDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, SampleMeta]],
        label_to_idx: dict[str, int],
        augment: bool = False,
        flip_labels: Optional[set[str]] = None,
        seed: int = 0,
        presence: bool = True,
    ):
        """`presence=True` feeds (60, 146): fabricated hand coordinates zeroed,
        plus two channels saying whether each hand was really detected.

        Worth +15.2 macro-F1 on an unseen signer (57.8 -> 73.0, TCN on
        khmer_var). Set False to reproduce pre-2026-09-02 results, which were
        trained on frozen fill values presented as measurements.

        Note the input width changes with this flag, so a model trained with it
        cannot load weights trained without it — `in_features` must match.
        """
        if not _TORCH:
            raise ImportError("PyTorch required to use SignDataset")
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.flip_labels = flip_labels or set()
        self.presence = presence
        self.rng = np.random.default_rng(seed)

    @property
    def n_features(self) -> int:
        """What to pass as the model's `in_features`."""
        return NUM_JOINTS * NUM_COORDS + (2 if self.presence else 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        npy, meta = self.samples[i]
        clip = np.load(npy).astype(np.float32)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            raise ValueError(f"{npy}: shape {clip.shape} != expected")
        # Presence is read from the RAW clip. Augmentation adds per-joint noise,
        # which would break the "21 identical joints" signature the mask is
        # derived from and silently turn every missing hand into a present one.
        if self.presence:
            from .landmarks import hand_presence, zero_missing
            pres = hand_presence(clip)
            clip = zero_missing(clip, pres)

        if self.augment:
            flip_prob = 0.5 if meta.label in self.flip_labels else 0.0
            clip = augment_clip(clip, flip_prob=flip_prob, rng=self.rng)

        feat = clip.reshape(SEQ_LEN, -1)                       # (60, 144)
        if self.presence:
            feat = np.concatenate([feat, pres], axis=1)        # (60, 146)
        y = self.label_to_idx[meta.label]
        return (torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32)),
                torch.tensor(y, dtype=torch.long))


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
