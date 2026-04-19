"""
Sign Language Dataset — PyTorch Dataset for loading recorded sequences.

Folder structure expected:
  data/sequences/
    hello/
      hello_001_clean.npy   (60, 51, 3)
      hello_001_raw.npy     (60, 51, 3)
    water/
      water_001_clean.npy
      water_001_raw.npy

Each subdirectory name = class label.
Both _clean and _raw files share the same label → domain randomization.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SignDataset(Dataset):
    """
    Loads all .npy sequence files from a directory of label folders.

    Parameters
    ----------
    data_dir : str or Path
        Path to the sequences root (e.g. 'data/sequences').
    """

    def __init__(self, data_dir: str):
        self.data_dir  = Path(data_dir)
        self.sequences: List[Path] = []
        self.labels:    List[int]  = []
        self.label_map: Dict[str, int] = {}
        self.index_map: Dict[int, str] = {}

        # Scan label directories
        label_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        for idx, label_dir in enumerate(label_dirs):
            label_name = label_dir.name
            self.label_map[label_name] = idx
            self.index_map[idx] = label_name

            npy_files = sorted(label_dir.glob('*.npy'))
            for f in npy_files:
                self.sequences.append(f)
                self.labels.append(idx)

        # Save / update the label mapping
        if len(self.label_map) > 0:
            labels_path = self.data_dir / 'labels.json'
            with open(labels_path, 'w') as f:
                json.dump(self.label_map, f, indent=2)

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = np.load(self.sequences[idx]).astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(seq), label

    def get_label_name(self, idx: int) -> str:
        return self.index_map.get(idx, f'unknown_{idx}')

    # ── Splitting ──

    def split(self,
              train_ratio: float = 0.8,
              val_ratio: float = 0.1
              ) -> Tuple[Subset, Subset, Subset]:
        """
        Deterministic train/val/test split.
        Files are sorted so the split is reproducible.

        Returns (train_subset, val_subset, test_subset)
        """
        n = len(self)
        indices = list(range(n))

        # Shuffle deterministically
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx   = indices[n_train:n_train + n_val]
        test_idx  = indices[n_train + n_val:]

        return (
            Subset(self, train_idx),
            Subset(self, val_idx),
            Subset(self, test_idx),
        )
