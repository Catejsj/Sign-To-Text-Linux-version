"""PyTorch models wrapped in a scikit-learn-style fit/predict interface.

Kept separate so the rest of the lab runs with no PyTorch installed — these are
only imported when someone actually selects `gru` or `cnn`.
"""
from __future__ import annotations

import numpy as np


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class _TorchBase:
    """Shared training loop. Subclasses build the network in _build()."""

    def __init__(self, epochs=25, lr=2e-3, batch_size=256, seed=0):
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.model = None

    def _build(self, n_classes):
        raise NotImplementedError

    def _shape(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        dev = _device()
        n_classes = int(np.max(y)) + 1
        self.model = self._build(n_classes).to(dev)

        xt = torch.tensor(self._shape(X), dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.long, device=dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                crit(self.model(xt[idx]), yt[idx]).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        dev = _device()
        self.model.eval()
        xe = torch.tensor(self._shape(X), dtype=torch.float32, device=dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(xe), 1024):
                out.append(self.model(xe[i:i + 1024]).argmax(-1).cpu().numpy())
        return np.concatenate(out)


class GRUClassifier(_TorchBase):
    """Treats each 28x28 image as a sequence of 28 rows."""

    def __init__(self, hidden=128, **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def _shape(self, X):
        return X.reshape(-1, 28, 28)

    def _build(self, n_classes):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden, n_cls):
                super().__init__()
                self.gru = nn.GRU(28, hidden, batch_first=True, bidirectional=True)
                self.head = nn.Sequential(nn.Dropout(0.3),
                                          nn.Linear(hidden * 2, n_cls))

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out.mean(1))

        return Net(self.hidden, n_classes)


class CNNClassifier(_TorchBase):
    """Small 2-block CNN — uses the spatial structure the others ignore."""

    def _shape(self, X):
        return X.reshape(-1, 1, 28, 28)

    def _build(self, n_classes):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
