"""Recurrent baselines: GRU, LSTM, and their bidirectional forms.

A GRU already existed, defined inline inside `algo_comparison/run_comparison.py`
where nothing else could reach it — which is why the deep category has only ever
been compared two-wide. These are the same idea, importable, so the deep side of
the comparison is as populated as the classical side.

Bidirectional is the honest default for *offline* scoring: a recorded take is
complete, so there is no reason to hide the second half from the model. It is
the wrong default for **live** recognition, which only ever has the past — a
bidirectional model cannot run on a rolling buffer without seeing the future.
`bidirectional=False` is what a streaming deployment would use, and the gap
between the two is the cost of going live.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .schema import FEATURE_DIM


class SignRNN(nn.Module):
    """GRU / LSTM over the per-frame feature sequence.

    Input  (B, T, F) — same contract as SignTCN.
    Output (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_features: int = FEATURE_DIM,
        hidden: int = 128,
        num_layers: int = 1,
        cell: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            in_features, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d = hidden * (2 if bidirectional else 1)
        # mean+max over time rather than the last hidden state: a sign's
        # identity is spread across the clip, and taking only the final step
        # weights the hands dropping back to rest as heavily as the sign.
        self.head = nn.Sequential(
            nn.LayerNorm(d * 2), nn.Dropout(dropout),
            nn.Linear(d * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                       # (B, T, d)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


def make_gru(num_classes, in_features, **kw):
    return SignRNN(num_classes, in_features, cell="gru",
                   bidirectional=False, **kw)


def make_bigru(num_classes, in_features, **kw):
    return SignRNN(num_classes, in_features, cell="gru",
                   bidirectional=True, **kw)


def make_lstm(num_classes, in_features, **kw):
    return SignRNN(num_classes, in_features, cell="lstm",
                   bidirectional=False, **kw)


def make_bilstm(num_classes, in_features, **kw):
    return SignRNN(num_classes, in_features, cell="lstm",
                   bidirectional=True, **kw)
