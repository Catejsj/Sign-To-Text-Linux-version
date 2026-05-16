"""SignTCN — small Temporal Convolutional Network for skeleton sequences.

Why a TCN instead of a Transformer for SignLink:
  • Strong inductive bias for local temporal patterns (better at small data scale)
  • ~3× fewer parameters than the Transformer baseline
  • Faster training, smaller deployable model
  • Established precedent in skeleton-based action recognition

Architecture:
  Input  (B, 60, 144)            ← skeleton sequence
    ├─ Linear projection 144 → 128
    ├─ 4 × Residual Temporal Conv block
    │     • Conv1d kernel=5, dilation=1,2,4,8
    │     • BatchNorm + GELU + Dropout
    │     • Skip connection
    ├─ Global mean + max pooling over time → (B, 256)
    └─ MLP head 256 → 128 → num_classes
  Output (B, num_classes)

Roughly ~700k parameters at default size.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .schema import SEQ_LEN, FEATURE_DIM


class ResidualTemporalBlock(nn.Module):
    """One residual block: 2× (Conv1d → BN → GELU) with a skip connection.

    The dilated convolution lets the receptive field grow exponentially
    without increasing parameter count: dilations 1,2,4,8 cover ~31 frames
    of context per block.
    """

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation // 2  # 'same' padding
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        return x


class SignTCN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_features: int = FEATURE_DIM,    # 144
        channels: int = 128,
        kernel: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, channels)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(channels, kernel, d, dropout) for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),     # *2 because we concat mean+max pool
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) — B=batch, T=time (60), F=features (144)
        x = self.input_proj(x)            # (B, T, C)
        x = x.transpose(1, 2)             # (B, C, T) for Conv1d
        for block in self.blocks:
            x = block(x)
        # Global mean + max pool over time → richer representation
        mean_pool = x.mean(dim=2)
        max_pool, _ = x.max(dim=2)
        pooled = torch.cat([mean_pool, max_pool], dim=1)   # (B, 2C)
        return self.head(pooled)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
