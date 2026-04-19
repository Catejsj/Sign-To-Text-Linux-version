"""SignTransformer — the v2 baseline. Spec from doc3_training.docx."""
from __future__ import annotations

import torch
import torch.nn as nn

from .schema import SEQ_LEN, FEATURE_DIM


class SignTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = SEQ_LEN,
        feature_dim: int = FEATURE_DIM,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 60, 144)
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
