"""
Sign Language Temporal CNN — Model Architecture

This file defines the neural network structure only.
It is the "car" — the same code runs everywhere.
The "battery" is the .pth weights file that gets swapped.

Architecture:
  Input:  (batch, seq_len, num_joints, 3)  e.g. (batch, 60, 51, 3)
  Reshape to (batch, 153, 60) — treat joint-coords as channels, time as length
  Conv1d along time axis: 153 → 256 → 512 → 256
  Global average pooling → 256
  FC:  256 → 128 → num_classes
"""

import torch
import torch.nn as nn


class SignLanguageCNN(nn.Module):

    def __init__(self, num_classes: int,
                 seq_len: int = 60,
                 num_joints: int = 51,
                 coords: int = 3):
        super().__init__()

        self.seq_len    = seq_len
        self.num_joints = num_joints
        self.coords     = coords
        in_channels     = num_joints * coords  # 153

        # ── Temporal convolution layers ──
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ── Classifier head ──
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, num_joints, coords)
            e.g. (32, 60, 51, 3)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        batch = x.size(0)

        # Flatten joints × coords → channels
        x = x.view(batch, self.seq_len, -1)   # (B, 60, 153)
        x = x.permute(0, 2, 1)                 # (B, 153, 60)

        # Temporal convolutions
        x = self.conv_layers(x)                 # (B, 256, 60)

        # Global average pooling over time
        x = x.mean(dim=2)                       # (B, 256)

        # Classification
        x = self.classifier(x)                  # (B, num_classes)
        return x

    # ── Convenience methods for weight management ──

    def save_weights(self, path: str):
        """Save model weights + metadata."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes':      self.classifier[-1].out_features,
            'seq_len':          self.seq_len,
            'num_joints':       self.num_joints,
            'coords':           self.coords,
        }, path)

    @classmethod
    def load_from_weights(cls, path: str, device: str = 'cpu') -> 'SignLanguageCNN':
        """Load a model from a weights file (auto-detects num_classes)."""
        checkpoint   = torch.load(path, map_location=device, weights_only=True)
        num_classes  = checkpoint['num_classes']
        seq_len      = checkpoint.get('seq_len', 60)
        num_joints   = checkpoint.get('num_joints', 51)
        coords       = checkpoint.get('coords', 3)

        model = cls(num_classes, seq_len, num_joints, coords)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
