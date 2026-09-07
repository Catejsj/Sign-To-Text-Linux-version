"""Spatial-Temporal Graph Convolutional Network over the hand/body skeleton.

STATUS: BUILT, MEASURED ACROSS ALL SEVEN SIGNERS, **NOT ADOPTED**. The TCN
remains the model to use. Measured 2026-09-07, mean over 7 held-out signers:

    model    complete    40%     60%   flicker   hello/thanks
    tcn         93.25   64.04   82.80     0.88       6.14
    stgcn       92.00   58.77   78.95     0.83       6.57

ST-GCN is better in only 1/7 folds on complete accuracy and 2/7 at 40%. Across
every fold it made 46 hello/thanks errors against the TCN's 43 — worse on the
metric it was built to fix.

**The architecture is correct**: shuffling the joint order changes its output,
which no other model here does, so it genuinely uses the graph. The hypothesis
was that ជម្រាប់សួរ and អរគុណ differ in HANDSHAPE, which a flat vector
represents worst and a graph convolution represents directly. That hypothesis
appears to be wrong — if handshape separated them, this would have found it.

ONE PROPERTY WORTH REMEMBERING
------------------------------
Its errors are much flatter than the TCN's:

    TCN pair errors   0  1  1  2  |  8   8  23
    ST-GCN            8  3  7  2  |  4   8  14

It does not nail the folds the TCN already handles, and degrades far less on
the ones it does not — 23 errors down to 14 on the weakest signer, 8 down to 4
on another. If worst-case behaviour ever matters more than the average, or if
recording quality drops, this is the better-shaped model. For the current goal
it is not.

Kept rather than deleted because the measurement is the useful part; see
docs/project/PROBLEM_LOG.md S. Nothing imports it.

WHY THIS EXISTS
---------------
Every other model here reads a frame as a flat vector of 144 numbers. Nothing
tells it that joint 7 hangs off joint 6, so it must infer the hand's anatomy
from a few hundred examples. `bones.py` approximates that by handing the model
parent-relative vectors, which helped a lot (+8.4 unseen-signer) — but it is
still a bag of numbers, just a better-chosen one.

ST-GCN encodes the skeleton as an actual graph. Each layer aggregates a joint
with its *neighbours*, so a fingertip is computed from the knuckle it attaches
to, and weights are shared across the skeleton rather than learned separately
per slot. Shuffling the joint order would now change the model, which is the
point.

WHY IT MIGHT HELP THE ONE PAIR THAT MATTERS
-------------------------------------------
ជម្រាប់សួរ and អរគុណ sit 14.97 apart where the next-closest pair is 43.76
(PROBLEM_LOG R.2) — three times closer than anything else. They converge in
trajectory, so what separates them is more likely *handshape*: the spatial
configuration of fingers relative to each other at a moment in time.

That is exactly what a flat vector represents worst and a graph convolution
represents best. Temporal models can only see handshape as a pattern of
correlated coordinates; a spatial graph conv computes it directly.

This is a hypothesis, not a claim. Measure it (see R.4 for how a confirmed
mechanism can still be the wrong change).

PARTITIONING
------------
Three partitions per the original ST-GCN, distance-based:
    identity  the joint itself
    inward    its parent, toward the body centre
    outward   its children, toward the extremities
Each gets its own weight matrix; a layer is their sum. That lets "move toward
the wrist" and "move away from the wrist" be learned as different operations,
which a single symmetric adjacency cannot express.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .bones import PARENT
from .schema import NUM_JOINTS


def build_adjacency() -> torch.Tensor:
    """(3, V, V) normalised adjacency: identity, inward, outward.

    Row-normalised rather than symmetric — a joint with five children should
    not dominate one with one child purely by degree.
    """
    V = NUM_JOINTS
    ident = np.eye(V, dtype=np.float32)
    inward = np.zeros((V, V), dtype=np.float32)
    outward = np.zeros((V, V), dtype=np.float32)
    for child, parent in enumerate(PARENT):
        if parent < 0:
            continue
        inward[child, parent] = 1.0        # child gathers from its parent
        outward[parent, child] = 1.0       # parent gathers from its children

    def norm(a):
        deg = a.sum(axis=1, keepdims=True)
        return a / np.maximum(deg, 1.0)

    return torch.from_numpy(np.stack([ident, norm(inward), norm(outward)]))


class STGCNBlock(nn.Module):
    """One spatial graph conv followed by one temporal conv, with a residual."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 stride: int = 1, kernel: int = 9, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("A", A)
        K = A.shape[0]
        # one weight matrix per partition, applied as a 1x1 conv
        self.gconv = nn.Conv2d(in_ch, out_ch * K, kernel_size=1)
        self.K, self.out_ch = K, out_ch
        # edge importance: let the network learn that some bones matter more
        self.edge_weight = nn.Parameter(torch.ones(A.shape))
        self.bn_s = nn.BatchNorm2d(out_ch)

        pad = ((kernel - 1) // 2, 0)
        self.tconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (kernel, 1), (stride, 1), pad),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.act = nn.ReLU()
        self.residual = (nn.Identity() if in_ch == out_ch and stride == 1
                         else nn.Sequential(
                             nn.Conv2d(in_ch, out_ch, 1, (stride, 1)),
                             nn.BatchNorm2d(out_ch)))

    def forward(self, x):                       # x: (N, C, T, V)
        res = self.residual(x)
        n, _, t, v = x.shape
        y = self.gconv(x).view(n, self.K, self.out_ch, t, v)
        # aggregate each partition over its neighbours, then sum the partitions
        A = self.A * self.edge_weight
        y = torch.einsum("nkctv,kvw->nctw", y, A).contiguous()
        y = self.act(self.bn_s(y))
        return self.act(self.tconv(y) + res)


class SignSTGCN(nn.Module):
    """Input (N, T, V, C) — the natural shape of a landmark clip.

    `in_channels` is per-JOINT, not per-frame: 3 for raw xyz, 7 for
    xyz + bone direction + visibility (what `stgcn_input` produces).
    """

    def __init__(self, num_classes: int, in_channels: int = 7,
                 base: int = 64, dropout: float = 0.2):
        super().__init__()
        A = build_adjacency()
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, base, A, dropout=dropout),
            STGCNBlock(base, base, A, dropout=dropout),
            STGCNBlock(base, base * 2, A, stride=2, dropout=dropout),
            STGCNBlock(base * 2, base * 2, A, dropout=dropout),
            STGCNBlock(base * 2, base * 4, A, stride=2, dropout=dropout),
        ])
        # mean+max over time AND joints, matching how SignTCN pools time
        self.head = nn.Sequential(
            nn.Linear(base * 8, base * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(base * 2, num_classes))

    def forward(self, x):                       # (N, T, V, C)
        n, t, v, c = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()          # (N, C, T, V)
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()
        for b in self.blocks:
            x = b(x)
        pooled = torch.cat([x.mean(dim=(2, 3)),
                            x.amax(dim=(2, 3))], dim=1)
        return self.head(pooled)


def stgcn_input(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> (T, 48, 7): xyz, bone direction xyz, visibility.

    Same information the sequence models get, arranged per joint instead of
    flattened, so the comparison isolates the architecture rather than the
    features.
    """
    from .bones import bone_vectors
    from .landmarks import hand_presence, zero_missing

    pres = hand_presence(clip)                      # (T, 2)
    xyz = zero_missing(clip, pres)                  # (T, 48, 3)
    bone = bone_vectors(clip, unit=True)            # (T, 48, 3)
    vis = np.ones((len(clip), NUM_JOINTS, 1), dtype=np.float32)
    vis[:, 6:27, 0] = pres[:, 0:1]
    vis[:, 27:48, 0] = pres[:, 1:2]
    return np.concatenate([xyz, bone, vis], axis=2).astype(np.float32)
