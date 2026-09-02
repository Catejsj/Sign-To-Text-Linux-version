"""One shape in, one shape out.

STATUS: BUILT, MEASURED, **NOT YET BETTER**. Do not switch to it on the
strength of the design. `landmarks.summary_valid` and `landmarks.deep_input`
remain the defaults because they win:

    khmer_var, macro-F1        same signer      unseen signer
    summary (original)             70.0             45.4
    summary_valid (landmarks)      79.1             55.4     <- default
    canonical.summary              77.8             52.0
    ---
    TCN raw (60,144)               85.4             57.8
    TCN presence (60,146)          87.7             73.0     <- default
    TCN canonical (60,192)         85.6             73.9

Two reasons it does not pay yet, both fixable, neither fixable from here:

1. **The gap policy is inert on existing data.** Median hand dropout is 25 of
   60 frames and **100% of episodes touch a clip edge** — hands are not up yet
   when a take starts. Only 0.1% of missing frames sit in gaps short enough to
   interpolate, so `apply_gap_policy` reduces to zeroing, which is exactly what
   `landmarks.zero_missing` already did.

2. **Per-joint visibility is not recoverable from (T, 48, 3).** MediaPipe
   returns a whole hand or nothing, so reconstruction can only ever produce
   per-hand resolution. The extra 46 channels carry no information the 2-channel
   version lacked, and on 337 samples that costs the classical models 1-3
   points to dimensionality.

**What would make it pay:** `capture.py` already computes a real per-joint
confidence (`sc[idx]` for RTMPose at line 299, `lm.visibility` for MediaPipe at
line 544) and uses it to *delete* joints — then `normalize._pt()` reads x/y/z
and discards it. Preserving it at capture time is the root fix; this module is
the consumer that would make it worth something. Until then it is scaffolding.

WHY THIS EXISTS
---------------
Landmark handling was spread across three files and disagreed with itself.
`capture.py` computed a confidence for every joint and used it to *delete*
low-confidence joints; `normalize._pt()` then read x/y/z and dropped the
confidence, so the reason a joint was missing was thrown away one function
after it was known. `fill_nans` replaced the hole with the joint's last known
position, which is indistinguishable from a joint that genuinely stopped
moving. By the time a model saw the clip, a fabricated coordinate and a
measured one looked identical.

This module makes the contract explicit:

    anything in  ->  (T, 48, 4)  ->  x, y, z, visibility

`visibility` is 1.0 where a joint was really measured, 0.0 where it was not,
and fractional where a short gap was interpolated across.

GAP POLICY — the part that matters
----------------------------------
Not all holes are equal, and the previous two attempts both got this wrong in
opposite directions:

  * `fill_nans` holds the last position for as long as the gap lasts. Across a
    30-frame dropout that invents most of a sign.
  * Zeroing everything (`landmarks.zero_missing`, 2026-09-02) is honest but
    throws away 2-frame flickers that are genuinely interpolable.

So: **interpolate short gaps, zero long ones.** `max_interp_gap` is the
boundary, in frames. Interpolated joints get a visibility below 1 so a model
can still tell them from measurements.

READS BOTH CONTRACTS
--------------------
New recordings should store (T, 48, 4) with real per-joint confidence from the
tracker. The 1.2 GB already recorded is (T, 48, 3), so for those the mask is
reconstructed — reliably for hands, conservatively for body. See
`reconstruct_visibility` for exactly how much is knowable.
"""
from __future__ import annotations

import numpy as np

from .schema import NUM_JOINTS, NUM_COORDS

BODY = slice(0, 6)
LEFT_HAND = slice(6, 27)
RIGHT_HAND = slice(27, 48)
HANDS = ((0, LEFT_HAND), (1, RIGHT_HAND))

#: Interpolate gaps up to this many frames; zero anything longer.
DEFAULT_MAX_INTERP_GAP = 3


def reconstruct_visibility(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> (T, 48) visibility, recovered from the damage pattern.

    What is reliably knowable from a clip that has already been through
    `fill_nans`:

    **Hands — reliable.** MediaPipe returns a whole hand or nothing (there is
    no per-joint filter at `capture.py:437`), so a lost hand leaves all 21
    joints on one exact point. That does not happen to a real hand.

    **Body — conservative.** A dropped body joint is frozen at its last
    position, which is genuinely ambiguous with a joint that stopped moving.
    Rather than guess, body joints are marked visible unless they are frozen
    for the WHOLE clip, which means they were never seen at all. Measured on
    khmer_var this affects 0.1% of joints, so little is lost by being careful
    here — and a wrong guess would be worse than none.
    """
    T = clip.shape[0]
    vis = np.ones((T, NUM_JOINTS), dtype=np.float32)

    for _col, sl in HANDS:
        block = np.round(clip[:, sl], 5)
        spread = np.abs(block - block[:, :1]).sum(axis=(1, 2))
        lost = spread <= 1e-9                       # all 21 on one point
        vis[lost, sl] = 0.0

    body = clip[:, BODY]
    frozen = np.abs(np.diff(body, axis=0)).sum(axis=(0, 2)) < 1e-9
    if frozen.any():
        idx = np.arange(BODY.start, BODY.stop)[frozen]
        vis[:, idx] = 0.0
    return vis


def _runs(mask: np.ndarray):
    """Yield (start, stop) half-open spans where mask is True."""
    if not mask.any():
        return
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        yield int(a), int(b)


def apply_gap_policy(xyz: np.ndarray, vis: np.ndarray,
                     max_interp_gap: int = DEFAULT_MAX_INTERP_GAP
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate short holes, zero long ones. Operates per joint.

    A gap touching either end of the clip has only one anchor, so it cannot be
    interpolated and is always zeroed regardless of length.
    """
    out = xyz.copy()
    v = vis.copy()
    T = out.shape[0]

    for j in range(out.shape[1]):
        missing = v[:, j] < 0.5
        if not missing.any():
            continue
        for a, b in _runs(missing):
            span = b - a
            if span <= max_interp_gap and a > 0 and b < T:
                lo, hi = out[a - 1, j], out[b, j]
                w = (np.arange(1, span + 1, dtype=np.float32) /
                     (span + 1))[:, None]
                out[a:b, j] = (1 - w) * lo + w * hi
                # interpolated, not measured — say so, and fade with distance
                v[a:b, j] = 0.5
            else:
                out[a:b, j] = 0.0
                v[a:b, j] = 0.0
    return out, v


def canonicalize(clip: np.ndarray,
                 visibility: np.ndarray | None = None,
                 max_interp_gap: int = DEFAULT_MAX_INTERP_GAP) -> np.ndarray:
    """Anything in -> (T, 48, 4).

    Accepts (T, 48, 3) or (T, 48, 4). If `visibility` is not supplied and the
    clip has no fourth channel, it is reconstructed from the damage pattern.

    The output shape never varies — that is the point of this layer. A clip
    where every hand was lost still returns (T, 48, 4); it just returns it with
    the visibility column telling you so.
    """
    clip = np.asarray(clip, dtype=np.float32)
    if clip.ndim != 3 or clip.shape[1] != NUM_JOINTS:
        raise ValueError(f"expected (T, {NUM_JOINTS}, 3 or 4), got {clip.shape}")

    if clip.shape[2] == NUM_COORDS + 1:
        xyz, vis = clip[..., :NUM_COORDS], clip[..., NUM_COORDS]
    elif clip.shape[2] == NUM_COORDS:
        xyz = clip
        vis = reconstruct_visibility(clip) if visibility is None else visibility
    else:
        raise ValueError(f"last axis must be 3 or 4, got {clip.shape[2]}")

    vis = np.asarray(vis, dtype=np.float32)
    if vis.shape != xyz.shape[:2]:
        raise ValueError(f"visibility {vis.shape} != {xyz.shape[:2]}")

    xyz, vis = apply_gap_policy(xyz, vis, max_interp_gap)
    return np.concatenate([xyz, vis[..., None]], axis=2).astype(np.float32)


def deep_input(clip: np.ndarray, **kw) -> np.ndarray:
    """(T, 48, 3 or 4) -> (T, 192) for the sequence models.

    48 joints x (x, y, z, visibility), flattened. Every model downstream reads
    this and nothing else.
    """
    return canonicalize(clip, **kw).reshape(clip.shape[0], -1)


def _stats(block: np.ndarray) -> np.ndarray:
    if block.shape[0] == 0:
        return np.zeros(block.shape[1] * block.shape[2] * 4, dtype=np.float32)
    return np.concatenate([block.mean(0).ravel(), block.std(0).ravel(),
                           block.min(0).ravel(), block.max(0).ravel()])


def summary(clip: np.ndarray, **kw) -> np.ndarray:
    """(T, 48, 3 or 4) -> 738 features for the classical models.

    Each group is summarised only over the frames it was actually measured in,
    so a frozen fill value can never become the min or the max. Groups with no
    measurement at all contribute zeros, and the visibility block says which.

    body 72 + left hand 252 + right hand 252 + visibility stats 144
    + presence summary 18 = 738.
    """
    c = canonicalize(clip, **kw)
    xyz, vis = c[..., :3], c[..., 3]

    parts = []
    for _col, sl in ((None, BODY), *HANDS):
        seen = vis[:, sl].mean(axis=1) > 0.5
        parts.append(_stats(xyz[seen][:, sl]) if seen.any()
                     else np.zeros((sl.stop - sl.start) * 3 * 4, np.float32))

    # visibility is a signal in its own right: which joints, how often
    parts.append(np.concatenate([vis.mean(0), vis.std(0),
                                 vis.min(0)]).astype(np.float32))

    def episodes(m):
        return float(sum(1 for _ in _runs(m)))

    pres = []
    for _col, sl in ((None, BODY), *HANDS):
        seen = vis[:, sl].mean(axis=1) > 0.5
        pres += [seen.mean(), episodes(~seen), float(seen.any()),
                 float(seen.all()), float(seen[:len(seen) // 2].mean()),
                 float(seen[len(seen) // 2:].mean())]
    parts.append(np.array(pres, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)
