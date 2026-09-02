"""Recovering what the tracker actually saw, from clips already recorded.

THE PROBLEM
-----------
`normalize.fill_nans` replaces an undetected joint with its last known position,
or 0 if it was never seen. For a hand that means all 21 joints collapse onto one
point — and nothing anywhere records that it happened. The saved array holds a
plausible-looking coordinate where there was no measurement.

Measured on `khmer_var`, 150 real takes:

    left hand not detected   50.8% of frames
    right hand not detected  35.9% of frames
    both                     33.1% of frames

Half the numbers feeding the models were invented, silently. Since feature
importance puts 92.3% on hand joints, that is most of the input.

THE RECOVERY
------------
No re-recording is needed. Twenty-one joints sharing one exact coordinate does
not happen to a real hand, so the mask is derivable from the stored clip. It is
also clearly real tracking loss rather than noise: the median take has ONE
contiguous dropout episode per hand, not scattered frames.

WHAT IT IS WORTH
----------------
`khmer_var`, 5-fold take-aware CV and leave-one-signer-out, macro-F1:

                                    same signer   unseen signer
    classical mean, as shipped          70.0          45.4
    classical mean, valid-only stats    79.1          55.6   (+9.1 / +10.3)
    TCN, as shipped                     86.2          57.8
    TCN, zeroed + presence              87.4          73.0   (+1.3 / +15.2)

**It helps more on a new person than on the same person**, which is the
signature of a real fix rather than a shortcut — a signer-identity leak would
do the opposite. That matters here because dropout rate varies 3x BY SIGNER
(82% vs 26% between two of the four), so presence does partly encode who is
signing.

ONE RESULT WORTH KEEPING IN MIND
--------------------------------
Adding presence channels while still feeding the frozen coordinates made the
TCN **worse** (-7.0 unseen-signer). Annotating the lie is not enough; the
fabricated coordinates have to go. Hence `zero_missing` before `presence`.
"""
from __future__ import annotations

import numpy as np

from .schema import NUM_JOINTS

# Joint layout: 0-5 body, 6-26 left hand, 27-47 right hand.
BODY = slice(0, 6)
LEFT_HAND = slice(6, 27)
RIGHT_HAND = slice(27, 48)
HANDS = (("left", LEFT_HAND), ("right", RIGHT_HAND))


def hand_presence(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> (T, 2) float mask, 1.0 where that hand was detected.

    Column 0 is the left hand, column 1 the right. A hand counts as missing in
    a frame when all 21 of its joints occupy one point, which is what
    `fill_nans` leaves behind and what a real hand never does.
    """
    if clip.ndim != 3 or clip.shape[1] != NUM_JOINTS:
        raise ValueError(f"expected (T, {NUM_JOINTS}, 3), got {clip.shape}")
    T = clip.shape[0]
    out = np.zeros((T, 2), dtype=np.float32)
    for col, (_name, sl) in enumerate(HANDS):
        block = np.round(clip[:, sl], 5)
        # a hand is present iff any joint differs from the first joint
        spread = np.abs(block - block[:, :1]).sum(axis=(1, 2))
        out[:, col] = (spread > 1e-9).astype(np.float32)
    return out


def zero_missing(clip: np.ndarray, presence: np.ndarray | None = None
                 ) -> np.ndarray:
    """Replace fabricated hand coordinates with 0 instead of a frozen copy.

    Pair this with the presence mask so a model can tell "hand at the origin"
    from "no hand" — on its own, zeroing is just a different lie.
    """
    p = hand_presence(clip) if presence is None else presence
    out = clip.copy()
    for col, (_name, sl) in enumerate(HANDS):
        out[p[:, col] == 0, sl] = 0.0
    return out


def deep_input(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> (T, 146) for the sequence models.

    144 coordinates with fabricated hands zeroed, plus 2 presence channels.
    This is variant C in the table above — the configuration that gained
    +15.2 macro-F1 on an unseen signer.
    """
    p = hand_presence(clip)
    flat = zero_missing(clip, p).reshape(clip.shape[0], -1)
    return np.concatenate([flat, p], axis=1).astype(np.float32)


def _stats(block: np.ndarray) -> np.ndarray:
    return np.concatenate([block.mean(0).ravel(), block.std(0).ravel(),
                           block.min(0).ravel(), block.max(0).ravel()])


def summary_valid(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> 582 features, summarising each hand over REAL frames only.

    The plain `summary` mode takes mean/std/min/max over all 60 frames, so a
    frozen fill value lands in the summary as though it were a measurement —
    and because it is held for many frames it very often *is* the min or max.
    Restricting each hand to the frames it existed in is the whole fix, and it
    is worth +9.1 same-signer / +10.3 unseen-signer averaged over nine
    algorithms.

    Layout: body stats (72) + left-hand stats (252) + right-hand stats (252)
    + 6 presence summaries = 582.

    A hand never detected in a take contributes zeros. That is honest — there
    is no measurement — and the presence block tells the model to read it that
    way rather than as a hand resting at the origin.
    """
    p = hand_presence(clip)
    parts = [_stats(clip[:, BODY])]
    for col, (_name, sl) in enumerate(HANDS):
        seen = p[:, col] == 1
        parts.append(_stats(clip[seen][:, sl]) if seen.any()
                     else np.zeros(21 * 3 * 4, dtype=np.float32))

    def episodes(mask: np.ndarray) -> float:
        """How broken up the tracking was — one long dropout and twenty short
        ones are different situations and the fraction alone cannot say so."""
        d = np.diff(mask.astype(int))
        return float((d == 1).sum() + int(mask[0]))

    parts.append(np.array([p[:, 0].mean(), p[:, 1].mean(),
                           episodes(p[:, 0] == 1), episodes(p[:, 1] == 1),
                           float(p[:, 0].any()), float(p[:, 1].any())],
                          dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def dropout_report(clips: list[np.ndarray]) -> dict:
    """Aggregate tracking loss over a set of clips, for `verify_pool.py`."""
    if not clips:
        return {"clips": 0}
    p = [hand_presence(c) for c in clips]
    left = np.array([1 - x[:, 0].mean() for x in p])
    right = np.array([1 - x[:, 1].mean() for x in p])
    both = np.array([((x[:, 0] == 0) & (x[:, 1] == 0)).mean() for x in p])
    return {
        "clips": len(clips),
        "left_missing_mean": float(left.mean()),
        "right_missing_mean": float(right.mean()),
        "both_missing_mean": float(both.mean()),
        "clips_left_never_seen": int((left == 1.0).sum()),
        "clips_right_never_seen": int((right == 1.0).sum()),
    }
