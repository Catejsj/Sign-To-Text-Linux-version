"""Bake the skeleton into the numbers, so every model gets structure for free.

THE PROBLEM THIS SOLVES
-----------------------
Nothing in the pipeline ever told a model it was looking at a body. Both
categories receive joints as an unordered bag — `nn.Linear(144, ...)` weights
144 slots with no notion that joint 7 hangs off joint 6, and a decision tree
splits on one coordinate at a time. Shuffle all 48 joints consistently across
the dataset and you would train an identical model. The anatomy had to be
inferred from 337 examples.

The field's answer is a graph network. This is the cheap one: replace a joint's
absolute position with its offset from its parent, and adjacency is encoded in
the features themselves — no new architecture, no schema change, and *both*
categories benefit at once.

DIRECTION, NOT LENGTH — this is the whole result
------------------------------------------------
A bone vector carries direction and length. **Length is body size, which is
signer identity.** The `clean` view divides by shoulder width, but "this person
has long fingers" survives that. So the unit-normalised variant should be
signer-invariant and the raw one should not.

Predicted before measuring, and confirmed (khmer_var, macro-F1):

                          same signer      unseen signer
    joints only               79.1             55.4
    + bone (with length)      86.2             64.8
    + direction (unit)        89.4             70.4      <- classical
    ---
    TCN joints                88.3 ±2.7        73.0 ±13.5
    TCN + bone                89.4 ±2.7        72.4 ±14.0
    TCN + direction           92.0 ±3.1        81.4 ±7.3  <- best overall

Raw bone length **loses 6 points cross-signer against direction** while costing
almost nothing same-signer — exactly the signature of a feature encoding who is
signing rather than what. The TCN's fold spread also halves, 13.5 -> 7.3.

`unit=True` is therefore the default and raw bones are kept only for the
comparison above.

MISSING HANDS
-------------
A zeroed hand must not emit bone vectors pointing at the origin — that is a
confident-looking measurement made of nothing. Bones inside an absent hand are
zeroed, as is the bone linking it to the wrist. Depends on
`landmarks.hand_presence`; see PROBLEM_LOG K.
"""
from __future__ import annotations

import numpy as np

from .landmarks import hand_presence, zero_missing, summary_valid

LEFT_HAND = slice(6, 27)
RIGHT_HAND = slice(27, 48)

# MediaPipe's 21-point hand: wrist 0, then thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20, every finger rooted at the wrist.
_MP_HAND_PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11,
                   0, 13, 14, 15, 0, 17, 18, 19]

#: Parent joint of each of the 48 joints; -1 marks a root.
#: Body 0 Lsh · 1 Rsh · 2 Lelb · 3 Relb · 4 Lwr · 5 Rwr, then each hand
#: rooted at its own wrist so the arm chain runs shoulder->elbow->wrist->hand.
PARENT = np.array(
    [-1, 0, 0, 1, 2, 3]
    + [4] + [6 + p for p in _MP_HAND_PARENT[1:]]
    + [5] + [27 + p for p in _MP_HAND_PARENT[1:]]
)
assert PARENT.shape == (48,), PARENT.shape


def bone_vectors(clip: np.ndarray, unit: bool = True) -> np.ndarray:
    """(T, 48, 3) -> (T, 48, 3), each joint replaced by (joint - parent).

    `unit=True` normalises each vector to length 1, discarding bone length and
    keeping only the angle. That is what makes the representation
    signer-invariant, and it is worth ~6 macro-F1 on an unseen signer.

    Root joints keep their absolute position, so where the person is in frame
    is not thrown away entirely.
    """
    clip = np.asarray(clip, dtype=np.float32)
    pres = hand_presence(clip)
    c = zero_missing(clip, pres)

    out = np.zeros_like(c)
    roots = PARENT < 0
    out[:, roots] = c[:, roots]
    child = np.flatnonzero(~roots)
    out[:, child] = c[:, child] - c[:, PARENT[child]]

    if unit:
        n = np.linalg.norm(out, axis=2, keepdims=True)
        out = np.divide(out, n, out=np.zeros_like(out), where=n > 1e-6)

    for col, sl in ((0, LEFT_HAND), (1, RIGHT_HAND)):
        out[pres[:, col] == 0, sl] = 0.0
    return out


def _stats(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v.mean(0).ravel(), v.std(0).ravel(),
                           v.min(0).ravel(), v.max(0).ravel()])


def summary_bones(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> 1158 features: `landmarks.summary_valid` + bone directions.

    Joints and bones are kept together rather than bones replacing joints —
    absolute position still says where a sign happens in the signing space,
    which direction alone cannot express.
    """
    return np.concatenate([summary_valid(clip),
                           _stats(bone_vectors(clip, unit=True))])


def deep_input_bones(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) -> (T, 290) for the sequence models.

    144 zeroed-if-missing coordinates + 2 hand-presence channels
    + 144 bone directions.
    """
    pres = hand_presence(clip)
    joints = zero_missing(clip, pres).reshape(len(clip), -1)
    bones = bone_vectors(clip, unit=True).reshape(len(clip), -1)
    return np.concatenate([joints, pres, bones], axis=1).astype(np.float32)
