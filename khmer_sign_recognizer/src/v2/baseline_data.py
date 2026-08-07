"""Load v2 samples as flat feature vectors for classical-ML baselines.

No PyTorch. Just numpy + scikit-learn-friendly arrays. Every teammate's
algorithm loads data through here so the 8-way comparison is apples-to-apples.

The AUTSL importer encodes the official train/val/test split in each
signer id (``autsl_p000_train``, ``autsl_p012_val``, ...). We read the
split from that suffix. Locally-recorded KSL data has no suffix, so it
falls into "train" by default.

THE ONE SAFETY RULE
-------------------
Evaluation splits (val / test) are ALWAYS loaded as REAL samples only,
no matter what ``source_mode`` you ask for. You never grade the model on
synthetic cards — that would measure performance on fake data, not real
signers. ``source_mode`` only ever affects the TRAIN split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from .schema import Source, View, SEQ_LEN, NUM_JOINTS, NUM_COORDS
from .dataset import discover_samples
from .augment import augment_clip

ROOT_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "sequences_v2"

VALID_SPLITS = ("train", "val", "test")
VALID_SOURCE_MODES = ("real", "synthetic", "both")
VALID_FEATURES = ("summary", "flat")


def split_of(signer_id: str) -> str:
    """Read train/val/test from the signer-id suffix; default 'train'."""
    for s in VALID_SPLITS:
        if signer_id.endswith("_" + s):
            return s
    return "train"


def _featurize(clip: np.ndarray, feature_mode: str) -> np.ndarray:
    """(60, 48, 3) clip -> 1-D feature vector.

    flat    : every number, 60*48*3 = 8640 features. Faithful but high-dim.
    summary : per-(joint,coord) mean/std/min/max over time, 48*3*4 = 576
              features. Lower-dim, usually better for classical models.
    """
    if feature_mode == "flat":
        return clip.reshape(-1).astype(np.float32)
    return np.concatenate([
        clip.mean(axis=0).reshape(-1),
        clip.std(axis=0).reshape(-1),
        clip.min(axis=0).reshape(-1),
        clip.max(axis=0).reshape(-1),
    ]).astype(np.float32)


NONE_LABEL = "__none__"


def make_negatives(clips: list, rng, per_clip: int = 2,
                   max_frac: float = 0.35) -> list:
    """Build "not a finished sign" samples out of real takes.

    Live recognition classifies a SLIDING window, so most of the time it holds a
    half-finished sign or a person sitting still — inputs a 7-class model never
    saw and cannot decline to answer, so it picks the nearest sign and flickers.

    Giving it an explicit "none" class to choose instead is what fixes that.
    Negatives are only CLEARLY partial (under `max_frac` of a sign) plus static
    holds; including near-complete fragments makes the class greedy and it starts
    swallowing real signs (measured: committed accuracy fell to 89.7%).

    Measured with max_frac=0.35, per_clip=2, committing on the best NON-none
    class:  flicker 1.15 -> 0.04 changes/sign, accuracy 96.8% -> 96.7%.
    """
    from .normalize import resample_time
    out = []
    cut = max(9, int(SEQ_LEN * max_frac))
    for clip in clips:
        for _ in range(per_clip):
            if rng.integers(0, 2) == 0:
                frag = clip[:rng.integers(8, cut)]          # sign still in progress
            else:
                frag = np.repeat(clip[rng.integers(0, SEQ_LEN)][None], 20, axis=0)
            out.append(resample_time(frag.astype(np.float32), SEQ_LEN))
    return out


def _group_map(all_samples) -> tuple[dict, dict]:
    """Map every sample to the REAL take it came from, so a random split can be
    leak-free.

    Synthetic samples are body-retargeted copies of one real take, but their
    filenames carry a fresh variant index with no back-reference. They are
    generated N-per-take in order, though, so when the synthetic count divides
    the real count exactly, synthetic variant v came from real take v // N.

    Returns (group_of, ratio_of) keyed by (label, signer_id). If a label's counts
    don't divide evenly the ratio is None and its synthetic samples must not be
    used in a random split — otherwise a copy of a test take could land in train.
    """
    from collections import defaultdict
    counts = defaultdict(lambda: {"real": 0, "syn": 0})
    for _, m in all_samples:
        if m.view != View.CLEAN:
            continue
        key = (m.label, m.signer_id)
        counts[key]["real" if m.source == Source.REAL else "syn"] += 1

    ratio_of = {}
    for key, c in counts.items():
        if c["real"] and c["syn"] and c["syn"] % c["real"] == 0:
            ratio_of[key] = c["syn"] // c["real"]
        else:
            ratio_of[key] = None
    return counts, ratio_of


def _group_of(meta, ratio_of) -> Optional[int]:
    """Which real take this sample belongs to (None = cannot tell)."""
    if meta.source == Source.REAL:
        return meta.variant
    ratio = ratio_of.get((meta.label, meta.signer_id))
    if not ratio:
        return None
    return meta.variant // ratio


def load_split(
    split: str,
    language: str,
    source_mode: str = "real",
    root: Optional[Path] = None,
    view: View = View.CLEAN,
    feature_mode: str = "summary",
    label_to_idx: Optional[dict[str, int]] = None,
    holdout_signer: Optional[str] = None,
    random_split: Optional[float] = None,
    split_seed: int = 0,
    augment_copies: int = 0,
    augment_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Return (X, y, signer_ids, label_to_idx) for one split.

    X            : (N, D) float32 feature matrix
    y            : (N,)   int class indices
    signer_ids   : list[str], len N — for per-signer scoring
    label_to_idx : the label->index map (pass it back in for other splits
                   so val/test use the same indexing as train)

    holdout_signer: if given, ignore the _train/_val/_test suffixes and do
                    leave-one-signer-out instead — the train split is every
                    signer EXCEPT this one, and val/test is ONLY this one.
                    Use when your data has no built-in val split (e.g. a few
                    teammates' recordings) and you want to test on one of them.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    if source_mode not in VALID_SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {VALID_SOURCE_MODES}")
    if feature_mode not in VALID_FEATURES:
        raise ValueError(f"feature_mode must be one of {VALID_FEATURES}")

    root = Path(root) if root is not None else ROOT_DEFAULT

    # SAFETY: only the train split may include synthetic. Eval = real only.
    effective_mode = source_mode if split == "train" else "real"
    allowed_sources = {
        "real": {Source.REAL},
        "synthetic": {Source.SYNTHETIC},
        "both": {Source.REAL, Source.SYNTHETIC},
    }[effective_mode]

    all_samples = discover_samples(root, language=language)

    # Build a stable label index from ALL real labels if not provided, so
    # train/val/test agree even if a rare label is missing from one split.
    if label_to_idx is None:
        labels = sorted({m.label for _, m in all_samples})
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # ── random split (for data with one signer and no built-in val split) ──
    eval_groups: dict = {}
    ratio_of: dict = {}
    if random_split is not None:
        if not 0.0 < random_split < 1.0:
            raise ValueError("random_split must be between 0 and 1")
        _, ratio_of = _group_map(all_samples)
        # Per (label, signer): hold out a fraction of the REAL takes. Splitting
        # by take — not by sample — is what keeps a take's synthetic copies on
        # the same side of the split.
        from collections import defaultdict
        real_groups = defaultdict(set)
        for _, m in all_samples:
            if m.source == Source.REAL and m.view == View.CLEAN:
                real_groups[(m.label, m.signer_id)].add(m.variant)
        rng = np.random.default_rng(split_seed)
        for key, groups in real_groups.items():
            g = np.array(sorted(groups))
            rng.shuffle(g)
            n_eval = max(1, int(round(len(g) * random_split)))
            eval_groups[key] = set(int(v) for v in g[:n_eval])

    X, y, signers = [], [], []
    for npy, meta in all_samples:
        if random_split is not None:
            key = (meta.label, meta.signer_id)
            group = _group_of(meta, ratio_of)
            in_eval = group is not None and group in eval_groups.get(key, set())
            if split == "train":
                # Untraceable synthetic (uneven counts) is dropped rather than
                # risked — it could be a copy of a held-out take.
                if group is None or in_eval:
                    continue
            else:
                if not in_eval:
                    continue
        elif holdout_signer is not None:
            is_holdout = (meta.signer_id == holdout_signer)
            # train = everyone except the held-out signer; val/test = only them
            if split == "train" and is_holdout:
                continue
            if split in ("val", "test") and not is_holdout:
                continue
        elif split_of(meta.signer_id) != split:
            continue
        if meta.source not in allowed_sources:
            continue
        if meta.view != view:
            continue
        clip = np.load(npy).astype(np.float32)
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            continue
        X.append(_featurize(clip, feature_mode))
        y.append(label_to_idx[meta.label])
        signers.append(meta.signer_id)

        # On-the-fly augmentation (TRAIN ONLY — never touch the eval set, or the
        # score stops meaning anything). The deep-model path already does this
        # via SignDataset; this brings the same treatment to the classical
        # models. time_warp + noise + rotation simulate the session-to-session
        # variation that body-proportion retargeting alone cannot.
        if augment_copies > 0 and split == "train":
            for k in range(augment_copies):
                rng = np.random.default_rng(
                    (augment_seed, hash(str(npy)) & 0xFFFF, k))
                X.append(_featurize(augment_clip(clip, rng=rng), feature_mode))
                y.append(label_to_idx[meta.label])
                signers.append(meta.signer_id)

    if not X:
        raise RuntimeError(
            f"no samples for split={split!r} language={language!r} "
            f"source={effective_mode!r} view={view.value!r} under {root}"
        )
    return (np.stack(X), np.asarray(y, dtype=np.int64), signers, label_to_idx)
