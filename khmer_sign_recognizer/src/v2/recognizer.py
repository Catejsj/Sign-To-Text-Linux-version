"""Saved sign recognizers: persist a trained model, then run it live.

Until now training threw the model away — `run_baseline.py` fit a classifier,
printed a score, and exited. Recognition needs the opposite: train once, save,
then load and predict on a live camera.

A saved model is one .joblib "bundle" holding everything inference needs:

    model         the fitted sklearn pipeline (StandardScaler + estimator)
    label_to_idx  label slug -> class index, exactly as used at training time
    idx_to_text   class index -> human text ("ជម្រាបសួរ"), for display
    feature_mode  "summary" | "flat"  — inference MUST featurize the same way
    view          which view the clip was normalized as (clean/noisy)
    meta          language, algo, scores, when it was trained

Getting `feature_mode` and `view` wrong silently produces garbage predictions
(the numbers still have the right shape), which is why they travel with the model
rather than being passed in by the caller.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .baseline_data import _featurize
from .normalize import clean_clip_from_frames, noisy_clip_from_frames
from .schema import SEQ_LEN, NUM_JOINTS, NUM_COORDS, View

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "recognizers"


# ── persistence ──────────────────────────────────────────────────────

def bundle_path(language: str, algo: str) -> Path:
    return MODELS_DIR / f"{language}__{algo}.joblib"


def save_bundle(model, label_to_idx: dict, language: str, algo: str,
                feature_mode: str = "summary", view: View = View.CLEAN,
                labels_text: Optional[dict] = None,
                meta: Optional[dict] = None) -> Path:
    """Persist a trained model plus everything needed to reproduce its input."""
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    bundle = {
        "model": model,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        # slug -> display text (e.g. "sl_001" -> "ជម្រាបសួរ")
        "labels_text": labels_text or {},
        "feature_mode": feature_mode,
        "view": view.value if isinstance(view, View) else str(view),
        "meta": {"language": language, "algo": algo,
                 "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 **(meta or {})},
    }
    path = bundle_path(language, algo)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path | str) -> dict:
    import joblib
    return joblib.load(Path(path))


# Listing loads every bundle just to read its metadata, and a random-forest
# bundle is several MB (measured ~40 ms total). Cache on the directory's
# modification time, so a newly trained model still appears immediately.
_LIST_CACHE: dict = {"key": None, "value": []}


def list_models() -> list[dict]:
    """Every saved recognizer, newest first — for the Recognize-mode picker."""
    if not MODELS_DIR.exists():
        return []
    try:
        key = (MODELS_DIR.stat().st_mtime,
               tuple(sorted((p.name, p.stat().st_mtime)
                            for p in MODELS_DIR.glob("*.joblib"))))
    except OSError:
        key = None
    if key is not None and key == _LIST_CACHE["key"]:
        return _LIST_CACHE["value"]
    out = []
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            b = load_bundle(p)
        except Exception:
            continue        # ignore unreadable/stale bundles
        m = b.get("meta", {})
        out.append({
            "name": p.stem,
            "file": p.name,
            "language": m.get("language", "?"),
            "algo": m.get("algo", "?"),
            "classes": len(b.get("label_to_idx", {})),
            # older bundles stored this as "accuracy"
            "accuracy": m.get("eval_accuracy", m.get("accuracy")),
            "saved_at": m.get("saved_at", ""),
        })
    out.sort(key=lambda d: d["saved_at"], reverse=True)
    _LIST_CACHE.update(key=key, value=out)
    return out


# ── live inference ───────────────────────────────────────────────────

@dataclass
class Prediction:
    label: str                 # slug
    text: str                  # display text
    confidence: float          # 0..1 (1.0 when the model exposes no scores)
    stable: bool               # survived the vote (safe to show as an answer)
    moving: bool               # was there enough motion to be a real sign
    committed: bool = False    # emitted at the END of a sign (most reliable)


@dataclass
class LiveRecognizer:
    """Rolling-window classifier over live landmark frames.

    Feed one landmark frame per camera frame via `push()`; call `predict()` to
    classify the most recent window. Two guards keep it from spraying nonsense
    when nobody is signing:

      motion gate — a still person produces a constant clip, which the model will
                    still confidently label something. We require movement above
                    `motion_threshold` before predicting at all.
      debounce    — a label must repeat `stable_frames` times before it counts as
                    stable, so single-frame flickers don't show up as answers.
    """
    bundle: dict
    window_seconds: float = 2.0
    fps: int = 30
    stable_frames: int = 3
    min_confidence: float = 0.35
    motion_threshold: float = 0.004
    motion_window: int = 6          # frames used to judge "still" (keep short)
    # majority vote over the last N raw predictions — a sliding window sees
    # partial signs the model never trained on, so single frames disagree a lot
    vote_window: int = 9
    vote_majority: float = 0.6      # fraction of the window that must agree
    # a "sign" ends when motion stops; that whole segment is then classified once
    segment_min_frames: int = 12
    # tuned by replaying real takes: 6 idle frames avoids committing on a brief
    # mid-sign pause, without making the answer feel slow
    idle_frames_to_commit: int = 6

    frames: deque = field(init=False)
    _votes: deque = field(init=False)
    _segment: list = field(init=False)
    _idle: int = field(init=False, default=0)
    _committed: Optional[Prediction] = field(init=False, default=None)
    _last: Optional[Prediction] = field(init=False, default=None)

    def __post_init__(self):
        maxlen = max(SEQ_LEN, int(self.window_seconds * self.fps))
        self.frames = deque(maxlen=maxlen)
        self._votes = deque(maxlen=self.vote_window)
        self._segment = []

    # ── input ──
    def push(self, frame_vec: np.ndarray) -> None:
        """Add one (NUM_JOINTS, NUM_COORDS) landmark frame."""
        if frame_vec is not None:
            self.frames.append(frame_vec)

    def clear(self) -> None:
        self.frames.clear()
        self._votes.clear()
        self._segment = []
        self._idle = 0
        self._committed = None
        self._last = None

    @property
    def ready(self) -> bool:
        # Enough frames to be worth classifying. Kept low deliberately: a high
        # warm-up threshold swallows the START of the very first sign, which
        # showed up in replay as a missed first word.
        return len(self.frames) >= max(6, self.frames.maxlen // 6)

    # ── motion gate ──
    def _motion(self) -> float:
        """Mean absolute frame-to-frame change — a proxy for 'is this a sign'.

        Deliberately a SHORT window: averaging over a long one keeps reporting
        motion for a second after the hands stop, which delays the end-of-sign
        detection and makes the answer feel laggy.
        """
        if len(self.frames) < 3:
            return 0.0
        n = min(len(self.frames), self.motion_window)
        arr = np.stack(list(self.frames)[-n:])
        diffs = np.abs(np.diff(arr, axis=0))
        return float(np.nanmean(diffs)) if diffs.size else 0.0

    # ── classification of one clip ──
    def _none_index(self) -> Optional[int]:
        return self.bundle.get("label_to_idx", {}).get("__none__")

    def _classify(self, frames_list, allow_none: bool = True
                  ) -> Optional[tuple[str, str, float]]:
        """Classify a clip.

        allow_none=False masks the "__none__" class, used when COMMITTING a
        finished sign: by then we know a sign happened, so the useful answer is
        the best real class. Letting none win at commit time cost ~2 points of
        accuracy in testing for no benefit.
        """
        """Run the model on a list of landmark frames -> (label, text, conf)."""
        view = self.bundle.get("view", "clean")
        builder = clean_clip_from_frames if view == "clean" else noisy_clip_from_frames
        clip = builder(list(frames_list))
        if clip.shape != (SEQ_LEN, NUM_JOINTS, NUM_COORDS):
            return None

        model = self.bundle["model"]

        # Deep models (TCN/Transformer) take the raw sequence, not summary stats.
        if self.bundle.get("feature_mode") == "sequence":
            import torch
            with torch.no_grad():
                x = torch.from_numpy(
                    clip.reshape(SEQ_LEN, -1)[None, ...].astype(np.float32))
                logits = model(x)[0]
                proba = torch.softmax(logits, dim=-1).numpy()
            idx = int(np.argmax(proba))
            label = self.bundle["idx_to_label"].get(idx, str(idx))
            text = self.bundle.get("labels_text", {}).get(label, label)
            return label, text, float(proba[idx])

        x = _featurize(clip, self.bundle.get("feature_mode", "summary"))[None, :]

        none_i = self._none_index()
        conf = 1.0
        idx = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(x)[0]
                if not allow_none and none_i is not None and none_i < len(proba):
                    proba = proba.copy()
                    proba[none_i] = -1.0        # never commit "none"
                idx = int(np.argmax(proba))
                conf = float(max(proba[idx], 0.0))
            except Exception:
                idx = None
        if idx is None and hasattr(model, "decision_function"):
            # e.g. SVC without probability=True — turn margins into a 0..1 score
            try:
                scores = np.atleast_2d(model.decision_function(x))[0]
                if scores.size > 1:
                    e = np.exp(scores - scores.max())
                    soft = e / e.sum()
                    idx = int(np.argmax(soft))
                    conf = float(soft[idx])
            except Exception:
                idx = None
        if idx is None:
            idx = int(model.predict(x)[0])

        label = self.bundle["idx_to_label"].get(idx, str(idx))
        text = self.bundle.get("labels_text", {}).get(label, label)
        return label, text, conf

    def _vote(self) -> Optional[tuple[str, float]]:
        """Winner of the recent-prediction vote, if it clears the majority bar."""
        if len(self._votes) < max(3, self.vote_window // 2):
            return None
        labels = [lab for lab, _ in self._votes]
        winner = max(set(labels), key=labels.count)
        share = labels.count(winner) / len(labels)
        if share < self.vote_majority:
            return None
        confs = [c for lab, c in self._votes if lab == winner]
        return winner, float(np.mean(confs))

    # ── output ──
    def predict(self) -> Optional[Prediction]:
        if not self.ready:
            return None

        moving = self._motion() >= self.motion_threshold

        # ── sign finished: classify the WHOLE segment once ──
        # This is the reliable answer, because it matches how the model was
        # trained (one complete take), unlike the sliding window which usually
        # holds a partial sign.
        if not moving:
            self._idle += 1
            if (self._segment and self._idle >= self.idle_frames_to_commit
                    and len(self._segment) >= self.segment_min_frames):
                got = self._classify(self._segment, allow_none=False)
                self._segment = []
                self._votes.clear()
                if got:
                    label, text, conf = got
                    self._committed = Prediction(label, text, conf, stable=True,
                                                 moving=False, committed=True)
                    return self._committed
            if self._idle >= self.idle_frames_to_commit:
                self._segment = []
            # between signs: keep showing the last committed answer
            if self._committed is not None:
                return self._committed
            return Prediction("", "", 0.0, stable=False, moving=False)

        # ── mid-sign: live feedback from the sliding window, majority-voted ──
        self._idle = 0
        if self.frames:
            self._segment.append(self.frames[-1])

        got = self._classify(self.frames)
        if got is None:
            return None
        label, text, conf = got
        # Model says "not a finished sign" — that is the whole point of the none
        # class: stay quiet mid-gesture instead of guessing and flickering.
        if label == "__none__":
            self._votes.clear()
            return Prediction("", "", 0.0, stable=False, moving=True)
        if conf >= self.min_confidence:
            self._votes.append((label, conf))

        won = self._vote()
        if won is None:
            # not settled yet — show the raw guess, but flagged unstable so the
            # UI can render it faintly instead of as an answer
            return Prediction(label, text, conf, stable=False, moving=True)

        win_label, win_conf = won
        win_text = self.bundle.get("labels_text", {}).get(win_label, win_label)
        pred = Prediction(win_label, win_text, win_conf, stable=True, moving=True)
        self._last = pred
        return pred
