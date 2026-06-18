"""Shared metrics for the baseline comparison.

Every teammate reports the SAME numbers computed the SAME way, so the
8-algorithm table is comparable. Metrics the teacher asked for plus the
one that makes leave-one-signer-out meaningful (per-signer F1).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             signer_ids: list[str], n_classes: int) -> dict:
    """Compute accuracy, macro-F1, per-signer macro-F1, confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro",
                              labels=list(range(n_classes)), zero_division=0))

    per_signer: dict[str, float] = {}
    sids = np.asarray(signer_ids)
    for sid in sorted(set(signer_ids)):
        m = sids == sid
        per_signer[sid] = float(f1_score(
            y_true[m], y_pred[m], average="macro",
            labels=list(range(n_classes)), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    signer_vals = list(per_signer.values())
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_signer_f1": per_signer,
        "per_signer_f1_mean": float(np.mean(signer_vals)) if signer_vals else 0.0,
        "per_signer_f1_std": float(np.std(signer_vals)) if signer_vals else 0.0,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def print_report(name: str, metrics: dict,
                 idx_to_label: dict[int, str] | None = None) -> None:
    print(f"\n{'='*56}\n  {name}\n{'='*56}")
    print(f"  samples           : {metrics['n_samples']}")
    print(f"  accuracy          : {metrics['accuracy']*100:5.1f}%")
    print(f"  macro-F1          : {metrics['macro_f1']*100:5.1f}%")
    print(f"  per-signer F1 mean: {metrics['per_signer_f1_mean']*100:5.1f}% "
          f"(+/- {metrics['per_signer_f1_std']*100:.1f})")
    print("  per-signer F1:")
    for sid, f in sorted(metrics["per_signer_f1"].items()):
        print(f"      {sid:<28s} {f*100:5.1f}%")

    cm = np.array(metrics["confusion_matrix"])
    n = cm.shape[0]
    labels = ([idx_to_label.get(i, str(i)) for i in range(n)]
              if idx_to_label else [str(i) for i in range(n)])
    short = [lab[:6] for lab in labels]
    print("\n  confusion matrix (rows=true, cols=pred):")
    print("           " + " ".join(f"{s:>6s}" for s in short))
    for i in range(n):
        print(f"    {short[i]:>7s} " + " ".join(f"{cm[i, j]:6d}" for j in range(n)))
