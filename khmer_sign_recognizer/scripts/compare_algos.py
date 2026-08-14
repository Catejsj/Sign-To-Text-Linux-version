"""Sweep all simple algorithms and rank them — find the best one fast.

Loads the data ONCE, then trains every algorithm in both modes (real and
real+synthetic) and prints a ranked table by macro-F1. Use this to pick a
strong baseline before committing to one algorithm.

    python scripts/compare_algos.py --lang autsl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.baseline_data import load_split          # noqa: E402
from src.v2.baseline_eval import evaluate            # noqa: E402
from src.v2.algorithms import build_model            # noqa: E402

ALGOS = ["knn", "logreg", "rf", "svm", "nb", "tree", "gboost", "lda"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="autsl")
    ap.add_argument("--eval-on", default="val", choices=["val", "test"])
    ap.add_argument("--split-random", type=float, default=None, metavar="FRAC",
                    help="hold out this fraction of your own takes for eval "
                         "(use when there is only one signer)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--features", default="summary", choices=["summary", "flat"])
    ap.add_argument("--skip", default="gboost",
                    help="comma-separated algorithms to skip (default: gboost, "
                         "which is very slow). Pass --skip '' to include all.")
    args = ap.parse_args()

    skip = {a.strip() for a in args.skip.split(",") if a.strip()}
    algos = [a for a in ALGOS if a not in skip]

    print(f"loading data (lang={args.lang}, features={args.features})...")
    split_kw = dict(random_split=args.split_random, split_seed=args.split_seed)
    Xtr_real, ytr_real, _, l2i = load_split(
        "train", args.lang, source_mode="real", feature_mode=args.features,
        **split_kw)
    Xtr_both, ytr_both, _, _ = load_split(
        "train", args.lang, source_mode="both", feature_mode=args.features,
        label_to_idx=l2i, **split_kw)
    Xev, yev, sev, _ = load_split(
        args.eval_on, args.lang, source_mode="real",
        feature_mode=args.features, label_to_idx=l2i, **split_kw)
    n_classes = len(l2i)
    print(f"train(real)={len(ytr_real)}  train(both)={len(ytr_both)}  "
          f"eval={len(yev)}  classes={n_classes}\n")

    rows = []
    for algo in algos:
        rec = {"algo": algo}
        for mode, (X, y) in (("real", (Xtr_real, ytr_real)),
                             ("both", (Xtr_both, ytr_both))):
            t0 = time.time()
            try:
                model, _, _ = build_model(algo)
                model.fit(X, y)
                pred = model.predict(Xev)
                m = evaluate(yev, pred, sev, n_classes)
                rec[mode] = m["macro_f1"]
            except Exception as e:
                rec[mode] = float("nan")
                print(f"  [warn] {algo} {mode}: {e}")
            rec[f"{mode}_t"] = time.time() - t0
        rec["delta"] = rec["both"] - rec["real"]
        rows.append(rec)

    rows.sort(key=lambda r: (r["both"] if r["both"] == r["both"] else -1),
              reverse=True)

    print(f"{'algo':<8} {'real F1':>8} {'both F1':>8} {'d(synth)':>9} {'fit s':>7}")
    print("-" * 46)
    for r in rows:
        print(f"{r['algo']:<8} {r['real']*100:7.1f}% {r['both']*100:7.1f}% "
              f"{r['delta']*100:+8.1f}  {r['both_t']:6.1f}")

    best = rows[0]
    print(f"\nbest by both-F1: {best['algo']}  "
          f"({best['both']*100:.1f}% macro-F1, "
          f"synthetic {'helped' if best['delta'] > 0 else 'did not help'} "
          f"{best['delta']*100:+.1f})")


if __name__ == "__main__":
    main()
