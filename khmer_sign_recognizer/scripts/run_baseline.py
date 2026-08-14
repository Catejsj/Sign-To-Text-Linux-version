"""Run ONE classical-ML algorithm on the imported dataset and report metrics.

This is the 8-person experiment driver. Each teammate owns one --algo and
runs it twice: once --mode real (baseline), once --mode both (real+synthetic).
If macro-F1 goes UP from real -> both, synthetic augmentation helped.

The held-out evaluation set is ALWAYS real (enforced in baseline_data.py),
so the comparison is honest.

EXAMPLES
--------
    # Person 1, k-NN, the two runs that matter:
    python scripts/run_baseline.py --algo knn --lang autsl --mode real
    python scripts/run_baseline.py --algo knn --lang autsl --mode both

    # optional sanity check — train on synthetic only:
    python scripts/run_baseline.py --algo knn --lang autsl --mode synthetic

Results append to data/experiments/baseline_results.csv so the whole team
builds one shared comparison table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.algorithms import build_model, print_list            # noqa: E402
from src.v2.baseline_data import load_split                      # noqa: E402
from src.v2.baseline_eval import evaluate, print_report          # noqa: E402




def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default=None,
                    help="which algorithm to train. Run --list to see them "
                         "all, including any added in custom_algos/.")
    ap.add_argument("--list", action="store_true",
                    help="show every available algorithm and exit")
    ap.add_argument("--lang", default="autsl",
                    help="language folder to use (default: autsl)")
    ap.add_argument("--mode", default="real",
                    choices=["real", "synthetic", "both"],
                    help="TRAIN data source. Eval is always real. "
                         "real=baseline, both=real+synthetic.")
    ap.add_argument("--eval-on", default="val", choices=["val", "test"],
                    help="which held-out split to score on (default: val)")
    ap.add_argument("--features", default="summary",
                    choices=["summary", "flat"],
                    help="feature representation (default: summary)")
    ap.add_argument("--holdout", default=None,
                    help="leave-one-signer-out: train on everyone EXCEPT this "
                         "signer, test on ONLY this signer. Use when your data "
                         "has no built-in val split (e.g. a few teammates' "
                         "recordings). Example: --holdout chingsan")
    ap.add_argument("--split-random", type=float, default=None, metavar="FRAC",
                    help="hold out this fraction of YOUR OWN takes for eval "
                         "(e.g. 0.2). Use when there is only one signer, so "
                         "--holdout can't work. Splits by take, so a take's "
                         "synthetic copies never straddle the split.")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for --split-random (default 0)")
    ap.add_argument("--augment", type=int, default=0, metavar="N",
                    help="add N augmented copies of each TRAIN sample "
                         "(time-warp + noise + rotation, from src/v2/augment.py). "
                         "Eval is never augmented. Measured to help only weak "
                         "models on same-signer data — expected to matter once "
                         "there are multiple signers.")
    ap.add_argument("--none-class", type=int, default=0, metavar="N",
                    help="add N 'not a finished sign' samples per training clip "
                         "(partial windows + static holds). Stops the live "
                         "recognizer flickering between signs mid-gesture. "
                         "Measured: flicker 1.15 -> 0.04 changes/sign, accuracy "
                         "unchanged. Recommended: 2")
    ap.add_argument("--save", action="store_true",
                    help="save the trained model to models/recognizers/ so the "
                         "web app's Recognize mode can load it")
    args = ap.parse_args()

    if args.list:
        print_list()
        return
    if not args.algo:
        raise SystemExit("--algo is required (or --list to see the options)")
    # Fail on a bad name now, not after minutes of loading data.
    build_model(args.algo)

    print(f"algo={args.algo}  lang={args.lang}  train-mode={args.mode}  "
          f"eval-on={args.eval_on}  features={args.features}"
          + (f"  holdout={args.holdout}" if args.holdout else ""))

    # Train split (mode-dependent), then eval split (always real) with the
    # SAME label index.
    Xtr, ytr, _, l2i = load_split(
        "train", args.lang, source_mode=args.mode, feature_mode=args.features,
        holdout_signer=args.holdout, random_split=args.split_random,
        split_seed=args.split_seed, augment_copies=args.augment,
        augment_seed=args.split_seed)
    Xev, yev, sev, _ = load_split(
        args.eval_on, args.lang, source_mode="real",
        feature_mode=args.features, label_to_idx=l2i,
        holdout_signer=args.holdout, random_split=args.split_random,
        split_seed=args.split_seed)

    if args.none_class > 0:
        # Negatives are built from the TRAIN clips only; the eval split never
        # sees them, so the reported score stays comparable to runs without.
        import numpy as _np
        from src.v2.baseline_data import (make_negatives, NONE_LABEL,
                                          _featurize as _feat)
        from src.v2.dataset import discover_samples as _disc
        from src.v2.schema import Source as _Src, View as _V
        _rng = _np.random.default_rng(args.split_seed)
        _clips = []
        for _p, _m in _disc(ROOT / "data" / "sequences_v2", language=args.lang):
            if _m.source is _Src.REAL and _m.view is _V.CLEAN:
                _clips.append(_np.load(_p).astype(_np.float32))
        _neg = make_negatives(_clips, _rng, per_clip=args.none_class)
        if _neg:
            l2i = dict(l2i)
            l2i.setdefault(NONE_LABEL, max(l2i.values()) + 1)
            _X = _np.stack([_feat(c, args.features) for c in _neg])
            _y = _np.full(len(_neg), l2i[NONE_LABEL], dtype=_np.int64)
            Xtr = _np.concatenate([Xtr, _X]); ytr = _np.concatenate([ytr, _y])
            print(f"none-class: +{len(_neg)} negative samples")

    idx_to_label = {i: lab for lab, i in l2i.items()}
    print(f"train: {Xtr.shape[0]} samples x {Xtr.shape[1]} features"
          f"   eval: {Xev.shape[0]} samples")

    model, algo_label, algo_origin = build_model(args.algo)
    print(f"model: {algo_label} ({algo_origin})")
    print("training...")
    model.fit(Xtr, ytr)
    ypred = model.predict(Xev)

    metrics = evaluate(yev, ypred, sev, n_classes=len(l2i))
    title = f"{args.algo}  |  train={args.mode}  |  eval={args.eval_on} ({args.lang})"
    print_report(title, metrics, idx_to_label)

    # Append to the shared results table.
    out_dir = ROOT / "data" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_results.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "algo", "lang", "train_mode", "eval_on",
                        "features", "n_train", "n_eval",
                        "accuracy", "macro_f1",
                        "per_signer_f1_mean", "per_signer_f1_std"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    args.algo, args.lang, args.mode, args.eval_on,
                    args.features, Xtr.shape[0], Xev.shape[0],
                    f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}",
                    f"{metrics['per_signer_f1_mean']:.4f}",
                    f"{metrics['per_signer_f1_std']:.4f}"])
    print(f"\nappended to {csv_path}")

    if args.save:
        # Retrain on train+eval before saving: the scores above already told us
        # how well this setup generalizes, so the model we ship should use every
        # sample we have rather than throwing the eval split away.
        from src.v2.recognizer import save_bundle
        import numpy as _np
        X_all = _np.concatenate([Xtr, Xev])
        y_all = _np.concatenate([ytr, yev])
        final, _, _ = build_model(args.algo)
        final.fit(X_all, y_all)

        labels_path = ROOT / "data" / "sequences_v2" / args.lang / "labels.json"
        labels_text = {}
        if labels_path.exists():
            try:
                labels_text = json.loads(labels_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        path = save_bundle(
            final, l2i, language=args.lang, algo=args.algo,
            feature_mode=args.features, labels_text=labels_text,
            # These metrics belong to the model scored above, NOT to the one
            # being saved here — that one was retrained on train+eval, so it
            # has no held-out score at all. Named so nobody quotes them as the
            # shipped model's accuracy.
            meta={"eval_accuracy": round(float(metrics["accuracy"]), 4),
                  "eval_macro_f1": round(float(metrics["macro_f1"]), 4),
                  "eval_holdout": args.holdout,
                  "train_mode": args.mode,
                  "n_train": int(X_all.shape[0]),
                  "shipped_trained_on": "train+eval (every sample, "
                                        "including the evaluation split)"},
        )
        print(f"saved model -> {path}")
        print("  (Recognize mode in the web app can now load it)")


if __name__ == "__main__":
    main()
