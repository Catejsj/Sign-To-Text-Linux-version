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
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.baseline_data import load_split                      # noqa: E402
from src.v2.baseline_eval import evaluate, print_report          # noqa: E402


def build_model(algo: str):
    """Registry of the 8 'simple' (non-deep-learning) algorithms.

    Each is wrapped in a StandardScaler pipeline so features are normalized
    the same way for everyone. All come straight from scikit-learn.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    estimators = {
        "knn":    KNeighborsClassifier(n_neighbors=5),
        "logreg": LogisticRegression(max_iter=2000),
        "rf":     RandomForestClassifier(n_estimators=300, random_state=0),
        "svm":    SVC(kernel="rbf", C=10, gamma="scale"),
        "nb":     GaussianNB(),
        "tree":   DecisionTreeClassifier(random_state=0),
        "gboost": GradientBoostingClassifier(random_state=0),
        "lda":    LinearDiscriminantAnalysis(),
    }
    if algo not in estimators:
        raise SystemExit(f"--algo must be one of {sorted(estimators)}")
    return make_pipeline(StandardScaler(), estimators[algo])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True,
                    help="knn | logreg | rf | svm | nb | tree | gboost | lda")
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
    args = ap.parse_args()

    print(f"algo={args.algo}  lang={args.lang}  train-mode={args.mode}  "
          f"eval-on={args.eval_on}  features={args.features}")

    # Train split (mode-dependent), then eval split (always real) with the
    # SAME label index.
    Xtr, ytr, _, l2i = load_split(
        "train", args.lang, source_mode=args.mode, feature_mode=args.features)
    Xev, yev, sev, _ = load_split(
        args.eval_on, args.lang, source_mode="real",
        feature_mode=args.features, label_to_idx=l2i)

    idx_to_label = {i: lab for lab, i in l2i.items()}
    print(f"train: {Xtr.shape[0]} samples x {Xtr.shape[1]} features"
          f"   eval: {Xev.shape[0]} samples")

    model = build_model(args.algo)
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


if __name__ == "__main__":
    main()
