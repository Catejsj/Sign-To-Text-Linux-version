"""Sign Language MNIST — train one algorithm and report its performance.

    python run.py --list                 # what can I run?
    python run.py --algo lda             # train one algorithm
    python run.py --all                  # train every algorithm
    python run.py --chart                # combine everyone's results into charts

The dataset is found automatically in ./data/ (the Kaggle zip, or the CSVs).
Each run writes results/<algo>.json, so several people can run different
algorithms on different machines and combine the results afterwards.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = HERE / "data"
RESULTS = HERE / "results"

# Sign Language MNIST omits 9 (J) and 25 (Z): those letters need motion.
LETTER = {i: chr(ord("A") + i) for i in range(26)}


# ── data ─────────────────────────────────────────────────────────────

def find_dataset() -> Path:
    """Locate the dataset inside ./data/ — a zip, a folder, or loose CSVs."""
    if not DATA.exists():
        raise SystemExit(f"No data folder. Create {DATA} and put the dataset in it.")
    zips = sorted(DATA.glob("*.zip"))
    if zips:
        return zips[0]
    csvs = list(DATA.rglob("*train*.csv"))
    if csvs:
        return DATA
    raise SystemExit(
        f"No dataset found in {DATA}\n\n"
        f"Download it from:\n"
        f"  https://www.kaggle.com/datasets/datamunge/sign-language-mnist\n"
        f"then put archive.zip (or the two CSV files) into that folder.")


def load(path: Path):
    """Return (X_train, y_train, X_test, y_test) scaled to 0-1."""
    import pandas as pd

    def read(part, src):
        if isinstance(src, zipfile.ZipFile):
            names = [n for n in src.namelist()
                     if part in n.lower() and n.lower().endswith(".csv")]
            if not names:
                raise SystemExit(f"{part}*.csv not found inside {path.name}")
            with src.open(sorted(names, key=len)[0]) as fh:
                return pd.read_csv(io.BytesIO(fh.read()))
        hits = sorted(src.rglob(f"*{part}*.csv"), key=lambda p: len(p.name))
        if not hits:
            raise SystemExit(f"{part}*.csv not found under {src}")
        return pd.read_csv(hits[0])

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            tr, te = read("train", z), read("test", z)
    else:
        tr, te = read("train", path), read("test", path)

    def xy(df):
        y = df.iloc[:, 0].to_numpy().astype(np.int64)
        X = df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0
        return X, y

    Xtr, ytr = xy(tr)
    Xte, yte = xy(te)
    return Xtr, ytr, Xte, yte


# ── training ─────────────────────────────────────────────────────────

def train_one(algo: str, Xtr, ytr, Xte, yte, train_size=0, seed=0) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    import algorithms

    label, model = algorithms.get(algo)

    if train_size and train_size < len(Xtr):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), train_size, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    print(f"  training {label} on {len(Xtr):,} images "
          f"({Xtr.shape[1]} features)...")
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(Xte)
    pred_s = time.time() - t0

    classes = np.unique(np.concatenate([ytr, yte]))
    return {
        "algo": algo, "name": label,
        "accuracy": float(accuracy_score(yte, pred) * 100),
        "macro_f1": float(f1_score(yte, pred, average="macro") * 100),
        "train_seconds": round(train_s, 1),
        "predict_seconds": round(pred_s, 1),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "n_classes": int(len(classes)),
        "classes": [LETTER.get(int(c), str(c)) for c in classes],
        "confusion": confusion_matrix(yte, pred, labels=classes).tolist(),
        "machine": f"{platform.system()} {platform.machine()} · Python "
                   f"{platform.python_version()}",
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }


# ── charts ───────────────────────────────────────────────────────────

def charts():
    """Combine every results/<algo>.json into comparison charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(RESULTS.glob("*.json"))
    runs = []
    for f in files:
        if f.name == "summary.json":
            continue
        try:
            runs.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {f.name})")
    if not runs:
        raise SystemExit(f"No results yet in {RESULTS}. Run an algorithm first.")

    runs.sort(key=lambda r: r["accuracy"], reverse=True)
    names = [r["name"] for r in runs]
    acc = [r["accuracy"] / 100 for r in runs]
    f1 = [r["macro_f1"] / 100 for r in runs]

    x = np.arange(len(runs)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(runs)), 5.5))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#4472C4")
    ax.bar(x + w / 2, f1, w, label="Macro F1", color="#ED7D31")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Sign Language MNIST: algorithm comparison\n"
                 f"{runs[0]['n_classes']} letters · {runs[0]['n_train']:,} train / "
                 f"{runs[0]['n_test']:,} test images", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(acc):
        ax.text(i - w / 2, v + 0.012, f"{v*100:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_chart.png", dpi=150)
    plt.close(fig)

    # confusion matrix for the best run
    best = runs[0]
    cm = np.array(best["confusion"], dtype=float)
    row = cm.sum(1, keepdims=True); row[row == 0] = 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm / row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(best["classes"])))
    ax.set_xticklabels(best["classes"], fontsize=8)
    ax.set_yticks(range(len(best["classes"])))
    ax.set_yticklabels(best["classes"], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {best['name']}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_best.png", dpi=150)
    plt.close(fig)

    (RESULTS / "summary.json").write_text(json.dumps(
        [{k: r[k] for k in ("algo", "name", "accuracy", "macro_f1",
                            "train_seconds", "machine", "when")} for r in runs],
        indent=2))

    print(f"\n{'algorithm':<24}{'accuracy':>10}{'macro F1':>10}{'train s':>10}")
    print("-" * 54)
    for r in runs:
        print(f"{r['name']:<24}{r['accuracy']:>9.2f}%{r['macro_f1']:>9.2f}%"
              f"{r['train_seconds']:>10.1f}")
    print(f"\ncharts -> {RESULTS}")


def samples_figure(Xtr, ytr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = np.unique(ytr)[:12]
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for ax, c in zip(axes.ravel(), classes):
        i = int(np.where(ytr == c)[0][0])
        ax.imshow(Xtr[i].reshape(28, 28), cmap="gray")
        ax.set_title(LETTER.get(int(c), str(c)), fontsize=10); ax.axis("off")
    for ax in axes.ravel()[len(classes):]:
        ax.axis("off")
    fig.suptitle("Sign Language MNIST — example images", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "samples.png", dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────

def main():
    import algorithms

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", help="which algorithm to train (see --list)")
    ap.add_argument("--all", action="store_true", help="train every algorithm")
    ap.add_argument("--list", action="store_true", help="list available algorithms")
    ap.add_argument("--chart", action="store_true",
                    help="combine all results/*.json into comparison charts")
    ap.add_argument("--train-size", type=int, default=0,
                    help="use only N training images (faster; 0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None,
                    help="path to the dataset (default: look in ./data/)")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if a.list:
        print("Available algorithms:\n")
        for k, (label, _) in sorted(algorithms.ALGORITHMS.items()):
            print(f"  {k:<10}{label}")
        print("\nAdd your own in algorithms.py")
        return

    if a.chart:
        charts()
        return

    if not a.algo and not a.all:
        ap.error("give --algo NAME, or --all, or --list, or --chart")

    path = Path(a.data).expanduser() if a.data else find_dataset()
    print(f"dataset: {path}")
    Xtr, ytr, Xte, yte = load(path)
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"classes {len(np.unique(ytr))}\n")
    try:
        samples_figure(Xtr, ytr)
    except Exception:
        pass

    todo = sorted(algorithms.ALGORITHMS) if a.all else [a.algo]
    for algo in todo:
        try:
            res = train_one(algo, Xtr, ytr, Xte, yte, a.train_size, a.seed)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  {algo}: FAILED ({type(e).__name__}: {e})")
            if "torch" in str(e).lower() or "No module named" in str(e):
                print("     (needs PyTorch: pip install torch)")
            continue
        (RESULTS / f"{algo}.json").write_text(json.dumps(res, indent=2))
        print(f"  {res['name']}: accuracy {res['accuracy']:.2f}%  "
              f"macro-F1 {res['macro_f1']:.2f}%  "
              f"({res['train_seconds']}s)")
        print(f"  saved -> results/{algo}.json\n")

    print("Combine everyone's results into charts with:  python run.py --chart")


if __name__ == "__main__":
    main()
