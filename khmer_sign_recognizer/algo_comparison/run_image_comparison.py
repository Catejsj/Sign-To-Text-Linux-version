"""Train every algorithm on Sign Language MNIST (image classification).

Companion to run_comparison.py (which uses the project's own landmark data).
This one uses the public Kaggle dataset so every teammate can reproduce the
comparison on identical, already-labelled data:

    https://www.kaggle.com/datasets/datamunge/sign-language-mnist

28x28 grayscale images, 24 static letters (J and Z are excluded because they
require motion). The official train/test split is used as published.

    python algo_comparison/run_image_comparison.py --data ~/Downloads/archive.zip
    python algo_comparison/run_image_comparison.py --data /path/to/folder

Outputs (algo_comparison/results_image/):
    results.json, comparison_chart.png, confusion_<best>.png, samples.png
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_image"

# Same 8 algorithms as the landmark comparison.
PRETTY = {"lda": "LDA", "svm": "SVM", "logreg": "Logistic Regression",
          "knn": "k-NN", "gboost": "Gradient Boosting", "mlp": "MLP",
          "rf": "Random Forest", "gru": "GRU (recurrent)"}
CLASSICAL = ["lda", "svm", "logreg", "knn", "gboost", "mlp", "rf"]

# Sign Language MNIST labels skip 9 (J) and 25 (Z) — motion letters.
LETTERS = {i: chr(ord("A") + i) for i in range(26)}


def build(algo: str):
    """Model registry. Pixel values are scaled to [0,1] beforehand, so the
    scaler used for the landmark features is not needed here."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    return {
        "lda": LinearDiscriminantAnalysis(),
        "svm": SVC(kernel="rbf", C=10, gamma="scale"),
        "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        # HistGradientBoosting rather than GradientBoosting: the latter trains
        # n_classes x n_estimators trees and takes hours on 784 features.
        "gboost": HistGradientBoostingClassifier(max_iter=150),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                             random_state=0),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }[algo]


def load_csvs(data: Path):
    """Accept the Kaggle zip, a folder, or the two CSVs directly."""
    import pandas as pd

    def read(name_part, source):
        if isinstance(source, zipfile.ZipFile):
            names = [n for n in source.namelist()
                     if name_part in n.lower() and n.lower().endswith(".csv")]
            if not names:
                raise SystemExit(f"{name_part}*.csv not found inside the zip")
            with source.open(sorted(names, key=len)[0]) as fh:
                return pd.read_csv(io.BytesIO(fh.read()))
        hits = sorted(source.rglob(f"*{name_part}*.csv"), key=lambda p: len(p.name))
        if not hits:
            raise SystemExit(f"{name_part}*.csv not found under {source}")
        return pd.read_csv(hits[0])

    if data.suffix.lower() == ".zip":
        with zipfile.ZipFile(data) as z:
            return read("train", z), read("test", z)
    return read("train", data), read("test", data)


def to_xy(df):
    y = df.iloc[:, 0].to_numpy().astype(np.int64)
    X = df.iloc[:, 1:].to_numpy().astype(np.float32) / 255.0
    return X, y


def run_gru(Xtr, ytr, Xev, n_classes, epochs=25, seed=0):
    """Read the image as a sequence of 28 rows, each a 28-value vector."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class GRUNet(nn.Module):
        def __init__(self, feat, n_cls, hidden=128):
            super().__init__()
            self.gru = nn.GRU(feat, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_cls))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out.mean(1))

    xt = torch.tensor(Xtr.reshape(-1, 28, 28), device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    model = GRUNet(28, n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        xe = torch.tensor(Xev.reshape(-1, 28, 28), device=dev)
        for i in range(0, len(xe), 1024):
            preds.append(model(xe[i:i + 1024]).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def sample_grid(X, y, classes, path):
    n = min(12, len(classes))
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for ax, c in zip(axes.ravel(), classes[:n]):
        i = int(np.where(y == c)[0][0])
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(LETTERS.get(int(c), str(c)), fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Sign Language MNIST — example images", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bar_chart(res, title, subtitle, path):
    algos = sorted(res, key=lambda a: res[a]["acc"], reverse=True)
    acc = [res[a]["acc"] / 100 for a in algos]
    f1s = [res[a]["f1"] / 100 for a in algos]
    x = np.arange(len(algos)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, acc, w, label="Accuracy", color="#4472C4")
    ax.bar(x + w/2, f1s, w, label="Macro F1", color="#ED7D31")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(a, a) for a in algos], rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.012, f"{v*100:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def confusion_chart(cm, classes, title, path):
    cm = cm.astype(float)
    row = cm.sum(1, keepdims=True); row[row == 0] = 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm / row, cmap="Blues", vmin=0, vmax=1)
    names = [LETTERS.get(int(c), str(c)) for c in classes]
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="archive.zip from Kaggle, or a folder containing the CSVs")
    ap.add_argument("--train-size", type=int, default=0,
                    help="subsample the training set for speed (0 = use all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = load_csvs(Path(a.data).expanduser())
    Xtr, ytr = to_xy(tr)
    Xev, yev = to_xy(te)
    classes = np.unique(np.concatenate([ytr, yev]))

    if a.train_size and a.train_size < len(Xtr):
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(Xtr), a.train_size, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    print(f"train {Xtr.shape}  test {Xev.shape}  classes {len(classes)}")
    sample_grid(Xtr, ytr, classes, OUT / "samples.png")

    res, conf = {}, {}
    for algo in CLASSICAL + ["gru"]:
        t0 = time.time()
        try:
            if algo == "gru":
                pred = run_gru(Xtr, ytr, Xev, int(classes.max()) + 1, seed=a.seed)
            else:
                m = build(algo)
                m.fit(Xtr, ytr)
                pred = m.predict(Xev)
        except Exception as e:
            print(f"  {algo}: skipped ({type(e).__name__}: {e})")
            continue
        acc = accuracy_score(yev, pred) * 100
        f1 = f1_score(yev, pred, average="macro") * 100
        res[algo] = {"acc": acc, "f1": f1, "seconds": round(time.time() - t0, 1)}
        conf[algo] = confusion_matrix(yev, pred, labels=classes)
        print(f"  {PRETTY.get(algo, algo):<22}{acc:>7.2f}%  F1 {f1:>6.2f}%"
              f"  ({res[algo]['seconds']}s)")

    best = max(res, key=lambda k: res[k]["acc"])
    bar_chart(res, "Sign Language MNIST: algorithm comparison",
              f"{len(classes)} letters · {len(Xtr):,} train / {len(Xev):,} test images",
              OUT / "comparison_chart.png")
    confusion_chart(conf[best], classes,
                    f"Confusion matrix — {PRETTY.get(best, best)}",
                    OUT / f"confusion_{best}.png")

    (OUT / "results.json").write_text(json.dumps({
        "dataset": "Sign Language MNIST (Kaggle: datamunge/sign-language-mnist)",
        "n_train": int(len(Xtr)), "n_test": int(len(Xev)),
        "n_classes": int(len(classes)),
        "classes": [LETTERS.get(int(c), str(c)) for c in classes],
        "results": res, "best": best,
    }, indent=2))
    print(f"\nbest: {PRETTY.get(best, best)} ({res[best]['acc']:.2f}%)  -> {OUT}")


if __name__ == "__main__":
    main()
