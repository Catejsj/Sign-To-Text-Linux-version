"""Train every algorithm on the SignLink dataset and produce comparison charts.

Self-contained: this whole folder can be deleted afterwards without affecting
the project. It only READS data/sequences_v2 — nothing is written outside
algo_comparison/.

    python algo_comparison/run_comparison.py --lang khmer

Outputs (into algo_comparison/results/):
    results.json          every metric, for the report
    comparison_chart.png  accuracy + macro-F1 per algorithm (same signer)
    cross_signer.png      same-signer vs unseen-signer accuracy
    synthetic_effect.png  effect of synthetic augmentation, per evaluation type
    confusion_<algo>.png  confusion matrix for the best algorithm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix  # noqa: E402

from src.v2.baseline_data import load_split                       # noqa: E402
from src.v2.algorithms import build_model, registry               # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# The 8 algorithms in the group: 7 classical + 1 recurrent (GRU).
CLASSICAL = ["lda", "svm", "logreg", "knn", "gboost", "mlp", "rf"]

# Anything in custom_algos/ joins the comparison automatically, so a
# teammate's own algorithm lands in the report without editing this file.
_TABLE, _ = registry()
CUSTOM = sorted(k for k, (_l, _f, origin) in _TABLE.items()
                if origin != "built-in" and k not in CLASSICAL)
CLASSICAL = CLASSICAL + CUSTOM
PRETTY = {
    "lda": "LDA", "svm": "SVM", "logreg": "Logistic Regression",
    "knn": "k-NN", "gboost": "Gradient Boosting", "mlp": "MLP",
    "rf": "Random Forest", "gru": "GRU (recurrent)",
}
PRETTY.update({k: _TABLE[k][0] for k in CUSTOM})


# ── GRU: a small recurrent baseline, trained on the raw 60-frame sequence ──
def run_gru(Xtr_seq, ytr, Xev_seq, yev, n_classes, epochs=40, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class GRUNet(nn.Module):
        def __init__(self, feat, n_cls, hidden=128):
            super().__init__()
            self.gru = nn.GRU(feat, hidden, num_layers=1, batch_first=True,
                              bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_cls))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out.mean(dim=1))       # average over time

    xt = torch.tensor(Xtr_seq, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    xe = torch.tensor(Xev_seq, dtype=torch.float32, device=dev)

    model = GRUNet(xt.shape[-1], n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    bs = 64
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xe).argmax(-1).cpu().numpy()
    return pred


def sequences(split, lang, mode, l2i=None, **kw):
    """Load the same split as flat (60,144) sequences for the GRU."""
    from src.v2.dataset import discover_samples
    from src.v2.schema import Source, View, SEQ_LEN
    # reuse load_split's selection by matching its feature rows is fragile, so
    # rebuild directly with the same arguments the classical path uses
    X, y, s, idx = load_split(split, lang, source_mode=mode, feature_mode="flat",
                              label_to_idx=l2i, **kw)
    return X.reshape(len(X), SEQ_LEN, -1), y, s, idx


def evaluate_all(lang, seeds, holdout=None, mode="real"):
    """Return {algo: {'acc':[], 'f1':[]}} over several splits."""
    res = {a: {"acc": [], "f1": []} for a in CLASSICAL + ["gru"]}
    conf = {}
    labels = None
    for seed in seeds:
        kw = ({"holdout_signer": holdout} if holdout
              else {"random_split": 0.2, "split_seed": seed})
        Xtr, ytr, _, l2i = load_split("train", lang, source_mode=mode, **kw)
        Xev, yev, _, _ = load_split("val", lang, source_mode="real",
                                    label_to_idx=l2i, **kw)
        labels = [k for k, _ in sorted(l2i.items(), key=lambda kv: kv[1])]

        for algo in CLASSICAL:
            m, _, _ = build_model(algo)
            m.fit(Xtr, ytr)
            p = m.predict(Xev)
            res[algo]["acc"].append(accuracy_score(yev, p) * 100)
            res[algo]["f1"].append(f1_score(yev, p, average="macro") * 100)
            conf.setdefault(algo, []).append(confusion_matrix(yev, p,
                                                             labels=range(len(l2i))))
        # GRU on the raw sequence
        try:
            Str, _, _, _ = sequences("train", lang, mode, **kw)
            Sev, _, _, _ = sequences("val", lang, "real", l2i=l2i, **kw)
            p = run_gru(Str, ytr, Sev, yev, len(l2i), seed=seed)
            res["gru"]["acc"].append(accuracy_score(yev, p) * 100)
            res["gru"]["f1"].append(f1_score(yev, p, average="macro") * 100)
            conf.setdefault("gru", []).append(
                confusion_matrix(yev, p, labels=range(len(l2i))))
        except Exception as e:                       # torch missing / OOM
            print(f"  (gru skipped: {type(e).__name__}: {e})")
    return res, conf, labels


def bar_chart(res, title, path, subtitle=None):
    algos = [a for a in res if res[a]["acc"]]
    algos.sort(key=lambda a: np.mean(res[a]["acc"]), reverse=True)
    acc = [np.mean(res[a]["acc"]) / 100 for a in algos]
    f1 = [np.mean(res[a]["f1"]) / 100 for a in algos]
    err = [np.std(res[a]["acc"]) / 100 for a in algos]

    x = np.arange(len(algos)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, acc, w, label="Accuracy", color="#4472C4",
           yerr=err, capsize=3, ecolor="#33415580")
    ax.bar(x + w/2, f1, w, label="Macro F1", color="#ED7D31")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(a, a) for a in algos], rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.015, f"{v*100:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grouped_chart(series, algos, title, path, ylabel="Accuracy (%)"):
    x = np.arange(len(algos)); n = len(series); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=colors[i % 4])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(a, a) for a in algos], rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", alpha=0.25); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def confusion_chart(cm, labels, texts, title, path):
    cm = cm.astype(float)
    row = cm.sum(1, keepdims=True); row[row == 0] = 1
    norm = cm / row
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    disp = [texts.get(l, l) for l in labels]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(disp, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(disp)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="khmer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--holdout", default=None,
                    help="signer to hold out for the cross-signer chart")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(a.seeds))
    t0 = time.time()

    labels_text = {}
    lp = ROOT / "data" / "sequences_v2" / a.lang / "labels.json"
    if lp.exists():
        labels_text = json.loads(lp.read_text(encoding="utf-8"))

    # Real recordings only — no synthetic augmentation anywhere in this report.
    print(f"[1/3] training all algorithms ({a.seeds} splits, real data only)...")
    same_real, conf, labels = evaluate_all(a.lang, seeds, mode="real")
    same_both = {}

    # cross-signer: hold out each signer in turn
    from src.v2.dataset import discover_samples
    from src.v2.schema import Source
    signers = sorted({m.signer_id for _, m in
                      discover_samples(ROOT / "data" / "sequences_v2", language=a.lang)
                      if m.source is Source.REAL})
    cross_real, cross_both = {}, {}
    if len(signers) > 1:
        for i, sg in enumerate(signers):
            print(f"[2/3] cross-signer, holdout {i+1}/{len(signers)}...")
            r, _, _ = evaluate_all(a.lang, [0], holdout=sg, mode="real")
            for algo in r:
                cross_real.setdefault(algo, []).extend(r[algo]["acc"])

    print("[3/3] charts...")
    bar_chart(same_real, "Khmer sign recognition: algorithm comparison",
              OUT / "comparison_chart.png",
              f"{a.lang} · {len(labels)} signs · {len(signers)} signers · "
              f"held-out takes (mean of {a.seeds} splits)")

    algos = [x for x in (CLASSICAL + ["gru"]) if same_real[x]["acc"]]
    algos.sort(key=lambda x: np.mean(same_real[x]["acc"]), reverse=True)
    if cross_real:
        grouped_chart(
            {"Same signer (seen)": [np.mean(same_real[x]["acc"]) for x in algos],
             "Unseen signer": [np.mean(cross_real.get(x, [0])) for x in algos]},
            algos, "Generalization gap: seen vs unseen signer",
            OUT / "cross_signer.png")

    best = algos[0]
    if best in conf:
        confusion_chart(sum(conf[best]), labels, labels_text,
                        f"Confusion matrix — {PRETTY.get(best, best)}",
                        OUT / f"confusion_{best}.png")

    out = {
        "language": a.lang, "labels": labels, "labels_text": labels_text,
        "signers": signers, "splits": a.seeds,
        "same_signer_real": {k: {m: v[m] for m in ("acc", "f1")}
                             for k, v in same_real.items() if v["acc"]},
        "same_signer_both": {k: {m: v[m] for m in ("acc", "f1")}
                             for k, v in same_both.items() if v["acc"]},
        "cross_signer_real": cross_real, "cross_signer_both": cross_both,
        "best": best, "seconds": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\ndone in {out['seconds']}s -> {OUT}")
    print(f"{'algorithm':<22}{'accuracy':>10}{'macro F1':>10}{'unseen signer':>15}")
    print("-" * 58)
    for x in algos:
        cs = f"{np.mean(cross_real[x]):.1f}%" if x in cross_real else "—"
        print(f"{PRETTY.get(x,x):<22}{np.mean(same_real[x]['acc']):>9.1f}%"
              f"{np.mean(same_real[x]['f1']):>9.1f}%{cs:>15}")


if __name__ == "__main__":
    main()
