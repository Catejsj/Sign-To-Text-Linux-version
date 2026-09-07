"""Task A — the complete khmer_var study, computed in one pass.

Answers the five questions asked of this project, plus the audit that says
whether any of the answers can be believed:

    1. improve detection on similar signs
    2. more data on the lower scores
    3. a DECISION on the algorithm, not a table dump
    4. which features actually carry the signal
    5. the random split re-tested as cross-validation

Everything is measured on `khmer_var` only — 337 real takes of 7 signs by 4
signers on a deliberate 12-cell grid. Nothing from any other corpus.

    python algo_comparison/run_task_a.py                # compute + charts
    python algo_comparison/make_task_a_report.py        # write the .docx

Splitting is take-aware throughout: a take's clean view, noisy view and every
synthetic copy stay on one side, and evaluation is always on real takes.
Both a same-signer and an unseen-signer score are produced for every model,
because the features identify the signer almost perfectly and a same-signer
number alone would flatter every result (see the audit, §F).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib import font_manager                                 # noqa: E402

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
from torch.utils.data import DataLoader, TensorDataset              # noqa: E402
from sklearn.ensemble import RandomForestClassifier                 # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,      # noqa: E402
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold            # noqa: E402

from src.v2.algorithms import registry, wrap                        # noqa: E402
from src.v2.baseline_data import _featurize                         # noqa: E402
from src.v2.bones import bone_vectors, summary_bones                # noqa: E402
from src.v2.dataset import (discover_samples, synthetic_ratios,     # noqa: E402
                            take_id)
from src.v2.landmarks import (dropout_report, hand_presence,        # noqa: E402
                              summary_valid, zero_missing)
from src.v2.model_rnn import SignRNN                                # noqa: E402
from src.v2.model_tcn import SignTCN                                # noqa: E402
from src.v2.model_transformer import SignTransformer                # noqa: E402
from src.v2.schema import Source, View                              # noqa: E402

import argparse                                                     # noqa: E402
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--lang", default="khmer_var")
ap.add_argument("--grid", type=int, default=12,
                help="takes per sign in the recording grid")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--quick", action="store_true",
                help="skip the deep models — classical only, ~1 minute")
ap.add_argument("--no-conditions", action="store_true",
                help="the corpus has no lighting/distance/position grid, so "
                     "condition transfer is meaningless. Task B is freestyle; "
                     "only Task A (khmer_var) was recorded on a grid.")
ap.add_argument("--cap-per-sign", type=int, default=None, metavar="N",
                help="use at most N real takes per (signer, sign), lowest "
                     "variant numbers first. Makes an uneven corpus uniform "
                     "for reporting without deleting anything. Task B was "
                     "specified as 30 per sign; some people recorded more.")
ap.add_argument("--tag", default=None,
                help="suffix for the output folder, so two runs on the same "
                     "corpus do not overwrite each other")
ap.add_argument("--charts-only", action="store_true",
                help="redraw the charts from an existing results.json without "
                     "recomputing anything")
A = ap.parse_args()

LANG, GRID, K, EPOCHS = A.lang, A.grid, A.folds, A.epochs
DATA = ROOT / "data" / "sequences_v2"
CAP = A.cap_per_sign
OUT = ROOT / "algo_comparison" / (
    f"results_{LANG}_taskA" + (f"_{A.tag}" if A.tag else ""))
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

KHMER_FONT = ROOT / "fonts" / "NotoSansKhmer-Regular.ttf"
if KHMER_FONT.exists():
    font_manager.fontManager.addfont(str(KHMER_FONT))
    KH = font_manager.FontProperties(fname=str(KHMER_FONT), size=9)
else:
    KH = None
BLUE, GREEN, GREY, RED, AMBER = "#4472C4", "#70AD47", "#A6A6A6", "#C00000", "#ED7D31"

# ── charts (defined early so --charts-only can skip the compute) ──
def draw_charts(R):
    """Everything below reads only from R, so --charts-only can redraw
    without re-running a single model."""
    print("charts ...")
    ALGOS = list(R["classical"])
    classical, deep = R["classical"], R["deep"]
    LABELS, NAME = R["labels"], R["label_text"]
    best_c = R["decision"]["best_classical"]
    best_d = R["decision"]["best_deep"]
    cm = np.array(R["confusion"])
    per_f1 = R["per_sign"]
    split_cmp = R["split_comparison"]["per_algo"]
    K = R["folds"]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False})


    def save(fig, name):
        fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)


    # 1 — the journey
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    stages = ["original", "hand fix", "+ bones"]
    cls = [np.mean([classical[a][f]["unseen"]["f1"] for a in ALGOS])
           for f in ("original", "hand_fix", "bones")]
    ax.plot(stages, cls, "o-", color=BLUE, lw=2.5, ms=8, label="classical (mean)")
    if deep:
        ax.axhline(deep[best_d]["unseen"], color=GREEN, ls="--", lw=2,
                   label=f"best deep ({best_d}) {deep[best_d]['unseen']:.1f}")
    ax.set_ylabel("macro-F1 on an unseen signer (%)")
    ax.set_title("What each fix was worth", fontweight="bold")
    for i, v in enumerate(cls):
        ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontweight="bold")
    ax.legend(frameon=False); ax.set_ylim(0, 100)
    save(fig, "ta_journey.png")

    # 2 — same vs unseen, every model
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    rows = [(a, classical[a]["bones"]["same"]["f1"],
             classical[a]["bones"]["unseen"]["f1"], BLUE) for a in ALGOS]
    rows += [(k, v["same"], v["unseen"], GREEN) for k, v in deep.items()]
    rows.sort(key=lambda t: t[2])
    ypos = np.arange(len(rows))
    ax.barh(ypos - .2, [r[1] for r in rows], .38, color=GREY, label="same signer")
    ax.barh(ypos + .2, [r[2] for r in rows], .38,
            color=[r[3] for r in rows], label="unseen signer")
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("macro-F1 (%)"); ax.set_xlim(0, 100)
    ax.set_title("Every model, scored both ways", fontweight="bold", pad=26)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="same signer"),
                       Patch(color=BLUE, label="unseen signer — classical"),
                       Patch(color=GREEN, label="unseen signer — deep")],
              frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), fontsize=8)
    save(fig, "ta_models.png")

    # 3 — confusion
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    if KH:
        ax.set_xticklabels(NAME, fontproperties=KH, rotation=45, ha="right")
        ax.set_yticklabels(NAME, fontproperties=KH)
    else:
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 55 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion — {best_c}, pooled over {K} folds", fontweight="bold")
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=.8, label="% of the true sign")
    save(fig, "ta_confusion.png")

    # 4 — per sign
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    order = sorted(LABELS, key=lambda l: per_f1[l]["f1"])
    vals = [per_f1[l]["f1"] for l in order]
    cols = [RED if v < 75 else AMBER if v < 88 else GREEN for v in vals]
    ax.bar(range(len(order)), vals, color=cols)
    ax.set_xticks(range(len(order)))
    if KH:
        ax.set_xticklabels([per_f1[l]["text"] for l in order], fontproperties=KH)
    else:
        ax.set_xticklabels(order, rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 105)
    ax.set_title("Per-sign score — where more recordings would help",
                 fontweight="bold")
    save(fig, "ta_per_sign.png")

    # 5 — split protocol
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.arange(len(ALGOS))
    ax.bar(xs - .2, [split_cmp[a]["random_f1"] for a in ALGOS], .38, color=GREY,
           yerr=[split_cmp[a]["random_sd"] for a in ALGOS], capsize=3,
           label="8 random draws")
    ax.bar(xs + .2, [split_cmp[a]["cv_f1"] for a in ALGOS], .38, color=BLUE,
           yerr=[split_cmp[a]["cv_sd"] for a in ALGOS], capsize=3,
           label=f"{K}-fold cross-validation")
    ax.set_xticks(xs); ax.set_xticklabels(ALGOS, rotation=30, ha="right")
    ax.set_ylabel("macro-F1 (%)")
    ax.set_title("Same data, two protocols — note the error bars",
                 fontweight="bold")
    ax.legend(frameon=False)
    save(fig, "ta_split.png")

    # 6 — features
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax_, (title, d) in zip(axes, [
            ("by body part", R["feature_importance"]["by_part"]),
            ("by coordinate", R["feature_importance"]["by_coord"]),
            ("by statistic", R["feature_importance"]["by_stat"])]):
        ks = list(d); vs = [d[k] for k in ks]
        ax_.bar(ks, vs, color=[RED if v == max(vs) else BLUE for v in vs])
        ax_.set_title(title, fontsize=10, fontweight="bold")
        ax_.set_ylabel("% importance" if title.endswith("part") else "")
        ax_.tick_params(axis="x", rotation=20)
    save(fig, "ta_features.png")



if A.charts_only:
    R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    draw_charts(R)
    print(f"redrew charts in {OUT}")
    raise SystemExit(0)


R: dict = {"lang": LANG, "grid": GRID, "folds": K, "epochs": EPOCHS,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S")}


# ── data ─────────────────────────────────────────────────────────────
print(f"loading {LANG} ...")
samples = [(p, m) for p, m in discover_samples(DATA, language=LANG)
           if m.view is View.CLEAN]
RATIOS = synthetic_ratios(samples)

clips, ys, sgs, gids, slots, tids = [], [], [], [], [], []
syn_clips, syn_y, syn_g, syn_sg = [], [], [], []
gmap: dict = {}
for p, m in samples:
    t = take_id(m, RATIOS)
    g = gmap.setdefault(t, len(gmap))
    c = np.load(p).astype(np.float32)
    if m.source is Source.REAL:
        clips.append(c); ys.append(m.label); sgs.append(m.signer_id)
        gids.append(g); tids.append(t); slots.append(m.variant % GRID)
    else:
        syn_clips.append(c); syn_y.append(m.label); syn_g.append(g)
        syn_sg.append(m.signer_id)

if CAP:
    # Keep the lowest variant numbers per (signer, label) so the choice is
    # deterministic and reproducible rather than a random sample. Synthetic
    # children of a dropped take go with it, or the ratio breaks.
    from collections import Counter
    order = defaultdict(list)
    for i, (lab, s) in enumerate(zip(ys, sgs)):
        order[(s, lab)].append(i)
    keep = set()
    for key, idxs in order.items():
        idxs.sort(key=lambda i: tids[i][2])      # take variant
        keep.update(idxs[:CAP])
    kept_groups = {gids[i] for i in keep}
    dropped = len(ys) - len(keep)
    clips = [c for i, c in enumerate(clips) if i in keep]
    ys = [v for i, v in enumerate(ys) if i in keep]
    sgs = [v for i, v in enumerate(sgs) if i in keep]
    slots = [v for i, v in enumerate(slots) if i in keep]
    tids = [v for i, v in enumerate(tids) if i in keep]
    gids = [v for i, v in enumerate(gids) if i in keep]
    keep_syn = [i for i, g in enumerate(syn_g) if g in kept_groups]
    syn_clips = [syn_clips[i] for i in keep_syn]
    syn_y = [syn_y[i] for i in keep_syn]
    syn_g = [syn_g[i] for i in keep_syn]
    print(f"  cap {CAP}/sign: kept {len(ys)} real takes, dropped {dropped}")

LABELS = sorted(set(ys)); l2i = {l: i for i, l in enumerate(LABELS)}
y = np.array([l2i[t] for t in ys]); groups = np.array(gids)
sg = np.array(sgs); slot = np.array(slots)
SIGNERS = sorted(set(sgs)); N = len(clips)
TEXTS = json.loads((DATA / LANG / "labels.json").read_text(encoding="utf-8"))
NAME = [TEXTS.get(l, l) for l in LABELS]

R.update(n_real=N, n_synth=len(syn_clips), cap_per_sign=CAP,
         labels=LABELS, label_text=NAME,
         signers=SIGNERS,
         per_signer={s: int((sg == s).sum()) for s in SIGNERS})
print(f"  {N} real + {len(syn_clips)} synthetic · {len(LABELS)} signs "
      f"· {len(SIGNERS)} signers · {DEV}")

X = np.stack([summary_bones(c) for c in clips])
Xv = np.stack([summary_valid(c) for c in clips])       # pre-bones, for the delta
Xo = np.stack([_featurize(c, "summary") for c in clips])   # the original


def deep_x(cs):
    out = []
    for c in cs:
        p = hand_presence(c)
        out.append(np.concatenate(
            [zero_missing(c, p).reshape(len(c), -1), p,
             bone_vectors(c, unit=True).reshape(len(c), -1)], axis=1))
    return np.stack(out).astype(np.float32)


XD = deep_x(clips)
R["n_features"] = {"original": int(Xo.shape[1]), "hand_fix": int(Xv.shape[1]),
                   "bones": int(X.shape[1]), "deep": int(XD.shape[2])}

TABLE, _ = registry()
ALGOS = [a for a in TABLE if a != "nb"]
R["custom_algos"] = [a for a in TABLE if TABLE[a][2] != "built-in"]

FOLDS = list(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=0)
             .split(np.zeros(N), y, groups))


def cv(Xf, fac, yy=None):
    yy = y if yy is None else yy
    acc, f1, pre, rec = [], [], [], []
    for tr, te in FOLDS:
        m = wrap(fac()); m.fit(Xf[tr], yy[tr]); p = m.predict(Xf[te])
        acc.append(accuracy_score(yy[te], p) * 100)
        f1.append(f1_score(yy[te], p, average="macro", zero_division=0) * 100)
        pre.append(precision_score(yy[te], p, average="macro", zero_division=0) * 100)
        rec.append(recall_score(yy[te], p, average="macro", zero_division=0) * 100)
    return dict(acc=float(np.mean(acc)), f1=float(np.mean(f1)),
                pre=float(np.mean(pre)), rec=float(np.mean(rec)),
                sd=float(np.std(f1)))


def loso(Xf, fac, yy=None):
    yy = y if yy is None else yy
    f1, per = [], {}
    for s in SIGNERS:
        te = sg == s
        m = wrap(fac()); m.fit(Xf[~te], yy[~te]); p = m.predict(Xf[te])
        v = f1_score(yy[te], p, average="macro", zero_division=0) * 100
        f1.append(v); per[s] = float(v)
    return dict(f1=float(np.mean(f1)), sd=float(np.std(f1)), per_signer=per)


# ── 5. cross-validation vs the old repeated random split ─────────────
print("ask 5: cross-validation vs repeated random split ...")


def random_draws(seeds=8):
    takes = np.unique(groups)
    for s in range(seeds):
        rs = np.random.default_rng(s)
        held = set(rs.choice(takes, size=max(1, len(takes) // 4),
                             replace=False).tolist())
        yield np.array([g in held for g in groups])


split_cmp = {}
for a in ALGOS:
    _l, fac, _o = TABLE[a]
    rnd = []
    for ev in random_draws():
        m = wrap(fac()); m.fit(X[~ev], y[~ev])
        rnd.append(f1_score(y[ev], m.predict(X[ev]), average="macro",
                            zero_division=0) * 100)
    c = cv(X, fac)
    split_cmp[a] = {"random_f1": float(np.mean(rnd)),
                    "random_sd": float(np.std(rnd)),
                    "cv_f1": c["f1"], "cv_sd": c["sd"]}

seen = defaultdict(int)
for ev in random_draws():
    for g in np.unique(groups[ev]):
        seen[g] += 1
never = sum(1 for t in np.unique(groups) if seen[t] == 0)
R["split_comparison"] = {
    "per_algo": split_cmp, "n_takes": int(len(np.unique(groups))),
    "never_tested_random": int(never),
    "times_tested_min": int(min(seen.values(), default=0)),
    "times_tested_max": int(max(seen.values(), default=0)),
    "mean_sd_random": float(np.mean([v["random_sd"] for v in split_cmp.values()])),
    "mean_sd_cv": float(np.mean([v["cv_sd"] for v in split_cmp.values()])),
}

# ── classical: all algorithms, both protocols, three feature sets ────
print("classical: 9 algorithms x 3 feature sets x 2 protocols ...")
feat_sets = {"original": Xo, "hand_fix": Xv, "bones": X}
classical = {}
for a in ALGOS:
    _l, fac, origin = TABLE[a]
    classical[a] = {"name": TABLE[a][0], "origin": origin}
    for fname, Xf in feat_sets.items():
        classical[a][fname] = {"same": cv(Xf, fac), "unseen": loso(Xf, fac)}
R["classical"] = classical

# ── deep ─────────────────────────────────────────────────────────────
NC, NF = len(LABELS), XD.shape[2]
DEEP_MAKERS = {
    "gru":         lambda: SignRNN(NC, NF, cell="gru", bidirectional=False),
    "bigru":       lambda: SignRNN(NC, NF, cell="gru", bidirectional=True),
    "lstm":        lambda: SignRNN(NC, NF, cell="lstm", bidirectional=False),
    "bilstm":      lambda: SignRNN(NC, NF, cell="lstm", bidirectional=True),
    "tcn":         lambda: SignTCN(NC, in_features=NF),
    "transformer": lambda: SignTransformer(NC, feature_dim=NF),
}
LIVE_CAPABLE = {"gru", "lstm", "tcn", "transformer"}


def fit_deep(make, tr, te, seed=0):
    torch.manual_seed(seed)
    net = make().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(XD[tr]),
                                  torch.from_numpy(y[tr])),
                    batch_size=32, shuffle=True)
    net.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); crit(net(xb), yb).backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(XD[te]).to(DEV)).argmax(1).cpu().numpy()


deep = {}
if not A.quick:
    print(f"deep: 6 architectures x {EPOCHS} epochs ...")
    for name, make in DEEP_MAKERS.items():
        t0 = time.time()
        cvf = [f1_score(y[te], fit_deep(make, tr, te), average="macro",
                        zero_division=0) * 100 for tr, te in FOLDS]
        lo, per = [], {}
        for s in SIGNERS:
            te = np.where(sg == s)[0]; tr = np.where(sg != s)[0]
            v = f1_score(y[te], fit_deep(make, tr, te), average="macro",
                         zero_division=0) * 100
            lo.append(v); per[s] = float(v)
        deep[name] = dict(
            same=float(np.mean(cvf)), same_sd=float(np.std(cvf)),
            unseen=float(np.mean(lo)), unseen_sd=float(np.std(lo)),
            per_signer=per, live_capable=name in LIVE_CAPABLE,
            params=int(sum(p.numel() for p in make().parameters()
                           if p.requires_grad)),
            secs=round(time.time() - t0, 1))
        print(f"  {name:<12} same {np.mean(cvf):5.1f}  unseen {np.mean(lo):5.1f}")
R["deep"] = deep

# ── the decision (ask 3) ─────────────────────────────────────────────
best_c = max(ALGOS, key=lambda a: classical[a]["bones"]["unseen"]["f1"])
best_d = max(deep, key=lambda k: deep[k]["unseen"]) if deep else None
best_live = (max((k for k in deep if deep[k]["live_capable"]),
                 key=lambda k: deep[k]["unseen"]) if deep else None)
R["decision"] = {
    "best_classical": best_c,
    "best_classical_unseen": classical[best_c]["bones"]["unseen"]["f1"],
    "best_deep": best_d,
    "best_deep_unseen": deep[best_d]["unseen"] if best_d else None,
    "best_live": best_live,
    "best_live_unseen": deep[best_live]["unseen"] if best_live else None,
}

# ── 1 + 2. confusion and per-sign, using the chosen model ────────────
print("asks 1 & 2: confusion and per-sign ...")
_l, fac_best, _o = TABLE[best_c]
cm = np.zeros((len(LABELS), len(LABELS)), int)
cond_hits = defaultdict(lambda: [0, 0])
for tr, te in FOLDS:
    m = wrap(fac_best()); m.fit(X[tr], y[tr]); p = m.predict(X[te])
    cm += confusion_matrix(y[te], p, labels=range(len(LABELS)))
    if not A.no_conditions:
        # `slot` only means anything on a corpus recorded to a grid. On a
        # freestyle corpus it is variant % GRID, which is not a condition.
        for t_, pr, s_ in zip(y[te], p, slot[te]):
            light = "full" if s_ < 6 else "dim"
            dist = "near" if (s_ % 6) < 3 else "far"
            cell = cond_hits[(LABELS[t_], light, dist)]
            cell[1] += 1; cell[0] += int(t_ == pr)

per_f1 = {}
for i, lab in enumerate(LABELS):
    tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    reca = tp / (tp + fn) if tp + fn else 0.0
    per_f1[lab] = {"f1": float(200 * prec * reca / (prec + reca)) if prec + reca else 0.0,
                   "precision": float(prec * 100), "recall": float(reca * 100),
                   "n": int(cm[i].sum()), "text": NAME[i]}
pairs = []
for i in range(len(LABELS)):
    for j in range(i + 1, len(LABELS)):
        n = int(cm[i, j] + cm[j, i])
        if n:
            pairs.append({"a": LABELS[i], "a_text": NAME[i],
                          "b": LABELS[j], "b_text": NAME[j], "errors": n,
                          "a_to_b": int(cm[i, j]), "b_to_a": int(cm[j, i])})
pairs.sort(key=lambda d: -d["errors"])
R.update(confusion=cm.tolist(), per_sign=per_f1, confused_pairs=pairs,
         conditions_per_sign={f"{k[0]}|{k[1]}|{k[2]}": v
                              for k, v in cond_hits.items()})

# ── 4. which features ────────────────────────────────────────────────
print("ask 4: feature importance ...")
rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
rf.fit(X, y)
imp = rf.feature_importances_
NV = Xv.shape[1]                       # summary_valid block, then bones
imp_joint, imp_bone = float(imp[:NV].sum()), float(imp[NV:].sum())
rf2 = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
rf2.fit(Xo, y)
i2 = rf2.feature_importances_
STAT = ["mean", "std", "min", "max"]
part = {"body": range(6), "left hand": range(6, 27), "right hand": range(27, 48)}
R["feature_importance"] = {
    "joints_block": imp_joint * 100, "bones_block": imp_bone * 100,
    "by_part": {k: float(sum(i2[s * 144 + j * 3 + c] for s in range(4)
                             for j in v for c in range(3)) * 100)
                for k, v in part.items()},
    "by_coord": {c: float(sum(i2[s * 144 + j * 3 + ci] for s in range(4)
                              for j in range(48)) * 100)
                 for ci, c in enumerate("xyz")},
    "by_stat": {s: float(sum(i2[si * 144 + j * 3 + c] for j in range(48)
                             for c in range(3)) * 100)
                for si, s in enumerate(STAT)},
    "half_importance_in": int(np.searchsorted(np.cumsum(np.sort(imp)[::-1]),
                                              0.5) + 1),
    "total_features": int(X.shape[1]),
}

# ── conditions ───────────────────────────────────────────────────────
print("condition transfer ...")


def cond_transfer(train_mask, test_mask, fac):
    m = wrap(fac()); m.fit(X[train_mask], y[train_mask])
    return float(f1_score(y[test_mask], m.predict(X[test_mask]),
                          average="macro", zero_division=0) * 100)


R["has_grid"] = not A.no_conditions
R["condition_transfer"] = [] if A.no_conditions else [
    {"axis": "Lighting", "trained_on": "full light", "tested_on": "dim light",
     "f1": cond_transfer(slot < 6, slot >= 6, fac_best)},
    {"axis": "Distance", "trained_on": "near", "tested_on": "far",
     "f1": cond_transfer((slot % 6) < 3, (slot % 6) >= 3, fac_best)},
    {"axis": "Position", "trained_on": "middle/left", "tested_on": "right",
     "f1": cond_transfer((slot % 3) < 2, (slot % 3) == 2, fac_best)},
]

# ── synthetic ────────────────────────────────────────────────────────
print("synthetic: real vs real+synthetic ...")
Xs = np.stack([summary_bones(c) for c in syn_clips]) if syn_clips else None
syn_gr = np.array(syn_g)
syn_yy = np.array([l2i[t] for t in syn_y]) if syn_y else None
synth = {}
if Xs is not None:
    for a in ALGOS:
        _l, fac, _o = TABLE[a]
        f1r, f1b = [], []
        for tr, te in FOLDS:
            held = set(groups[te])
            add = ~np.isin(syn_gr, list(held))       # synthetic of held-out OUT
            m = wrap(fac()); m.fit(X[tr], y[tr])
            f1r.append(f1_score(y[te], m.predict(X[te]), average="macro",
                                zero_division=0) * 100)
            Xb = np.concatenate([X[tr], Xs[add]])
            yb = np.concatenate([y[tr], syn_yy[add]])
            m2 = wrap(fac()); m2.fit(Xb, yb)
            f1b.append(f1_score(y[te], m2.predict(X[te]), average="macro",
                                zero_division=0) * 100)
        synth[a] = {"real": float(np.mean(f1r)), "both": float(np.mean(f1b)),
                    "delta": float(np.mean(f1b) - np.mean(f1r))}
R["synthetic"] = synth

# ── audit ────────────────────────────────────────────────────────────
print("audit ...")
h = defaultdict(list)
for i, c in enumerate(clips):
    h[hashlib.blake2b(np.ascontiguousarray(c).tobytes(),
                      digest_size=16).hexdigest()].append(i)
dups = {k: v for k, v in h.items() if len(v) > 1}
cross = [v for v in dups.values() if len({tids[i] for i in v}) > 1]

rng = np.random.default_rng(0)
uniq = np.unique(groups)
take_label = {g: y[groups == g][0] for g in uniq}
perm = rng.permutation(len(uniq))
shuffled = {g: take_label[uniq[perm[i]]] for i, g in enumerate(uniq)}
y_shuf = np.array([shuffled[g] for g in groups])
s2i = {s: i for i, s in enumerate(SIGNERS)}
y_sig = np.array([s2i[s] for s in sgs])

R["audit"] = {
    "duplicate_groups": len(dups),
    "cross_take_duplicates": len(cross),
    "folds_disjoint": all(not (set(groups[tr]) & set(groups[te]))
                          for tr, te in FOLDS),
    "permutation_same": cv(X, fac_best, y_shuf)["f1"],
    "permutation_unseen": loso(X, fac_best, y_shuf)["f1"],
    "chance": 100.0 / len(LABELS),
    "signer_probe": cv(X, fac_best, y_sig)["f1"],
    "signer_chance": 100.0 / len(SIGNERS),
}
R["hand_dropout"] = dropout_report(clips)

draw_charts(R)
(OUT / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
print(f"\nwrote {OUT}/results.json and 6 charts")
print(f"decision: classical={best_c} ({R['decision']['best_classical_unseen']:.1f}) "
      f"deep={best_d} ({R['decision']['best_deep_unseen']:.1f}) "
      f"live={best_live}")
