"""Do camera images add anything to landmarks? Three models, same test takes.

    A  landmarks only   the TCN and features run_task_a uses (290 per frame)
    B  images only      a frozen ImageNet ResNet-18 per frame, then the same TCN
    C  both             landmark and image features side by side, per frame

Every arm is scored on the SAME held-out takes of the multimodal folder
(default khmer_mm), so C minus A is the gain from adding images and nothing
else. That difference is the number this script exists to produce.

USAGE
-----
    # your takes alone: a pipeline check, NOT a finding (see PROTOCOL)
    python algo_comparison/run_multimodal.py

    # add the old landmark-only corpus to A and C's training data
    python algo_comparison/run_multimodal.py --extra khmer

    # the separate synthetic comparison
    python algo_comparison/run_multimodal.py --extra khmer --synthetic --tag syn

PROTOCOL
--------
With two or more people in the multimodal folder it is leave-one-signer-out:
each person is held out in turn, and **all** of their data leaves training,
including their takes in the --extra folder. Without that, the "unseen" person
has been seen. With one person it falls back to grouped 5-fold cross-
validation over their takes, which only says whether images help on someone
the model already knows. Images make that easier than landmarks do (same face,
clothes, room), so a gain there is not evidence for the research question.

Old landmark-only takes have no images. In C they enter with the image part
zeroed and a has-image flag of 0; they are never tested on. Synthetic copies
(--synthetic) are landmark-only too, train-side only, and always follow their
parent take: a test take's copies are never trained on.

Image features are cached per take under data/media/_cache/ and recomputed if
the take changes. The first run downloads ResNet-18's ImageNet weights (about
45 MB) into torch's cache. Output: algo_comparison/results_multimodal_<lang>
[_tag]/results.json, with every fold, per-take predictions, and the
hand-dropout subset (do images help most where the hand tracker lost a hand?).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
from torch.utils.data import DataLoader, TensorDataset              # noqa: E402
from sklearn.metrics import accuracy_score, f1_score                # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold            # noqa: E402

from src.v2.bones import bone_vectors                               # noqa: E402
from src.v2.dataset import (discover_samples, synthetic_ratios,     # noqa: E402
                            take_id)
from src.v2.landmarks import hand_presence, zero_missing            # noqa: E402
from src.v2.model_tcn import SignTCN                                # noqa: E402
from src.v2.schema import SEQ_LEN, Source, View                     # noqa: E402

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lang", default="khmer_mm",
                help="folder with landmarks AND media (default khmer_mm)")
ap.add_argument("--extra", default=None,
                help="landmark-only folder added to A and C's training, "
                     "e.g. khmer. Never tested on")
ap.add_argument("--extra-cap", type=int, default=30,
                help="at most N takes per person per sign from --extra, lowest "
                     "numbers first, as in the cap30 results (0 = all)")
ap.add_argument("--synthetic", action="store_true",
                help="add the synthetic copies of every training take "
                     "(landmark arms only)")
ap.add_argument("--protocol", default="auto", choices=["auto", "loso", "cv"],
                help="auto = leave-one-signer-out if 2+ people, else 5-fold")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--crop", default="full", choices=["full", "body"],
                help="full camera frame, or a box around the signer's joints "
                     "(drops most of the room, keeps the face)")
ap.add_argument("--arms", default="A,B,C")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default=None)
A = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ = ROOT / "data" / "sequences_v2"
MEDIA = ROOT / "data" / "media"
ARMS = [a.strip().upper() for a in A.arms.split(",") if a.strip()]
OUT = ROOT / "algo_comparison" / (
    f"results_multimodal_{A.lang}" + (f"_{A.tag}" if A.tag else ""))
IMG_DIM = 512


# ── landmark features: exactly run_task_a's deep input ──────────────
def deep_x(c: np.ndarray) -> np.ndarray:
    p = hand_presence(c)
    return np.concatenate(
        [zero_missing(c, p).reshape(len(c), -1), p,
         bone_vectors(c, unit=True).reshape(len(c), -1)],
        axis=1).astype(np.float32)                         # (60, 290)


def load_clean(lang: str):
    """Clean views of a folder: (path, meta, take_id) for real and synthetic."""
    s = [(p, m) for p, m in discover_samples(SEQ, language=lang)
         if m.view is View.CLEAN]
    r = synthetic_ratios(s)
    return [(p, m, take_id(m, r)) for p, m in s]


# ── image features ───────────────────────────────────────────────────
_NET = None


def backbone():
    global _NET
    if _NET is None:
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        _NET = net.eval().to(DEV)
    return _NET


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def body_box(noisy: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Square box around every joint the tracker really found, +15%."""
    c = zero_missing(noisy, hand_presence(noisy))[..., :2].reshape(-1, 2)
    c = c[np.abs(c).sum(1) > 0]
    if len(c) == 0:
        return 0, 0, w, h
    x0, y0 = c.min(0) * (w, h)
    x1, y1 = c.max(0) * (w, h)
    side = max(x1 - x0, y1 - y0) * 1.15
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(side, w, h)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(side), int(side)


def read_frames(side: Path, meta: dict) -> list[np.ndarray]:
    fdir = side.parent / f"{side.stem}_frames"
    if fdir.is_dir():
        return [cv2.imread(str(f)) for f in sorted(fdir.glob("*.jpg"))]
    cap = cv2.VideoCapture(str(side.with_suffix(".mp4")))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def embed_take(side: Path, noisy_npy: Path) -> np.ndarray:
    """(n_frames, 512) ResNet-18 features for one media take, cached."""
    rel = side.relative_to(MEDIA)
    cache = MEDIA / "_cache" / f"resnet18_{A.crop}" / rel.with_suffix(".npz")
    stamp = side.stat().st_mtime_ns
    if cache.exists():
        z = np.load(cache)
        if int(z["stamp"]) == stamp:
            return z["emb"]
    meta = json.loads(side.read_text(encoding="utf-8"))
    frames = read_frames(side, meta)
    if A.crop == "body":
        h, w = frames[0].shape[:2]
        x, y, s, _ = body_box(np.load(noisy_npy), w, h)
        frames = [f[y:y + s, x:x + s] for f in frames]
    batch = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB),
                                 (224, 224)) for f in frames])
    batch = ((batch.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2)
    embs = []
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            embs.append(backbone()(torch.from_numpy(batch[i:i + 64]).to(DEV))
                        .cpu().numpy())
    emb = np.concatenate(embs).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, stamp=stamp)
    return emb


def align(emb: np.ndarray, lm: list | None) -> np.ndarray:
    """Image features at the 60 time points of the landmark clip.

    The landmark clip is the take's pose frames stretched evenly to 60, and
    each image records which pose frame it was captured with (`lm`). So image
    features are interpolated at the same 60 pose-frame positions. Images with
    no pose frame are left out, exactly as the landmark clip leaves them out.
    Sidecars from before `lm` existed fall back to even spacing.
    """
    pairs = [(l, i) for i, l in enumerate(lm or []) if l is not None]
    if len(pairs) >= 2:
        xp = np.array([p[0] for p in pairs], np.float64)
        src = emb[[p[1] for p in pairs]]
        pos = np.linspace(0, xp.max(), SEQ_LEN)
    else:
        xp = np.arange(len(emb), dtype=np.float64)
        src = emb
        pos = np.linspace(0, len(emb) - 1, SEQ_LEN)
    return np.stack([np.interp(pos, xp, src[:, d])
                     for d in range(src.shape[1])], axis=1).astype(np.float32)


# ── data ─────────────────────────────────────────────────────────────
t0 = time.time()
print(f"loading {A.lang} ...")
mm_all = load_clean(A.lang)
mm_real = [(p, m, t) for p, m, t in mm_all if m.source is Source.REAL]
mm_syn = [(p, m, t) for p, m, t in mm_all if m.source is Source.SYNTHETIC]

takes = []          # one entry per multimodal take that has images
no_media = 0
for p, m, t in mm_real:
    side = MEDIA / A.lang / m.label / f"{m.signer_id}__real__{m.variant:04d}.json"
    if not side.exists():
        no_media += 1
        continue
    clip = np.load(p).astype(np.float32)
    meta = json.loads(side.read_text(encoding="utf-8"))
    emb = embed_take(side, p.with_name(p.name.replace("__clean__", "__noisy__")))
    pres = hand_presence(clip).mean(0)
    takes.append(dict(take=t, label=m.label, signer=m.signer_id,
                      lm=deep_x(clip), img=align(emb, meta.get("lm")),
                      left=float(pres[0]), right=float(pres[1])))
if not takes:
    raise SystemExit(f"no takes with media in {A.lang}")
if no_media:
    print(f"  {no_media} landmark takes in {A.lang} have no video/frames "
          f"and are left out")

LABELS = sorted({d["label"] for d in takes})
L2I = {l: i for i, l in enumerate(LABELS)}
SIGNERS = sorted({d["signer"] for d in takes})
N = len(takes)
y_mm = np.array([L2I[d["label"]] for d in takes])
sg_mm = np.array([d["signer"] for d in takes])
LM = np.stack([d["lm"] for d in takes])
IMG = np.stack([d["img"] for d in takes])
TAKE_IDS = [d["take"] for d in takes]
print(f"  {N} takes with images · {len(LABELS)} signs · signers {SIGNERS} "
      f"· image features in {time.time() - t0:.0f}s")

# synthetic copies of the multimodal takes, by parent take
syn_by_take = defaultdict(list)
if A.synthetic:
    for p, m, t in mm_syn:
        if m.label in L2I:
            syn_by_take[t].append(deep_x(np.load(p).astype(np.float32)))

# landmark-only extra corpus
extra = []          # (take_id, signer, label_idx, features)
if A.extra:
    ex_all = load_clean(A.extra)
    ex_real = [(p, m, t) for p, m, t in ex_all
               if m.source is Source.REAL and m.label in L2I]
    if A.extra_cap:
        per = defaultdict(list)
        for row in ex_real:
            per[(row[1].signer_id, row[1].label)].append(row)
        ex_real = [r for rows in per.values()
                   for r in sorted(rows, key=lambda r: r[2][2])[:A.extra_cap]]
    kept = {t for _p, _m, t in ex_real}
    for p, m, t in ex_real:
        extra.append((t, m.signer_id, L2I[m.label],
                      deep_x(np.load(p).astype(np.float32))))
    if A.synthetic:
        for p, m, t in ex_all:
            if m.source is Source.SYNTHETIC and t in kept:
                extra.append((t, m.signer_id, L2I[m.label],
                              deep_x(np.load(p).astype(np.float32))))
    ex_signers = sorted({e[1] for e in extra})
    print(f"  extra {A.extra}: {len(extra)} landmark clips · signers "
          f"{ex_signers}")
    missing = [s for s in SIGNERS if s not in ex_signers]
    if missing:
        print(f"  note: {missing} not found in {A.extra} under that exact "
              f"tag. If they ARE in it under another spelling, their old "
              f"takes will stay in training when they are held out.")

# ── splits ───────────────────────────────────────────────────────────
proto = A.protocol
if proto == "auto":
    proto = "loso" if len(SIGNERS) >= 2 else "cv"
if proto == "loso":
    if len(SIGNERS) < 2:
        raise SystemExit("leave-one-signer-out needs 2+ people in "
                         f"{A.lang}; only {SIGNERS}")
    FOLDS = [(s, np.where(sg_mm != s)[0], np.where(sg_mm == s)[0])
             for s in SIGNERS]
else:
    sgkf = StratifiedGroupKFold(n_splits=A.folds, shuffle=True,
                                random_state=0)
    FOLDS = [(f"fold{k}", tr, te) for k, (tr, te) in enumerate(
        sgkf.split(np.zeros(N), y_mm, groups=np.arange(N)))]
print(f"protocol: {proto} · {len(FOLDS)} folds · arms {ARMS} · "
      f"{A.epochs} epochs · {DEV}")
if proto == "cv":
    print("  same-signer: every test person is also in training. "
          "A pipeline check, not evidence for unseen signers.")


# ── training ─────────────────────────────────────────────────────────
def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    torch.manual_seed(A.seed)
    net = SignTCN(len(LABELS), in_features=Xtr.shape[2]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.epochs)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(A.seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                  torch.from_numpy(ytr)),
                    batch_size=32, shuffle=True, generator=g)
    net.train()
    for _ in range(A.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(Xte).to(DEV)).argmax(1).cpu().numpy()


def score(y, p) -> dict:
    return dict(f1=float(f1_score(y, p, average="macro", zero_division=0) * 100),
                acc=float(accuracy_score(y, p) * 100))


preds = {a: np.full(N, -1) for a in ARMS}
folds_out = []
for name, tr, te in FOLDS:
    tf = time.time()
    held = set(sg_mm[te]) if proto == "loso" else set()

    # landmark-only training clips: multimodal synthetic + extra corpus
    lm_only, lm_only_y = [], []
    for i in tr:
        for f in syn_by_take.get(TAKE_IDS[i], []):
            lm_only.append(f)
            lm_only_y.append(y_mm[i])
    # Extra takes are other recordings from another folder, so a matching
    # take number is a coincidence; only the held-out PERSON is excluded.
    for _t, s, yi, f in extra:
        if s in held:
            continue
        lm_only.append(f)
        lm_only_y.append(yi)
    lm_only = (np.stack(lm_only) if lm_only
               else np.zeros((0, SEQ_LEN, LM.shape[2]), np.float32))
    lm_only_y = np.array(lm_only_y, dtype=np.int64)

    # image features standardised on this fold's training takes only
    mu = IMG[tr].reshape(-1, IMG_DIM).mean(0)
    sd = IMG[tr].reshape(-1, IMG_DIM).std(0) + 1e-6
    img = ((IMG - mu) / sd).astype(np.float32)

    ytr = y_mm[tr].astype(np.int64)
    row = dict(fold=name, n_train_mm=int(len(tr)), n_test=int(len(te)),
               n_train_landmark_only=int(len(lm_only)))
    for arm in ARMS:
        if arm == "A":
            Xtr = np.concatenate([LM[tr], lm_only])
            Ytr = np.concatenate([ytr, lm_only_y])
            Xte = LM[te]
        elif arm == "B":
            Xtr, Ytr, Xte = img[tr], ytr, img[te]
        elif arm == "C":
            one = np.ones((N, SEQ_LEN, 1), np.float32)
            fused = np.concatenate([LM, img, one], axis=2)
            pad = np.zeros((len(lm_only), SEQ_LEN, IMG_DIM + 1), np.float32)
            Xtr = np.concatenate([fused[tr],
                                  np.concatenate([lm_only, pad], axis=2)])
            Ytr = np.concatenate([ytr, lm_only_y])
            Xte = fused[te]
        else:
            raise SystemExit(f"unknown arm {arm}")
        p = fit_predict(Xtr.astype(np.float32), Ytr, Xte.astype(np.float32))
        preds[arm][te] = p
        row[arm] = score(y_mm[te], p)
    if "A" in ARMS and "C" in ARMS:
        row["boost_C_minus_A"] = row["C"]["f1"] - row["A"]["f1"]
    folds_out.append(row)
    print(f"  {name:<14} " + "  ".join(
        f"{a} {row[a]['f1']:5.1f}" for a in ARMS)
        + (f"   C-A {row['boost_C_minus_A']:+5.1f}"
           if "boost_C_minus_A" in row else "")
        + f"   ({time.time() - tf:.0f}s)")


# ── summary ──────────────────────────────────────────────────────────
def mean_sd(vals):
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))


summary = {a: {"f1": mean_sd([r[a]["f1"] for r in folds_out]),
               "acc": mean_sd([r[a]["acc"] for r in folds_out])}
           for a in ARMS}
if "A" in ARMS and "C" in ARMS:
    b = [r["boost_C_minus_A"] for r in folds_out]
    summary["boost_C_minus_A"] = dict(
        **mean_sd(b), folds_up=int(sum(v > 0 for v in b)),
        folds_down=int(sum(v < 0 for v in b)))

# Where the tracker lost a hand in more than half the frames. Signs that use
# one hand are naturally here too, so read it per sign before concluding.
poor = np.array([min(d["left"], d["right"]) < 0.5 for d in takes])
subset = {}
for nm, mask in (("hand_lost_over_half", poor), ("hands_mostly_found", ~poor)):
    if mask.any():
        subset[nm] = dict(n=int(mask.sum()), **{
            a: score(y_mm[mask], preds[a][mask]) for a in ARMS})

try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=ROOT, capture_output=True, text=True
                            ).stdout.strip()
except OSError:
    commit = None

texts = json.loads((SEQ / A.lang / "labels.json").read_text(encoding="utf-8"))
R = dict(
    generated=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit,
    config=vars(A), protocol=proto, device=DEV,
    arms={"A": "landmarks (290/frame)", "B": f"ResNet-18 images, {A.crop}",
          "C": "landmarks + images + has-image flag"},
    labels=LABELS, label_text=[texts.get(l, l) for l in LABELS],
    signers=SIGNERS, n_takes=N,
    n_extra_clips=len(extra),
    folds=folds_out, summary=summary, hand_subset=subset,
    predictions=[dict(signer=d["signer"], label=d["label"],
                      variant=int(d["take"][2]),
                      **{a: LABELS[int(preds[a][i])] for a in ARMS})
                 for i, d in enumerate(takes)],
)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False),
                                  encoding="utf-8")

print(f"\nmacro-F1 over {len(FOLDS)} folds ({proto}):")
for a in ARMS:
    s = summary[a]
    print(f"  {a}  {R['arms'][a]:<38} {s['f1']['mean']:5.1f} ± "
          f"{s['f1']['sd']:4.1f}   (accuracy {s['acc']['mean']:5.1f})")
if "boost_C_minus_A" in summary:
    b = summary["boost_C_minus_A"]
    print(f"  C - A  {b['mean']:+5.1f} ± {b['sd']:4.1f}   "
          f"(up in {b['folds_up']}, down in {b['folds_down']} of "
          f"{len(FOLDS)} folds)")
for nm, d in subset.items():
    print(f"  {nm:<20} n={d['n']:<4} " + "  ".join(
        f"{a} acc {d[a]['acc']:5.1f}" for a in ARMS))
if proto == "cv":
    print("\nSame-signer protocol: do not quote this as a result about "
          "new signers.")
print(f"\nwrote {OUT / 'results.json'}  ({time.time() - t0:.0f}s)")
