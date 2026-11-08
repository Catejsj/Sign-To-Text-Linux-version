"""Sentence recognition: an RNN reads every frame, CTC turns it into signs.

The sign models (run_task_a) give one answer per clip. A sentence is several
signs with no marks for where one ends and the next begins, so here a model
outputs, for every frame, either a sign or "blank", and CTC training learns the
alignment by itself. Decoding removes repeats and blanks: "- A A - B B - C" is
A B C. Signs it knows can then be read in an order it never saw.

Scored the way the project scores everything, on people the model never saw:

  unseen signer    leave one person out (needs 2+ people; with one person it
                   falls back to grouped 5-fold and says the result is same-person)
  unseen sentence  hold out one sentence at a time, train on the rest; only
                   sentences whose every sign appears in another one are tested

  WER              word error rate: wrong + missing + extra signs, divided by
                   the number of signs said. Lower is better; 0 is perfect
  sentence %       takes where the whole sign list came out exactly right

Input per frame is the same 290 numbers run_task_a's deep models use:
positions with missing hands zeroed, a 2-value hand-presence flag, and unit
bone directions. Sentence takes are read from the full-length .full.npz files
(every frame), never the 60-frame .npy clips.

Each sentence's signs, in the order they are signed, come from signs.json in
the language folder, e.g. {"sl_004": ["WEATHER", "TODAY", "HOW"]}. Sign names
are free text, but the same sign must have the same name in every sentence.

--stitch FOLDER adds made-up sentences built by joining single-sign takes from
another folder, to training only and only from people in the training set.
A sign name in signs.json must equal that folder's labels.json text to be used.

--demo FOLDER needs no sentence recordings: every sentence, train and test, is
stitched from single-sign takes. It checks the whole pipeline works and shows
whether CTC can read sign orders it never trained on. Stitched sentences are
easier than real ones (clean joins, no co-articulation), so demo scores are not
sentence results.

Examples:
    python algo_comparison/run_sentences.py --demo khmer --models bigru --epochs 15
    python algo_comparison/run_sentences.py --lang khmer_sent
    python algo_comparison/run_sentences.py --lang khmer_sent --stitch khmer
    python algo_comparison/run_sentences.py --lang khmer_sent --save bigru
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
from torch.nn.utils.rnn import (pack_padded_sequence,               # noqa: E402
                                pad_packed_sequence)

from src.v2.bones import bone_vectors                               # noqa: E402
from src.v2.landmarks import hand_presence, zero_missing            # noqa: E402
from src.v2.normalize import resample_time                          # noqa: E402
from src.v2.schema import load_full                                 # noqa: E402

MODELS = ["gru", "bigru", "lstm", "bilstm", "tcn", "transformer"]
# can run while the person is still signing (no look at future frames)
LIVE_CAPABLE = {"gru", "lstm"}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--lang", help="sentence folder, e.g. khmer_sent")
src.add_argument("--demo", metavar="FOLDER",
                 help="single-sign folder to stitch every sentence from")
ap.add_argument("--models", default=",".join(MODELS),
                help=f"comma list from {MODELS}")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--stitch", metavar="FOLDER", default=None,
                help="single-sign folder for extra stitched training sentences")
ap.add_argument("--stitch-n", type=int, default=400,
                help="stitched training sentences per fold")
ap.add_argument("--save", metavar="MODEL", default=None,
                help="also train MODEL on everything and save it to models/sentence/")
ap.add_argument("--tag", default=None, help="suffix for the results folder")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

DATA = ROOT / "data" / "sequences_v2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KINDS = [m.strip() for m in A.models.split(",") if m.strip()]
bad = [m for m in KINDS if m not in MODELS]
if bad:
    sys.exit(f"unknown model(s) {bad}; choose from {MODELS}")
if A.save and A.save not in MODELS:
    sys.exit(f"--save must be one of {MODELS}")
rng = np.random.default_rng(A.seed)
torch.manual_seed(A.seed)


# ── features ─────────────────────────────────────────────────────────
def frame_features(clip: np.ndarray) -> np.ndarray:
    """(T, 48, 3) clean view -> (T, 290), as run_task_a's deep_x."""
    p = hand_presence(clip)
    return np.concatenate([zero_missing(clip, p).reshape(len(clip), -1), p,
                           bone_vectors(clip, unit=True).reshape(len(clip), -1)],
                          axis=1).astype(np.float32)


# ── data ─────────────────────────────────────────────────────────────
def load_single_signs(folder: str) -> list[dict]:
    """Real single-sign takes (60-frame clean view) with their sign name."""
    d = DATA / folder
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    out = []
    for slug, text in names.items():
        for p in sorted((d / slug).glob("*__real__clean__*.npy")):
            out.append(dict(clip=np.load(p).astype(np.float32), sign=text,
                            signer=p.name.split("__")[0]))
    if not out:
        sys.exit(f"no real takes in {d}")
    return out


def load_sentences(folder: str) -> list[dict]:
    d = DATA / folder
    sp = d / "signs.json"
    if not sp.exists():
        sys.exit(f"{sp} is missing. Write each sentence's signs in signing order, "
                 f'e.g. {{"sl_004": ["WEATHER", "TODAY", "HOW"]}}.')
    signs = json.loads(sp.read_text(encoding="utf-8"))
    empty = [k for k, v in signs.items() if not v]
    if empty:
        sys.exit(f"signs.json has no signs yet for {empty}. Fill them in from the "
                 f"videos (signing order) before training.")
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    out, missing = [], []
    for slug in names:
        files = sorted((d / slug).glob("*.full.npz"))
        if not files:
            continue
        if slug not in signs:
            missing.append(slug)
            continue
        for p in files:
            clean, _, fps = load_full(p)
            stem = p.name.replace(".full.npz", "")
            signer, _, var = stem.split("__")
            out.append(dict(clip=clean, signs=list(signs[slug]), sentence=slug,
                            signer=signer, take=f"{signer}/{slug}/{var}"))
    if missing:
        sys.exit(f"signs.json has no entry for {missing}")
    if not out:
        sys.exit(f"no .full.npz takes in {d}. Takes recorded before 8 Nov 2026 "
                 f"have none; record them again.")
    return out


def stitch(pool: list[dict], seqs: list[list[str]], n: int) -> list[dict]:
    """n sentences, each a random choice from `seqs`, built from takes in `pool`
    (one random take per sign, one person per sentence). Each sign is resampled
    to 24-44 frames, about the length of a sign inside a real sentence."""
    by = defaultdict(list)
    for s in pool:
        by[(s["signer"], s["sign"])].append(s["clip"])
    people = sorted({s["signer"] for s in pool})
    out = []
    for _ in range(n):
        seq = seqs[rng.integers(len(seqs))]
        ok = [p for p in people if all((p, w) in by for w in seq)]
        if not ok:
            continue
        who = ok[rng.integers(len(ok))]
        parts = []
        for w in seq:
            takes = by[(who, w)]
            c = takes[rng.integers(len(takes))]
            parts.append(resample_time(c, int(rng.integers(24, 45))))
        out.append(dict(clip=np.concatenate(parts), signs=list(seq),
                        signer=who, sentence=" ".join(seq), take="stitched"))
    return out


# ── models ───────────────────────────────────────────────────────────
class SentenceNet(nn.Module):
    def __init__(self, kind: str, nf: int, nc: int, h: int = 128):
        super().__init__()
        self.kind = kind
        if kind in ("gru", "bigru", "lstm", "bilstm"):
            cls = nn.GRU if "gru" in kind else nn.LSTM
            bi = kind.startswith("bi")
            self.rnn = cls(nf, h, num_layers=2, batch_first=True,
                           bidirectional=bi, dropout=0.2)
            out = h * (2 if bi else 1)
        elif kind == "tcn":
            self.inp = nn.Conv1d(nf, h, 1)
            self.convs = nn.ModuleList(
                nn.Conv1d(h, h, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8))
            out = h
        else:                                        # transformer
            self.inp = nn.Linear(nf, h)
            layer = nn.TransformerEncoderLayer(h, 4, 4 * h, 0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            out = h
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(out, nc + 1)           # +1: CTC blank, index 0

    def forward(self, x, lengths):
        if hasattr(self, "rnn"):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            y, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True,
                                       total_length=x.shape[1])
        elif self.kind == "tcn":
            y = self.inp(x.transpose(1, 2))
            for c in self.convs:
                y = y + torch.relu(c(self.drop(y)))
            y = y.transpose(1, 2)
        else:
            T = x.shape[1]
            pos = torch.arange(T, device=x.device)[:, None]
            div = torch.exp(torch.arange(0, self.inp.out_features, 2, device=x.device)
                            * (-np.log(10000.0) / self.inp.out_features))
            pe = torch.zeros(T, self.inp.out_features, device=x.device)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            pad = torch.arange(T, device=x.device)[None] >= lengths[:, None]
            y = self.enc(self.inp(x) + pe, src_key_padding_mask=pad)
        return self.head(self.drop(y)).log_softmax(-1)


def batches(items, size, shuffle):
    idx = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def to_tensors(group, vocab, augment):
    xs, ys = [], []
    for it in group:
        f = it["feat"]
        if augment:                                 # signing speed 0.8x to 1.25x
            f = resample_time(f[:, :, None], max(8, int(len(f) * rng.uniform(.8, 1.25))))[:, :, 0]
        xs.append(torch.from_numpy(np.ascontiguousarray(f)))
        ys.append([vocab[w] for w in it["signs"]])
    lengths = torch.tensor([len(x) for x in xs])
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    targets = torch.tensor([v for y in ys for v in y])
    tlen = torch.tensor([len(y) for y in ys])
    return x.to(DEV), lengths.to(DEV), targets, tlen


def train(kind, train_items, vocab, nf, mean, std):
    net = SentenceNet(kind, nf, len(vocab)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.epochs)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    m, s = torch.tensor(mean, device=DEV), torch.tensor(std, device=DEV)
    for _ in range(A.epochs):
        net.train()
        for g in batches(train_items, 16, True):
            x, L, y, tl = to_tensors(g, vocab, augment=True)
            lp = net((x - m) / s, L)
            loss = ctc(lp.transpose(0, 1), y, L, tl)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    net.m, net.s = m, s
    return net


@torch.no_grad()
def decode(net, items, vocab):
    inv = {i: w for w, i in vocab.items()}
    net.eval()
    out = []
    for g in batches(items, 32, False):
        x, L, _, _ = to_tensors(g, vocab, augment=False)
        best = net((x - net.m) / net.s, L).argmax(-1).cpu().numpy()
        for row, n in zip(best, L.cpu().numpy()):
            seq, prev = [], 0
            for k in row[:n]:
                if k != prev and k != 0:
                    seq.append(inv[int(k)])
                prev = k
            out.append(seq)
    return out


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


def score(refs, hyps):
    e = sum(edits(r, h) for r, h in zip(refs, hyps))
    n = sum(len(r) for r in refs)
    return dict(wer=100.0 * e / max(n, 1),
                sentence=100.0 * np.mean([r == h for r, h in zip(refs, hyps)]),
                n=len(refs))


def norm_stats(items):
    f = np.concatenate([it["feat"] for it in items])
    return f.mean(0), f.std(0) + 1e-6


def run_fold(kind, tr, te, vocab, nf):
    mean, std = norm_stats(tr)
    net = train(kind, tr, vocab, nf, mean, std)
    hyp = decode(net, te, vocab)
    return [it["signs"] for it in te], hyp


# ── build the corpus ─────────────────────────────────────────────────
t0 = time.time()
single = None
if A.demo:
    single = load_single_signs(A.demo)
    signs = sorted({s["sign"] for s in single})
    people = sorted({s["signer"] for s in single})
    # every 3-sign order with no sign twice; ~1/5 are kept out of training
    # entirely, to test orders the model never saw
    orders = [[a, b, c] for a in signs for b in signs for c in signs
              if len({a, b, c}) == 3]
    rng.shuffle(orders)
    k = max(1, len(orders) // 5)
    unseen_orders, seen_orders = orders[:k], orders[k:]
    corpus_name = f"demo_{A.demo}"
    print(f"demo: {len(single)} single-sign takes, {len(signs)} signs, "
          f"{len(people)} people; {len(seen_orders)} sign orders for training, "
          f"{len(unseen_orders)} kept unseen")
else:
    takes = load_sentences(A.lang)
    for t in takes:
        t["feat"] = frame_features(t["clip"])
    people = sorted({t["signer"] for t in takes})
    signs = sorted({w for t in takes for w in t["signs"]})
    corpus_name = A.lang
    if A.stitch:
        single = load_single_signs(A.stitch)
        usable = {s["sign"] for s in single}
        kept = [s for s in single if s["sign"] in signs]
        print(f"stitch: {len(kept)} single-sign takes from {A.stitch} match a sign "
              f"in signs.json ({sorted(usable & set(signs)) or 'none'})")
        single = kept or None
    n_sent = len({t["sentence"] for t in takes})
    print(f"{A.lang}: {len(takes)} sentence takes, {n_sent} sentences, "
          f"{len(signs)} signs, {len(people)} people · {DEV}")

vocab = {w: i + 1 for i, w in enumerate(signs)}      # 0 is the CTC blank
NF = 290
OUT = ROOT / "algo_comparison" / (f"results_{corpus_name}_sentences"
                                  + (f"_{A.tag}" if A.tag else ""))
OUT.mkdir(parents=True, exist_ok=True)


def featurize(items):
    for it in items:
        it["feat"] = frame_features(it["clip"])
    return items


def stitched_for(train_people, seqs, n):
    pool = [s for s in single if s["signer"] in train_people]
    return featurize(stitch(pool, seqs, n))


R: dict = dict(corpus=corpus_name, people=people, signs=signs, epochs=A.epochs,
         models={}, protocol={})

# ── unseen signer ────────────────────────────────────────────────────
if A.demo:
    folds = [(set(people) - {p}, {p}) for p in people]
    R["protocol"]["unseen_signer"] = "leave one person out, stitched sentences"
elif len(people) >= 2:
    folds = [(set(people) - {p}, {p}) for p in people]
    R["protocol"]["unseen_signer"] = "leave one person out"
else:
    # one person: grouped 5-fold over takes; this is NOT an unseen-signer score
    order = rng.permutation(len(takes))
    folds = [("idx", set(order[i::5].tolist())) for i in range(5)]
    R["protocol"]["unseen_signer"] = ("only one person: grouped 5-fold, "
                                      "same person (not an unseen-signer score)")
print("unseen signer:", R["protocol"]["unseen_signer"])

for kind in KINDS:
    refs, hyps = [], []
    for tr_people, te in folds:
        if A.demo:
            tr = stitched_for(tr_people, seen_orders, 600)
            te_items = featurize(stitch([s for s in single if s["signer"] in te],
                                        seen_orders + unseen_orders, 120))
        elif tr_people == "idx":
            tr = [t for i, t in enumerate(takes) if i not in te]
            te_items = [t for i, t in enumerate(takes) if i in te]
        else:
            tr = [t for t in takes if t["signer"] in tr_people]
            te_items = [t for t in takes if t["signer"] in te]
        if single is not None and not A.demo:
            seqs = [t["signs"] for t in tr]
            tp = {t["signer"] for t in tr}
            tr = tr + stitched_for(tp, seqs, A.stitch_n)
        r, h = run_fold(kind, tr, te_items, vocab, NF)
        refs += r
        hyps += h
    sc = score(refs, hyps)
    R["models"].setdefault(kind, {})["unseen_signer"] = sc
    if A.demo:   # split the demo score into orders seen / never seen in training
        unseen_set = {tuple(o) for o in unseen_orders}
        for name, keep in (("seen_orders", False), ("unseen_orders", True)):
            sel = [(r, h) for r, h in zip(refs, hyps) if (tuple(r) in unseen_set) == keep]
            R["models"][kind][name] = score(*zip(*sel)) if sel else None
    print(f"  {kind:<12} WER {sc['wer']:5.1f}   sentence {sc['sentence']:5.1f}%   "
          f"({sc['n']} test sentences, {time.time() - t0:.0f}s)")
    if A.demo:
        s1, s2 = R["models"][kind]["seen_orders"], R["models"][kind]["unseen_orders"]
        print(f"  {'':<12} orders seen in training WER {s1['wer']:5.1f} · "
              f"never seen WER {s2['wer']:5.1f}")

# ── unseen sentence ──────────────────────────────────────────────────
if not A.demo:
    sents = sorted({t["sentence"] for t in takes})
    sign_of = {t["sentence"]: t["signs"] for t in takes}
    testable = [s for s in sents
                if all(any(w in sign_of[o] for o in sents if o != s) for w in sign_of[s])]
    R["protocol"]["unseen_sentence"] = dict(
        testable=testable,
        skipped={s: [w for w in sign_of[s]
                     if not any(w in sign_of[o] for o in sents if o != s)]
                 for s in sents if s not in testable})
    if not testable:
        print("unseen sentence: no sentence has all its signs in another one; skipped")
    else:
        print(f"unseen sentence: {len(testable)} of {len(sents)} sentences testable "
              f"({', '.join(testable)})")
        for kind in KINDS:
            refs, hyps = [], []
            for s in testable:
                tr = [t for t in takes if t["sentence"] != s]
                te_items = [t for t in takes if t["sentence"] == s]
                r, h = run_fold(kind, tr, te_items, vocab, NF)
                refs += r
                hyps += h
            sc = score(refs, hyps)
            R["models"][kind]["unseen_sentence"] = sc
            print(f"  {kind:<12} WER {sc['wer']:5.1f}   sentence {sc['sentence']:5.1f}%")

(OUT / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
print(f"\nwrote {OUT / 'results.json'}")

# ── save a model for later use ───────────────────────────────────────
if A.save:
    if A.demo:
        sys.exit("--save is for real sentence folders, not --demo")
    items = list(takes)
    if single is not None:
        items += stitched_for(set(people), [t["signs"] for t in takes], A.stitch_n)
    mean, std = norm_stats(items)
    net = train(A.save, items, vocab, NF, mean, std)
    d = ROOT / "models" / "sentence"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{A.lang}_{A.save}.pt"
    torch.save(dict(kind=A.save, state=net.state_dict(), vocab=vocab, nf=NF,
                    mean=mean, std=std, live=A.save in LIVE_CAPABLE,
                    sentences=json.loads((DATA / A.lang / "labels.json")
                                         .read_text(encoding="utf-8")),
                    signs=json.loads((DATA / A.lang / "signs.json")
                                     .read_text(encoding="utf-8")),
                    trained_on=len(items), epochs=A.epochs), path)
    print(f"saved {path}")
