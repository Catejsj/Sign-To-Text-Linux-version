"""Training loop. Callable from Colab or locally. No CLI flags yet — edit CONFIG."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    SignDataset, discover_samples, build_label_index,
    split_random, split_leave_one_signer_out, source_stats,
)
from .model_transformer import SignTransformer, num_params


@dataclass
class TrainConfig:
    data_root: str
    weights_out: str
    labels_out: str
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    seed: int = 42
    device: str = "cuda"
    held_out_signer: str | None = None   # enables leave-one-signer-out if set
    flip_labels: list[str] | None = None


def _eval(model, loader, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(cfg: TrainConfig) -> dict:
    data_root = Path(cfg.data_root)
    weights_out = Path(cfg.weights_out)
    labels_out = Path(cfg.labels_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(data_root)
    if not samples:
        raise RuntimeError(f"no samples found under {data_root}")

    label_to_idx = build_label_index(samples)
    labels_out.write_text(json.dumps(label_to_idx, indent=2, ensure_ascii=False))
    print(f"labels: {list(label_to_idx)}")
    print(f"stats: {source_stats(samples)}")

    if cfg.held_out_signer:
        train_s, val_s = split_leave_one_signer_out(samples, cfg.held_out_signer)
        print(f"held out signer: {cfg.held_out_signer}")
    else:
        train_s, val_s = split_random(samples, cfg.val_frac, cfg.seed)
    print(f"train={len(train_s)}  val={len(val_s)}")

    flip_set = set(cfg.flip_labels or [])
    train_ds = SignDataset(train_s, label_to_idx, augment=True,
                           flip_labels=flip_set, seed=cfg.seed)
    val_ds = SignDataset(val_s, label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)

    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model = SignTransformer(num_classes=len(label_to_idx)).to(device)
    print(f"model params: {num_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = nn.CrossEntropyLoss()

    history = []
    best_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.numel()
            tr_correct += (logits.argmax(-1) == y).sum().item()
            tr_total += y.numel()
        sched.step()

        val_loss, val_acc = _eval(model, val_loader, device)
        tr_acc = tr_correct / max(tr_total, 1)
        tr_loss /= max(tr_total, 1)
        dt = time.time() - t0
        print(f"ep {epoch:3d}  tr_loss {tr_loss:.4f}  tr_acc {tr_acc:.3f}"
              f"  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  ({dt:.1f}s)")
        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "label_to_idx": label_to_idx,
                "config": asdict(cfg),
                "val_acc": val_acc,
                "epoch": epoch,
            }, weights_out)
            print(f"  → saved best ({val_acc:.3f}) to {weights_out}")

    return {"best_val_acc": best_acc, "history": history,
            "label_to_idx": label_to_idx}
