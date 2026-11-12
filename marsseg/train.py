"""Training loop: AdamW, linear warm-up + cosine decay, bf16 autocast, class-weighted cross-entropy
(weights 1 - class share, as in Swan et al. 2021, Sec. 4), best checkpoint chosen on validation mIoU."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from .metrics import confusion, scores
from .model import build


def nonempty(samples, cache: Path | None = None):
    """Drop frames whose merged label has no labelled pixel at all (they carry no training signal)."""
    known = json.loads(cache.read_text()) if cache and cache.exists() else {}
    keep = []
    for s in samples:
        if s.name not in known:
            known[s.name] = bool((D.load_label(s) != D.IGNORE).any())
        if known[s.name]:
            keep.append(s)
    if cache:
        cache.write_text(json.dumps(known))
    return keep


def splits(root: Path, seed: int = 0, max_val: int = 250):
    """Block-wise 95/5 split of the crowd-labelled frames (done before dropping empty labels, so it does not
    depend on the filter), then the validation part is cut in two disjoint halves: `select` (<= max_val frames,
    used to pick the best epoch) and `calib` (the rest, used only to fit the temperature)."""
    train_s, val_s = D.split_train_val(D.msl_samples(root, "train"), 0.05, seed)
    cache = Path(root) / "nonempty_train.json"
    train_s, val_s = nonempty(train_s, cache), nonempty(val_s, cache)
    step = max(1, len(val_s) // max_val)
    select = val_s[::step][:max_val]
    chosen = set(select)
    calib = [v for v in val_s if v not in chosen]
    return train_s, select, calib


def class_shares(samples, n: int = 1500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros(4)
    for i in rng.choice(len(samples), size=min(n, len(samples)), replace=False):
        lab = D.load_label(samples[i])
        counts += np.bincount(lab[lab != D.IGNORE].ravel(), minlength=4)[:4]
    return counts / counts.sum()


@torch.no_grad()
def validate(model, loader, device, size_out=None) -> dict:
    model.eval()
    cm = np.zeros((4, 4), np.int64)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = torch.nn.functional.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear",
                                                 align_corners=False)
        cm += confusion(logits.argmax(1), y)
    return scores(cm) | {"cm": cm}


def run(root: Path, out: Path, arch: str = "segformer-b2", epochs: int = 12, batch: int = 8, size: int = 512,
        lr: float = 2e-4, workers: int = 6, seed: int = 0, max_train: int | None = None,
        budget_min: float | None = None, max_val: int = 250) -> Path:
    """budget_min: wall-clock budget for the optimisation steps; after 60 steps the step time is measured and
    the cosine schedule is shortened so that it ends inside the budget (validation time not included)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    train_s, val_sub, val_calib = splits(root, seed, max_val)
    if max_train:
        train_s = train_s[:max_train]
    shares = class_shares(train_s)
    weights = torch.tensor(1 - shares, dtype=torch.float32, device=device)
    print(f"train {len(train_s)} val-select {len(val_sub)} val-calib {len(val_calib)} "
          f"class shares {np.round(shares, 4)}")

    tl = DataLoader(D.TrainSet(train_s, size), batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(D.EvalSet(val_sub, size), batch_size=4, num_workers=workers)
    model = build(arch).to(device)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": rest, "lr": lr * 10}], weight_decay=0.01)
    plan = {"total": epochs * len(tl)}
    warm = min(500, plan["total"] // 10)

    def lr_factor(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, plan["total"] - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    step, t_steps = 0, None
    lossf = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=D.IGNORE)

    log, best = [], -1.0
    ck = out / f"{arch}.pt"
    for ep in range(epochs):
        if step >= plan["total"]:
            break
        model.train()
        t0, run_loss, n_it = time.time(), 0.0, 0
        for k, (x, y) in enumerate(tl):
            if step >= plan["total"]:
                break
            if step == 10:
                t_steps = time.time()
            if step == 70 and budget_min:
                per = (time.time() - t_steps) / 60
                plan["total"] = min(plan["total"], int(budget_min * 60 / per))
                print(f"  {per:.2f} s/step -> schedule set to {plan['total']} steps "
                      f"({plan['total'] / len(tl):.2f} epochs)", flush=True)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            n_it += 1
            run_loss += loss.item()
            if k % 200 == 0:
                print(f"  ep {ep} it {k}/{len(tl)} loss {loss.item():.3f} "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        v = validate(model, vl, device)
        rec = {"epoch": ep, "steps": step, "loss": run_loss / max(1, n_it), "val_miou": v["miou"], "val_acc": v["pixel_acc"],
               "val_iou": [float(a) for a in v["iou"]], "minutes": (time.time() - t0) / 60}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if v["miou"] > best:
            best = v["miou"]
            torch.save({"arch": arch, "state": model.state_dict(), "epoch": ep, "val": rec,
                        "class_shares": shares.tolist(), "p_drop": 0.1, "size": size}, ck)
        (out / f"{arch}_log.json").write_text(json.dumps(log, indent=1))
    return ck
