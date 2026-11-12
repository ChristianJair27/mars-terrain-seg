"""Evaluation on the expert test sets, temperature scaling, Monte-Carlo dropout and calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import data as D
from .metrics import Calibration, confusion, entropy, scores
from .model import set_mc


@torch.no_grad()
def features(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.decoder(model.encoder(x))


@torch.no_grad()
def predict(model, x, mc: int = 0, temperature: float = 1.0, out_size=None):
    """Returns (mean probability (B, C, H, W), mutual information (B, H, W) or None).

    mc = 0: deterministic. mc = T > 0: T stochastic passes of the dropout + classifier head on shared
    decoder features (the only stochastic layer), averaged in probability space. The mutual information
    H[mean p] - mean H[p] is the epistemic part of the uncertainty (BALD; Houlsby et al. 2011)."""
    feat = features(model, x).float()
    head = model.segmentation_head
    passes = max(mc, 1)
    set_mc(model, mc > 0)
    probs, ents = 0.0, 0.0
    for _ in range(passes):
        logits = head(feat) / temperature
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        p = logits.softmax(1)
        probs = probs + p / passes
        if mc:
            ents = ents + torch.stack([entropy(pi) for pi in p]) / passes
    set_mc(model, False)
    mi = (torch.stack([entropy(pi) for pi in probs]) - ents) if mc else None
    return probs, mi


@torch.no_grad()
def fit_temperature(model, samples, device="cuda", size=512, n_pix=400_000, seed=0) -> float:
    """Guo et al. (2017) temperature scaling: one scalar T minimising the NLL on held-out pixels."""
    loader = DataLoader(D.EvalSet(samples, size), batch_size=4, num_workers=4)
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model.segmentation_head(features(model, x).float())
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        z = logits.permute(0, 2, 3, 1).reshape(-1, 4)
        yy = y.reshape(-1)
        ok = (yy != D.IGNORE).nonzero().squeeze(1)
        pick = ok[torch.randperm(len(ok), generator=g)[: max(1, n_pix // len(loader))].to(ok.device)]
        zs.append(z[pick].cpu())
        ys.append(yy[pick].cpu())
    z, y = torch.cat(zs), torch.cat(ys)
    logt = torch.zeros(1, requires_grad=True)
    with torch.enable_grad():
        opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z / logt.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
    return float(logt.exp())


@torch.no_grad()
def evaluate(model, root, splits=("test-min1", "test-min2", "test-min3"), device="cuda", size=512,
             temperature=1.0, mc=0) -> dict:
    res = {}
    for split in splits:
        samples = D.msl_samples(root, split)
        loader = DataLoader(D.EvalSet(samples, size), batch_size=2, num_workers=4)
        cm = np.zeros((4, 4), np.int64)
        cal_raw, cal = Calibration(), Calibration()
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            p, _ = predict(model, x, mc=mc, temperature=temperature, out_size=y.shape[-2:])
            cm += confusion(p.argmax(1), y)
            for pi, yi in zip(p, y):
                cal.update(pi, yi)
            if temperature != 1.0 and not mc:
                p1, _ = predict(model, x, out_size=y.shape[-2:])
                for pi, yi in zip(p1, y):
                    cal_raw.update(pi, yi)
        s = scores(cm)
        r = cal.result()
        res[split] = {
            "n_images": len(samples), "n_pixels": s["n_pixels"], "pixel_acc": s["pixel_acc"], "miou": s["miou"],
            "iou": dict(zip(D.CLASSES, map(float, s["iou"]))),
            "confusion_row_pct": np.round(s["row_pct"], 2).tolist(), "confusion_counts": cm.tolist(),
            "ece": r["ece"], "reliability": {k: np.asarray(v).tolist() for k, v in r.items() if k != "ece"},
        }
        if temperature != 1.0 and not mc:
            rr = cal_raw.result()
            res[split]["ece_before_temperature"] = rr["ece"]
            res[split]["reliability_raw"] = {k: np.asarray(v).tolist() for k, v in rr.items() if k != "ece"}
        print(f"{split}: acc {s['pixel_acc']:.4f} mIoU {s['miou']:.4f} ECE {r['ece']:.4f} "
              f"IoU {np.round(s['iou'], 3)}", flush=True)
    return res


def save(res: dict, path: Path):
    Path(path).write_text(json.dumps(res, indent=1))


@torch.no_grad()
def evaluate_pairs(model, pairs, device="cuda", long_side=512, temperature=1.0, limit=None) -> dict:
    """Score (image, label) pairs of any size/aspect (used for the Perseverance crowd labels)."""
    from PIL import Image
    from .predict import prepare
    cm = np.zeros((4, 4), np.int64)
    cal = Calibration()
    for img_path, lab_path in pairs[:limit]:
        img, x = prepare(img_path, long_side)
        lab = np.asarray(Image.open(lab_path), dtype=np.uint8)
        if lab.shape != img.shape:
            lab = np.asarray(Image.open(lab_path).resize(img.shape[::-1], Image.NEAREST))
        y = torch.from_numpy(lab.astype(np.int64)).to(device)
        y[y > 3] = D.IGNORE
        p, _ = predict(model, x.to(device), temperature=temperature, out_size=img.shape)
        cm += confusion(p[0].argmax(0), y)
        cal.update(p[0], y)
    s = scores(cm)
    return {"n_images": len(pairs[:limit]), "n_pixels": s["n_pixels"], "pixel_acc": s["pixel_acc"],
            "miou": s["miou"], "iou": dict(zip(D.CLASSES, map(float, s["iou"]))),
            "confusion_row_pct": np.round(s["row_pct"], 2).tolist(), "ece": cal.result()["ece"]}


@torch.no_grad()
def fit_class_bias(model, samples, cls: int = 3, grid=None, device="cuda", size=512, temperature=1.0) -> dict:
    """Post-hoc prior shift for one rare class: subtract b from its logit and pick the b that maximises that
    class's IoU on held-out frames (never on the test set). Returns the sweep and the chosen b."""
    grid = np.round(np.arange(0.0, 6.01, 0.5), 2) if grid is None else np.asarray(grid)
    loader = DataLoader(D.EvalSet(samples, size), batch_size=4, num_workers=4)
    cms = np.zeros((len(grid), 4, 4), np.int64)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model.segmentation_head(features(model, x).float()) / temperature
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        for k, b in enumerate(grid):
            z = logits.clone()
            z[:, cls] -= float(b)
            cms[k] += confusion(z.argmax(1), y)
    sweep = []
    for b, cm in zip(grid, cms):
        s = scores(cm)
        sweep.append({"bias": float(b), "iou": float(s["iou"][cls]), "miou": float(s["miou"]),
                      "recall": float(s["recall"][cls]),
                      "precision": float(cm[cls, cls] / max(1, cm[:, cls].sum()))})
    best = max(sweep, key=lambda r: r["iou"])
    return {"class": D.CLASSES[cls], "sweep": sweep, "chosen_bias": best["bias"]}


@torch.no_grad()
def evaluate_with_bias(model, root, bias: float, cls: int = 3, splits=("test-min1", "test-min2", "test-min3"),
                       device="cuda", size=512, temperature=1.0) -> dict:
    res = {}
    for split in splits:
        loader = DataLoader(D.EvalSet(D.msl_samples(root, split), size), batch_size=2, num_workers=4)
        cm = np.zeros((4, 4), np.int64)
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            z = model.segmentation_head(features(model, x).float()) / temperature
            z = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)
            z[:, cls] -= bias
            cm += confusion(z.argmax(1), y)
        s = scores(cm)
        res[split] = {"pixel_acc": s["pixel_acc"], "miou": s["miou"], "iou": dict(zip(D.CLASSES, map(float, s["iou"]))),
                      "precision_cls": float(cm[cls, cls] / max(1, cm[:, cls].sum())),
                      "recall_cls": float(s["recall"][cls]),
                      "confusion_row_pct": np.round(s["row_pct"], 2).tolist(), "confusion_counts": cm.tolist()}
        print(f"{split} (bias {bias}): acc {s['pixel_acc']:.4f} mIoU {s['miou']:.4f} IoU {np.round(s['iou'], 3)}",
              flush=True)
    return res
