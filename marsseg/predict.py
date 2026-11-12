"""Segment arbitrary rover images: class map, uncertainty and drive-risk overlay per image."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import data as D
from . import hazard
from .evaluate import predict
from .metrics import entropy
from .model import load


def prepare(path: Path, long_side: int = 512):
    img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h, w = img.shape
    s = long_side / max(h, w)
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    small = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return img, D.to_tensor(small).unsqueeze(0)


@torch.no_grad()
def segment(model, path: Path, mc: int = 10, long_side: int = 512):
    """Returns the grey image, probabilities (4, H, W), normalised entropy and mutual information."""
    img, x = prepare(path, long_side)
    T = getattr(model, "temperature", 1.0)
    p, mi = predict(model, x.cuda(), mc=mc, temperature=T, out_size=img.shape)
    p = p[0]
    return img, p.cpu().numpy(), entropy(p).cpu().numpy(), (mi[0].cpu().numpy() if mi is not None else None)


def colorize(classes: np.ndarray) -> np.ndarray:
    rgb = np.zeros(classes.shape + (3,), np.uint8)
    ok = classes < 4
    rgb[ok] = D.COLORS[classes[ok]]
    return rgb


def run(checkpoint: Path, images: list[Path], out: Path, mc: int = 10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = load(checkpoint)
    for path in images:
        img, p, ent, mi = segment(model, path, mc)
        cls = p.argmax(0).astype(np.uint8)
        risk = hazard.risk_map(p, ent)
        grey = np.repeat(img[..., None], 3, -1).astype(np.float32)
        over = (0.55 * grey + 0.45 * colorize(cls)).astype(np.uint8)
        Image.fromarray(over).save(out / f"{path.stem}_classes.png")
        Image.fromarray((ent * 255).astype(np.uint8)).save(out / f"{path.stem}_entropy.png")
        Image.fromarray((np.nan_to_num(risk) * 255).astype(np.uint8)).save(out / f"{path.stem}_risk.png")
        print(path.name, "class shares", np.round(np.bincount(cls.ravel(), minlength=4) / cls.size, 3))
