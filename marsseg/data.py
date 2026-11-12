"""AI4Mars (MSL NAVCAM) indexing and PyTorch datasets.

Layout after `marsseg download` (paths relative to the data root):
    msl/ncam/images/edr/<ID>EDR<...>.JPG          1024x1024 greyscale NAVCAM frames
    msl/ncam/images/mxy/<ID>MXY<...>.png          rover mask (1 = rover hardware)
    msl/ncam/images/rng-30m/<ID>RNG<...>.png      range mask (1 = farther than 30 m)
    msl/ncam/labels/train/<ID>EDR<...>.png        merged crowd labels
    msl/ncam/labels/test/masked-gold-minK-100agree/<ID>EDR<...>_merged.png   expert labels, K = 1, 2, 3
Label values: 0 soil, 1 bedrock, 2 sand, 3 big rock, 255 no label.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLASSES = ["soil", "bedrock", "sand", "big rock"]
IGNORE = 255
# Display colours (also used in the README figures).
COLORS = np.array([[205, 133, 63], [112, 128, 144], [238, 214, 125], [178, 34, 34]], dtype=np.uint8)


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    rover: Path | None
    range30: Path | None

    @property
    def name(self) -> str:
        return self.image.stem


def _aux(root: Path, kind: str, tag: str, image: Path) -> Path | None:
    p = root / "msl/ncam/images" / kind / (image.stem.replace("EDR", tag) + ".png")
    return p if p.exists() else None


def msl_samples(root: Path, split: str) -> list[Sample]:
    """split = 'train' or 'test-min1' / 'test-min2' / 'test-min3'."""
    root = Path(root)
    edr = root / "msl/ncam/images/edr"
    if split == "train":
        labels = sorted((root / "msl/ncam/labels/train").glob("*.png"))
    else:
        k = split.split("min")[-1]
        labels = sorted((root / f"msl/ncam/labels/test/masked-gold-min{k}-100agree").glob("*.png"))
    out = []
    for lab in labels:
        img = edr / (lab.stem.replace("_merged", "") + ".JPG")
        if img.exists():
            out.append(Sample(img, lab, _aux(root, "mxy", "MXY", img), _aux(root, "rng-30m", "RNG", img)))
    return out


def split_train_val(samples: list[Sample], val_frac: float = 0.05, seed: int = 0):
    """Split by sol-ordered blocks rather than single frames: consecutive NAVCAM frames overlap heavily,
    so a random frame split would leak near-duplicates into validation."""
    s = sorted(samples, key=lambda x: x.name)
    blocks = [s[i:i + 20] for i in range(0, len(s), 20)]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    n_val = max(1, int(round(len(blocks) * val_frac)))
    val = [x for b in blocks[:n_val] for x in b]
    train = [x for b in blocks[n_val:] for x in b]
    return train, val


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def load_label(sample: Sample) -> np.ndarray:
    lab = np.asarray(Image.open(sample.label), dtype=np.uint8).copy()
    return lab


def invalid_mask(sample: Sample, shape) -> np.ndarray:
    """True where the rover hardware or terrain beyond 30 m is (pixels nobody should be scored on)."""
    m = np.zeros(shape, bool)
    for p in (sample.rover, sample.range30):
        if p is not None:
            a = np.asarray(Image.open(p))
            if a.shape != shape:
                a = np.asarray(Image.open(p).resize(shape[::-1], Image.NEAREST))
            m |= a > 0
    return m


MEAN, STD = 0.45, 0.225  # rough greyscale ImageNet statistics


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.array(img, copy=True)).float().div_(255.0).sub_(MEAN).div_(STD)
    return x.unsqueeze(0).expand(3, -1, -1).contiguous()  # replicate grey to the 3 pretrained channels


class TrainSet(torch.utils.data.Dataset):
    """Random-resized square crops (scale 0.35-1 of the frame) resized to `size`, horizontal flips,
    mild brightness/contrast jitter. Rover and >30 m pixels are set to IGNORE."""

    def __init__(self, samples: list[Sample], size: int = 512, train: bool = True):
        self.samples, self.size, self.train = samples, size, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s.image)
        lab = load_label(s)
        if lab.shape != img.shape:
            lab = np.asarray(Image.fromarray(lab).resize(img.shape[::-1], Image.NEAREST)).copy()
        lab[invalid_mask(s, img.shape)] = IGNORE
        if self.train:
            h, w = img.shape
            side = int(min(h, w) * random.uniform(0.35, 1.0))
            y0, x0 = random.randint(0, h - side), random.randint(0, w - side)
            img, lab = img[y0:y0 + side, x0:x0 + side], lab[y0:y0 + side, x0:x0 + side]
            if random.random() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((self.size, self.size), Image.BILINEAR))
        lab = np.asarray(Image.fromarray(np.ascontiguousarray(lab)).resize((self.size, self.size), Image.NEAREST))
        return to_tensor(img), torch.from_numpy(lab.astype(np.int64))


class EvalSet(torch.utils.data.Dataset):
    """Whole frame resized to `size` for the network; label kept at native resolution."""

    def __init__(self, samples: list[Sample], size: int = 512):
        self.samples, self.size = samples, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_image(s.image)
        lab = load_label(s)
        if lab.shape != img.shape:
            lab = np.asarray(Image.fromarray(lab).resize(img.shape[::-1], Image.NEAREST)).copy()
        lab[invalid_mask(s, img.shape)] = IGNORE
        small = np.asarray(Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR))
        return to_tensor(small), torch.from_numpy(lab.astype(np.int64)), i


def m2020_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Perseverance NAVCAM images with merged crowd labels in the same four NAV classes (AI4Mars v0.5+).
    These labels rank below MSL NAVCAM in the dataset's own cleaning order (changelog, v0.5), so they are a
    noisier yardstick than the MSL expert test set."""
    root = Path(root)
    imgs = {p.stem: p for p in (root / "m2020/images/ncam").glob("*.jp*g")}
    out = []
    for lab in sorted((root / "m2020/labels/NAV").glob("*.png")):
        stem = lab.stem.rsplit("_merged", 1)[0]
        if stem in imgs:
            out.append((imgs[stem], lab))
    return out
