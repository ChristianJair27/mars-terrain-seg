"""Segmentation and calibration metrics (numpy/torch, no external dependencies)."""
from __future__ import annotations

import numpy as np
import torch

from .data import IGNORE


def confusion(pred: torch.Tensor, target: torch.Tensor, n: int = 4) -> np.ndarray:
    """Rows = true class, columns = predicted class, counted over labelled pixels only."""
    ok = target != IGNORE
    idx = target[ok].long() * n + pred[ok].long()
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy().astype(np.int64)


def scores(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (cm.sum(0) + cm.sum(1) - tp)
        recall = tp / cm.sum(1)
    present = cm.sum(1) > 0
    return {
        "pixel_acc": float(tp.sum() / cm.sum()) if cm.sum() else float("nan"),
        "iou": iou,
        "miou": float(np.nanmean(iou[present])),
        "recall": recall,
        "row_pct": 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1),
        "n_pixels": int(cm.sum()),
    }


class Calibration:
    """Accumulates a reliability diagram: confidence of the predicted class vs. its accuracy, in equal-width
    bins. ECE = sum_b (n_b / N) |acc_b - conf_b| (Guo et al. 2017)."""

    def __init__(self, bins: int = 15):
        self.bins = bins
        self.n = np.zeros(bins)
        self.conf = np.zeros(bins)
        self.acc = np.zeros(bins)

    def update(self, prob: torch.Tensor, target: torch.Tensor):
        """prob: (C, H, W) probabilities; target: (H, W)."""
        ok = target != IGNORE
        conf, pred = prob.max(0)
        conf, correct = conf[ok].float(), (pred[ok] == target[ok]).float()
        b = torch.clamp((conf * self.bins).long(), max=self.bins - 1)
        self.n += torch.bincount(b, minlength=self.bins).cpu().numpy()
        self.conf += torch.bincount(b, weights=conf, minlength=self.bins).cpu().numpy()
        self.acc += torch.bincount(b, weights=correct, minlength=self.bins).cpu().numpy()

    def result(self) -> dict:
        n = np.maximum(self.n, 1)
        acc, conf = self.acc / n, self.conf / n
        ece = float(np.sum(self.n / self.n.sum() * np.abs(acc - conf)))
        return {"ece": ece, "bin_acc": acc, "bin_conf": conf, "bin_n": self.n.copy()}


def entropy(prob: torch.Tensor) -> torch.Tensor:
    """Predictive entropy over the class axis 0, normalised to [0, 1] by log(C)."""
    c = prob.shape[0]
    return -(prob.clamp_min(1e-12).log() * prob).sum(0) / np.log(c)
