"""A drive-risk overlay from class probabilities and model uncertainty.

This is a DEMONSTRATION of how a terrain classifier could feed a path planner, not a validated mobility
model. The per-class costs are illustrative and only ordered by the hazards named in Swan et al. (2021):
sand traps (Spirit was immobilised by one; Curiosity was nearly embedded at Hidden Valley), rocks taller
than ~30 cm (the definition of the "big rock" class), and pointy rocks on hard ground (wheel damage).
Uncertainty is added as a cost: where the network does not know, a cautious planner should not trust it."""
from __future__ import annotations

import numpy as np

# soil, bedrock, sand, big rock
DEFAULT_COST = np.array([0.10, 0.30, 0.75, 1.00], dtype=np.float32)


def risk_map(prob: np.ndarray, uncertainty: np.ndarray | None = None, cost=DEFAULT_COST,
             unc_weight: float = 1.0, invalid: np.ndarray | None = None) -> np.ndarray:
    """prob: (4, H, W) probabilities; uncertainty: (H, W) in [0, 1] (e.g. normalised entropy).
    Returns expected cost + unc_weight * uncertainty, clipped to [0, 1]; NaN where invalid."""
    r = np.tensordot(np.asarray(cost, np.float32), prob, axes=(0, 0))
    if uncertainty is not None:
        r = r + unc_weight * uncertainty
    r = np.clip(r, 0.0, 1.0)
    if invalid is not None:
        r = r.astype(np.float32)
        r[invalid] = np.nan
    return r


def zones(risk: np.ndarray, go: float = 0.3, nogo: float = 0.6) -> np.ndarray:
    """0 = go, 1 = caution, 2 = no-go, 255 = not assessed."""
    z = np.full(risk.shape, 255, np.uint8)
    ok = np.isfinite(risk)
    z[ok & (risk < go)] = 0
    z[ok & (risk >= go) & (risk < nogo)] = 1
    z[ok & (risk >= nogo)] = 2
    return z
