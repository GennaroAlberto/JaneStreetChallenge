"""Weighted zero-mean R² — the Jane Street competition metric.

Note: this is *not* the standard centered R². It uses ``sum(w·y²)`` in the
denominator (i.e. assumes E[y] = 0 under the weights). Higher is better.
"""

from __future__ import annotations

import numpy as np
import torch


def r2_weighted(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    num = float(np.average((y_pred - y_true) ** 2, weights=w))
    den = float(np.average(y_true ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def r2_weighted_torch(y_true: torch.Tensor, y_pred: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    num = torch.sum(w * (y_pred - y_true) ** 2)
    den = torch.sum(w * y_true ** 2) + 1e-38
    return 1.0 - num / den
