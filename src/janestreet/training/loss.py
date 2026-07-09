"""Loss = 1 - weighted-R². Matches Volkova's ``WeightedR2Loss``."""

from __future__ import annotations

import torch
from torch import nn


class WeightedR2Loss(nn.Module):
    """Returns the (positive) weighted MSE / weighted second moment ratio.

    Minimizing this is equivalent to maximizing the competition metric.
    """

    def __init__(self, epsilon: float = 1e-38) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        num = torch.sum(weights * (y_pred - y_true) ** 2)
        den = torch.sum(weights * y_true ** 2) + self.epsilon
        return num / den
