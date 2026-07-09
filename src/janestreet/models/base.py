"""Common contract for every model used in this pipeline.

The interface is intentionally minimal: ``fit`` / ``predict`` / ``update``
(optional). Inputs / outputs are numpy so we can swap GRU, Transformer, XGB,
or anything else through the same orchestration code.

Notation
--------
* ``X``: (N, K) feature matrix, already standardized and NaN-filled.
* ``resp``: (N, A) auxiliary-target matrix. May be empty when A = 0.
* ``y``: (N,) primary target.
* ``w``: (N,) sample weights.
* ``meta``: indexing arrays (symbols, dates, times) — needed by recurrent
  models to reshape into (D, T, K) per date.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FitData:
    X: np.ndarray
    resp: np.ndarray
    y: np.ndarray
    w: np.ndarray
    symbols: np.ndarray
    dates: np.ndarray
    times: np.ndarray


class BaseModel(ABC):
    """Common interface for any model in the pipeline."""

    # Whether this model needs the per-date (D, T, K) shape (recurrent models)
    # vs. flat (N, K) tabular layout (XGB / MLP).
    sequence_model: bool = False

    @abstractmethod
    def fit(self, train: FitData, valid: FitData | None = None, verbose: bool = False) -> None: ...

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        n_times: int | None = None,
        state: object | None = None,
    ) -> tuple[np.ndarray, object | None]:
        """Return predictions and (optional) state to carry across calls."""

    def update(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, n_times: int) -> None:
        """Optional online refit hook. No-op by default."""
        return None

    # Hooks for serialization. Default keeps it simple; subclasses override.
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None
