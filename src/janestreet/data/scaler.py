"""Online-safe standardization.

Two scalers:

* ``OnlineStandardizer`` keeps a Welford running mean / variance per feature.
  ``fit`` consumes a numpy array and updates state; ``transform`` z-scores
  in place. We use it both to standardize training data and (with a learning
  rate) keep adapting to the incoming test distribution at inference time.

* ``QuantileClipper`` is a complementary outlier killer — fit once on the
  training data, then clip to the learned quantile range. NaN-safe.

The default training pipeline applies (clip → standardize → nan_to_num).
"""

from __future__ import annotations

import numpy as np


class OnlineStandardizer:
    """Welford running mean / variance per column."""

    def __init__(self, n_features: int | None = None) -> None:
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None  # sum of squared deviations
        self._count: float = 0.0
        if n_features is not None:
            self._mean = np.zeros(n_features, dtype=np.float64)
            self._m2 = np.zeros(n_features, dtype=np.float64)

    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray) -> OnlineStandardizer:
        """Initialize state from a single block (NaN-safe)."""
        self._mean = np.nanmean(x, axis=0).astype(np.float64)
        var = np.nanvar(x, axis=0).astype(np.float64)
        # We store M2 = var * count so subsequent partial_fit calls work.
        valid = np.sum(~np.isnan(x), axis=0).astype(np.float64)
        self._count = float(np.nanmedian(valid)) if valid.size else 0.0
        self._m2 = var * self._count
        return self

    def partial_fit(self, x: np.ndarray) -> OnlineStandardizer:
        """Welford update with another batch."""
        if self._mean is None or self._m2 is None:
            return self.fit(x)
        # NaN-safe per-column update — use column means / counts.
        x_mean = np.nanmean(x, axis=0).astype(np.float64)
        n_new = np.sum(~np.isnan(x), axis=0).astype(np.float64)
        n_new = np.nanmedian(n_new) if n_new.size else 0.0
        if n_new == 0:
            return self
        delta = x_mean - self._mean
        total = self._count + n_new
        new_mean = self._mean + delta * (n_new / total)
        # Pooled variance update (Chan et al. parallel algorithm).
        x_var = np.nanvar(x, axis=0).astype(np.float64)
        m2_new = x_var * n_new
        self._m2 = self._m2 + m2_new + delta * delta * (self._count * n_new / total)
        self._mean = new_mean
        self._count = float(total)
        return self

    # ------------------------------------------------------------------
    @property
    def mean_(self) -> np.ndarray:
        assert self._mean is not None
        return self._mean

    @property
    def std_(self) -> np.ndarray:
        assert self._m2 is not None
        return np.sqrt(np.maximum(self._m2 / max(self._count, 1.0), 1e-12))

    # ------------------------------------------------------------------
    def transform(self, x: np.ndarray) -> np.ndarray:
        # The Welford stats are Float64 for accuracy of the running update;
        # broadcasting them against a Float16/Float32 x would silently promote
        # the whole (potentially multi-GB) array to Float64 during transform.
        # Cast the stats to Float32 for the arithmetic — precision loss on
        # standardised features is well below the noise floor.
        mean32 = self.mean_.astype(np.float32, copy=False)
        std32 = self.std_.astype(np.float32, copy=False)
        out = (x.astype(np.float32, copy=False) - mean32) / std32
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class QuantileClipper:
    """Clip features to a learned [low_q, high_q] range to kill outliers."""

    def __init__(self, low_q: float = 0.001, high_q: float = 0.999) -> None:
        self.low_q = low_q
        self.high_q = high_q
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> QuantileClipper:
        self.lo_ = np.nanquantile(x, self.low_q, axis=0).astype(np.float32)
        self.hi_ = np.nanquantile(x, self.high_q, axis=0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.lo_ is not None and self.hi_ is not None
        return np.clip(x, self.lo_, self.hi_)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
