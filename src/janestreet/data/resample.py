"""Date-level resampling for bagged model training.

Each model in an ensemble is trained on a *different* resample of the
training date pool (dates >= 700, where ``n_times`` stabilizes at 968).
Because the sequence models treat one date as one batch (see
``DateBatchDataset``), resampling *dates* is exactly resampling *batches* —
which is what gives the bagged models decorrelated errors, the property a
simple-average ensemble needs.

The preprocessor (standardization stats) is fit once on the whole pool and
shared across all bags — only the choice of training dates differs.

Modes
-----

* ``subsample`` (default): draw ``frac × |pool|`` dates WITHOUT replacement.
  Each date appears at most once, so ``DateBatchDataset``'s group-by-date
  yields distinct batches. This is "subbagging" and is the clean, RAM-light
  default (a 0.5-frac model trains on ~half the dates → ~half the memory).
  ``frac=0.63`` reproduces the ~63% unique-sample fraction of a bootstrap.

* ``bootstrap``: draw ``|pool|`` dates WITH replacement. Duplicates are real
  — the caller must assign each drawn occurrence a distinct group id (see
  ``resolve_bootstrap_groups``) so ``DateBatchDataset`` doesn't merge them.
  Classic bagging; heavier (full pool size per model).
"""

from __future__ import annotations

import numpy as np


def sample_dates(
    pool: np.ndarray,
    seed: int,
    mode: str = "subsample",
    frac: float = 0.63,
) -> np.ndarray:
    """Return the date_ids this bag member trains on.

    Parameters
    ----------
    pool
        1-D array of candidate training date_ids (already filtered to the
        train range, e.g. 700..train_until).
    seed
        Per-model seed — different seeds give different resamples.
    mode
        ``"subsample"`` (without replacement) or ``"bootstrap"`` (with).
    frac
        Fraction of the pool to draw in ``subsample`` mode. Ignored for
        ``bootstrap`` (which always draws ``|pool|`` with replacement).

    Returns
    -------
    For ``subsample``: a sorted array of distinct date_ids.
    For ``bootstrap``: an array of date_ids of length ``|pool|`` that MAY
    contain duplicates, in draw order (caller assigns group ids).
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool)
    if mode == "subsample":
        k = max(1, int(round(frac * len(pool))))
        idx = rng.choice(len(pool), size=k, replace=False)
        return np.sort(pool[idx])
    if mode == "bootstrap":
        idx = rng.choice(len(pool), size=len(pool), replace=True)
        return pool[idx]  # order = draw order; duplicates kept
    raise ValueError(f"mode must be 'subsample' or 'bootstrap', got {mode!r}")


def resolve_bootstrap_groups(drawn_dates: np.ndarray) -> np.ndarray:
    """Assign a distinct group id to each drawn occurrence.

    For a bootstrap draw ``[712, 712, 900, ...]`` the two 712s must become
    two separate batches. We return a parallel array of synthetic group ids
    ``[0, 1, 2, ...]`` (one per occurrence). The caller uses these as the
    ``dates`` array fed to ``DateBatchDataset`` so its group-by-value keeps
    the duplicates apart, while ``times`` (0..967) still drives the
    within-batch reshape correctly.
    """
    return np.arange(len(drawn_dates), dtype=np.int64)
