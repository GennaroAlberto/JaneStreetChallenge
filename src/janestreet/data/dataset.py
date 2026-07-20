"""PyTorch dataset shaping rows into (symbol, time, features) per date.

This is a direct generalization of Volkova's ``CustomTensorDataset`` with
clearer naming. The invariant the Kaggle data guarantees is::

    each (symbol_id, date_id) row group has exactly T = 968 entries
    indexed by time_id = 0..T-1

So given a flat matrix of rows we can ``view(N // T, T, K)`` after sorting
by (symbol, date, time). The dataset then groups by ``date_id`` and at each
iteration yields the slice ``(D, T, K)`` where ``D`` is the number of
symbols active that date.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def flatten_collate_fn(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Concatenate per-date items along the batch dim — keeps shape (sum_D, T, ...)."""
    n = len(batch[0])
    out = tuple(torch.cat([b[i] for b in batch], dim=0) for i in range(n))
    return out


class DateBatchDataset(Dataset):
    """Group rows by ``date_id`` and yield 3-D tensors per date.

    Parameters
    ----------
    X:
        Features matrix of shape (N, K) — already standardized & NaN-filled.
    resp:
        Auxiliary-target columns matrix (N, A). May be empty (N, 0).
    y:
        Primary target vector (N,).
    weights:
        Sample weight vector (N,).
    symbols, dates, times:
        Indexing arrays (all length N, integer).
    n_times:
        Time-bucket count per (symbol, date). 968 for Jane Street data.
    on_batch:
        If True, the (D, T, K) reshape happens inside ``__getitem__`` and we
        don't pre-sort. Useful when T per date can vary at inference. Set to
        False during training for speed (Volkova's choice).
    """

    def __init__(
        self,
        X: np.ndarray,
        resp: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        symbols: np.ndarray,
        dates: np.ndarray,
        times: np.ndarray,
        n_times: int = 968,
        on_batch: bool = False,
    ) -> None:
        self.on_batch = on_batch
        self.n_times = n_times
        self.k = X.shape[1]

        # Memmap-aware path: if X is a numpy memmap (as produced by
        # scripts/precompute_dataset.py), do NOT eagerly torch.from_numpy
        # it — that would materialise the whole array in RAM and defeat the
        # purpose. We keep X as a numpy view; __getitem__ reads per-date
        # slices on demand. Requires ``on_batch=True`` because presort
        # would touch every row.
        self.x_is_memmap = isinstance(X, np.memmap)
        self._X_np: np.ndarray | None = None
        if self.x_is_memmap:
            if not on_batch:
                raise ValueError(
                    "memmap-backed X requires on_batch=True — cannot presort "
                    "without materialising the whole file."
                )
            self.X = X  # numpy memmap, streamed per __getitem__
        else:
            X_np = np.ascontiguousarray(X)
            if X_np.dtype == np.float32:
                # Zero-copy path: share the caller's buffer, clean NaNs in
                # place, and let _presort_and_view permute in place too.
                # The old out-of-place nan_to_num + fancy-index presort held
                # X three times transiently (~14 GB on a 240-date window) —
                # exactly what OOM-killed free-tier Colab runs.
                np.nan_to_num(X_np, copy=False)
                self._X_np = X_np
                self.X = torch.from_numpy(X_np)
            else:
                self.X = torch.nan_to_num(
                    torch.from_numpy(X_np).float(), 0.0)

        self.resp = torch.from_numpy(np.ascontiguousarray(resp)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
        self.weights = torch.from_numpy(np.ascontiguousarray(weights)).float()
        self.symbols = torch.from_numpy(np.ascontiguousarray(symbols)).long()
        self.dates = torch.from_numpy(np.ascontiguousarray(dates)).long()
        self.times = torch.from_numpy(np.ascontiguousarray(times)).long()

        if not self.on_batch:
            self._presort_and_view()

        unique, inv, counts = torch.unique(self.dates, return_inverse=True, return_counts=True)
        self.unique_dates = unique
        self.sorted_idx = torch.argsort(inv)
        self.group_end = torch.cumsum(counts, dim=0)
        self.group_start = torch.cat([torch.tensor([0]), self.group_end[:-1]])

    # ------------------------------------------------------------------
    def _presort_and_view(self) -> None:
        t = self.n_times
        n = self.X.shape[0]
        if n % t != 0:
            raise ValueError(
                f"Row count {n} not divisible by n_times={t} — cannot reshape. "
                "Pass on_batch=True if your time count varies."
            )
        # Sort by (symbol, date, time) using stable sorts in reverse priority.
        idx = torch.argsort(self.times, stable=True)
        idx = idx[torch.argsort(self.dates[idx], stable=True)]
        idx = idx[torch.argsort(self.symbols[idx], stable=True)]
        if self._X_np is not None:
            # In-place chunked permutation of the shared buffer: peak extra
            # memory is one (N, 16) slice (~0.7 GB at 280 dates) instead of
            # a full second copy of X. self.X already views this buffer.
            idx_np = idx.numpy()
            for j in range(0, self.k, 16):
                self._X_np[:, j:j + 16] = self._X_np[idx_np, j:j + 16]
        else:
            self.X = self.X[idx]
        self.resp = self.resp[idx]
        self.y = self.y[idx]
        self.weights = self.weights[idx]
        self.symbols = self.symbols[idx]
        self.dates = self.dates[idx]
        self.times = self.times[idx]

        groups = n // t
        self.X = self.X.view(groups, t, self.k)
        a = self.resp.shape[-1] if self.resp.ndim == 2 else 1
        self.resp = self.resp.view(groups, t, a) if a > 0 else self.resp.view(groups, t, 0)
        self.y = self.y.view(groups, t)
        self.weights = self.weights.view(groups, t)
        self.symbols = self.symbols.view(groups, t)
        self.dates = self.dates.view(groups, t)[:, 0]  # one date per group

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.unique_dates.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        start = int(self.group_start[index].item())
        end = int(self.group_end[index].item())
        rows = self.sorted_idx[start:end]

        if self.x_is_memmap:
            # Numpy fancy indexing on a memmap reads only the requested rows.
            # Cast Float16/32 → Float32 for the model; nan_to_num on the small
            # per-date slice is cheap (~20 MB) and avoids ever holding the
            # whole standardised matrix in memory.
            X_np = np.ascontiguousarray(self.X[rows.cpu().numpy()])
            X = torch.from_numpy(X_np).float()
            X = torch.nan_to_num(X, 0.0)
        else:
            X = self.X[rows]

        resp = self.resp[rows]
        y = self.y[rows]
        w = self.weights[rows]

        if self.on_batch:
            t = int(self.times[rows].max().item()) + 1
            # Rows may arrive in any within-date order: FitData slices are
            # symbol-major, Kaggle-style frames time-major. Sort explicitly
            # to (symbol, time) and reshape symbol-major — the old
            # time-major reshape silently interleaved the axes for
            # symbol-major input, placing future timesteps of the same
            # symbol along the cross-sectional axis (leaked into the xsec
            # attention; scrambled sequences for plain RNN validation).
            order = torch.argsort(self.symbols[rows] * t + self.times[rows])
            X = X[order].reshape(-1, t, self.k)
            resp = resp[order].reshape(-1, t, resp.shape[-1])
            y = y[order].reshape(-1, t)
            w = w[order].reshape(-1, t)

        return X, resp, y, w
