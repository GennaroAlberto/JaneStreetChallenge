"""Shared helpers for the torch-backed models."""

from __future__ import annotations

import torch


def auto_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reshape_flat_to_sequence(X: torch.Tensor, n_times: int) -> torch.Tensor:
    """Reshape (N, K) -> (D, T, K) where T = n_times and D = N // T.

    Assumes rows are sorted as repeating blocks of one (symbol, date) along
    consecutive ``time_id``s.
    """
    n, k = X.shape
    if n % n_times != 0:
        raise ValueError(f"N={n} not divisible by n_times={n_times}")
    # The Volkova convention: each block of n_times rows = one stock's day.
    # We sort externally so the layout is (time fastest, then stock).
    return X.view(n_times, n // n_times, k).swapaxes(0, 1).contiguous()


def reshape_sequence_to_flat(Y: torch.Tensor) -> torch.Tensor:
    """Reverse of ``reshape_flat_to_sequence`` for output predictions (D, T) -> (N,)."""
    return Y.swapaxes(0, 1).reshape(-1).contiguous()
