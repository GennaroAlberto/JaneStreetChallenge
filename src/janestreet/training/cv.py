"""Time-series CV on ``date_id`` with optional gap.

We split by *unique date* rather than by row, so symbols never leak across
fold boundaries. The pattern matches Volkova's: ``n_splits`` expanding-window
folds of ``test_size`` dates each, with an optional ``gap`` to simulate the
private-LB time gap.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class TimeSeriesDateSplit:
    """Expanding-window CV on a sorted array of unique dates.

    Each yielded pair is ``(train_dates, valid_dates)``.
    """

    def __init__(
        self,
        n_splits: int = 2,
        test_size: int = 200,
        gap: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(self, dates: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        d = np.sort(np.unique(dates))
        n = len(d)
        test_size = self.test_size if n > self.test_size * (self.n_splits + 1) else n // (self.n_splits + 1)
        # last index of the validation window for each fold (latest fold ends at n)
        valid_ends = [n - i * test_size for i in range(self.n_splits - 1, -1, -1)]
        for end in valid_ends:
            v_start = end - test_size
            t_end = v_start - self.gap
            if t_end <= 0:
                continue
            t_start = max(0, t_end - self.max_train_size) if self.max_train_size else 0
            yield d[t_start:t_end], d[v_start:end]
