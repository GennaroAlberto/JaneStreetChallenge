"""Online twin of the offline ``prepare_dataset`` / ``Preprocessor`` path.

``FeatureState`` rebuilds the exact model-input matrix (134 columns with the
default recipe) incrementally, one ``(date_id, time_id)`` batch at a time, so
a Kaggle inference kernel can serve the RNN blend without ever materializing
the offline frame. It is validated against the offline path by
``scripts/serving/replay_check.py``.

Offline recipe being mirrored (src/janestreet/data/features.py, polars 1.40
semantics verified empirically):

* raw features           — 79 native cols minus the 3 categoricals = 76,
                           passed through untouched (nulls stay NaN until the
                           standardizer maps them to 0).
* ``{c}_diff_rolling_avg_{T}`` — ``c - rolling_mean(c, window_size=T)`` per
  symbol, window counted in the symbol's rows across dates. polars
  ``rolling_mean(window_size=T)`` leaves ``min_samples`` at its default (=T):
  the value is null unless the trailing T-row window holds T NON-NULL values.
  Any null inside the window ⇒ null out. Mirrored here by a per-symbol ring
  buffer holding NaN for nulls: ``np.mean`` propagates the NaN naturally, and
  a partially-filled buffer emits NaN.
* ``{c}_rolling_std_{T}``  — same window, sample std (polars ddof=1).
* ``{c}_avg_per_date_time`` — mean over the symbols present at this
  ``(date, time)`` — i.e. a null-ignoring mean over the current batch.
* ``feature_time_id``      — ``float(time_id)``.
* ``responder_{r}_lag1d``  — yesterday's responder_r at the same
  ``(symbol, time)``; missing symbol/time (or no previous date) ⇒ 0.0,
  matching the offline ``fill_null(0.0)`` after the date+1 self-join.
* ``rsig_*``               — differences of the (already 0-filled) lagged
  responders, per features.py ``_add_responder_signal_features``.

Standardization mirrors ``Preprocessor.transform`` exactly (clip to train
quantiles → z-score with the Welford stats → nan_to_num(0)). The online
daily-refit protocol from ``run_cv`` is reproduced too: ``new_date`` first
``partial_fit``s the scaler on yesterday's clipped block (iff
``refit_scaler=True``, the default — this is what ``pipe.update(prev)`` does
via ``transform(refit=True)``) and returns yesterday's (X, y, w) so the
caller can run the model's online ``update``; today's ``push`` calls then
standardize with the just-updated stats, exactly like
``run_cv``'s update-then-predict loop.

TODO(v2): vol-scaling — the |y| XGB model consumes the same engineered block;
hook: ``FeatureState.last_raw`` (pre-clip engineered rows of the latest push)
and ``RefitBlock.X_raw``. TODO(v2): the XGB prediction stream (241-feature
pool) needs its own feature superset — extend the parser here rather than
forking the class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from janestreet.config import COL_ID, COL_TIME, COL_WEIGHT

_DIFF_RE = re.compile(r"^(?P<c>.+)_diff_rolling_avg_(?P<t>\d+)$")
_STD_RE = re.compile(r"^(?P<c>.+)_rolling_std_(?P<t>\d+)$")
_MAVG_RE = re.compile(r"^(?P<c>.+)_avg_per_date_time$")
_LAG_RE = re.compile(r"^responder_(?P<r>\d+)_lag1d$")

# rsig name -> (a, b) meaning responder_a_lag1d - responder_b_lag1d
# (features.py _add_responder_signal_features, computed AFTER the 0-fill).
_RSIG = {
    "rsig_momA_short": (8, 6),
    "rsig_momA_med": (6, 7),
    "rsig_momB_short": (5, 3),
    "rsig_momB_med": (3, 4),
    "rsig_spread_fine": (8, 5),
}


@dataclass
class RefitBlock:
    """Yesterday's tensors for the caller's online model refit.

    ``X`` is flat time-major — all symbols (ascending id) at the day's first
    time_id, then the next, … — with a CONSTANT symbol panel, which is exactly
    the layout ``RecurrentModel.update(X, y, w, n_times)`` /
    ``reshape_flat_to_sequence`` require. Symbols that were absent from any
    timestep (or lack a lag y) are dropped from the whole day to keep the
    rectangle.
    """

    X: np.ndarray          # (n_times * S, K) float32, standardized (or raw if no preprocessor)
    y: np.ndarray          # (n_times * S,) float32 — responder_6 from the lags frame
    w: np.ndarray          # (n_times * S,) float32 — weights cached from the test batches
    n_times: int
    symbol_ids: np.ndarray  # (S,) int64, ascending — the kept panel
    X_raw: np.ndarray      # (n_times * S, K) float32 pre-clip engineered block (v2 hooks)


@dataclass
class _Step:
    time_id: int
    symbols: np.ndarray    # (S,) int64 ascending
    raw: np.ndarray        # (S, K) float32 engineered, pre-clip (NaN preserved)
    weights: np.ndarray    # (S,) float32 (NaN if the batch had no weight col)


class _SymbolRoll:
    """Trailing ring buffer of the last ``window`` corr-feature rows."""

    __slots__ = ("buf", "pos", "count")

    def __init__(self, window: int, n_cols: int) -> None:
        self.buf = np.full((window, n_cols), np.nan, dtype=np.float64)
        self.pos = 0
        self.count = 0

    def push(self, row: np.ndarray) -> None:
        self.buf[self.pos] = row
        self.pos = (self.pos + 1) % self.buf.shape[0]
        if self.count < self.buf.shape[0]:
            self.count += 1


class FeatureState:
    """Incremental producer of the exact offline model-input matrix.

    Parameters
    ----------
    preprocessor:
        The pickled train-time ``janestreet.pipeline.Preprocessor`` (clipper +
        Welford scaler) from a checkpoint, or ``None`` to emit raw engineered
        values (NaN preserved) — the mode ``replay_check.py`` uses to isolate
        feature-engineering correctness.
    feature_cols:
        The checkpoint's ordered feature list (134 with the default recipe).
        This is authoritative: the recipe is parsed from the names.
    lagged_responders:
        ``cfg.lagged_responders`` — used only to cross-check the parsed names.
    rolling_window:
        ``cfg.rolling_window`` (1000) — must match the suffix in the names.
    n_symbols_hint:
        Advisory expected panel width; state is keyed by ``symbol_id`` and
        grows on demand, so appearing/disappearing symbols are handled.
    refit_scaler:
        Mirror ``Preprocessor.transform(refit=True)`` on each ``new_date``
        (partial_fit the Welford scaler with yesterday's clipped block).
        Default True = the ``run_cv`` walk protocol the checkpoints scored
        under. Ignored when ``preprocessor`` is None.
    """

    def __init__(
        self,
        preprocessor,
        feature_cols: list[str],
        lagged_responders: list[int],
        rolling_window: int = 1000,
        n_symbols_hint: int = 64,
        *,
        refit_scaler: bool = True,
        target_responder: int = 6,
    ) -> None:
        self.preprocessor = preprocessor
        self.feature_cols = list(feature_cols)
        self.rolling_window = int(rolling_window)
        self.n_symbols_hint = int(n_symbols_hint)
        self.refit_scaler = bool(refit_scaler)
        self.target_responder = int(target_responder)

        if preprocessor is not None and list(preprocessor.feature_cols) != self.feature_cols:
            raise ValueError(
                "preprocessor.feature_cols disagrees with feature_cols — "
                "wrong checkpoint/recipe pairing."
            )

        # ---- parse the recipe out of the column names --------------------
        corr_cols: list[str] = []          # ordered, unique
        corr_pos: dict[str, int] = {}
        raw_entries: list[tuple[int, str]] = []
        diff_entries: list[tuple[int, str]] = []
        std_entries: list[tuple[int, str]] = []
        mavg_entries: list[tuple[int, str]] = []
        lag_entries: list[tuple[int, int]] = []
        rsig_entries: list[tuple[int, tuple[int, int]]] = []
        time_out: int | None = None

        def _corr(c: str) -> int:
            if c not in corr_pos:
                corr_pos[c] = len(corr_cols)
                corr_cols.append(c)
            return corr_pos[c]

        for j, name in enumerate(self.feature_cols):
            m = _DIFF_RE.match(name)
            if m:
                if int(m["t"]) != self.rolling_window:
                    raise ValueError(
                        f"{name}: window {m['t']} != rolling_window {self.rolling_window}"
                    )
                _corr(m["c"])
                diff_entries.append((j, m["c"]))
                continue
            m = _STD_RE.match(name)
            if m:
                if int(m["t"]) != self.rolling_window:
                    raise ValueError(
                        f"{name}: window {m['t']} != rolling_window {self.rolling_window}"
                    )
                _corr(m["c"])
                std_entries.append((j, m["c"]))
                continue
            m = _MAVG_RE.match(name)
            if m:
                _corr(m["c"])
                mavg_entries.append((j, m["c"]))
                continue
            m = _LAG_RE.match(name)
            if m:
                lag_entries.append((j, int(m["r"])))
                continue
            if name == "feature_time_id":
                time_out = j
                continue
            if name in _RSIG:
                rsig_entries.append((j, _RSIG[name]))
                continue
            # anything left is a raw passthrough column from the test batch
            raw_entries.append((j, name))

        found_lags = sorted({r for _, r in lag_entries})
        if lagged_responders and found_lags != sorted(set(lagged_responders)):
            raise ValueError(
                f"lag features in feature_cols {found_lags} != "
                f"cfg.lagged_responders {sorted(set(lagged_responders))}"
            )

        # responders we must keep from each lags frame: the lag features, the
        # rsig operands, and the target (for the refit y).
        need_resp = set(found_lags) | {self.target_responder}
        for _, (a, b) in rsig_entries:
            need_resp |= {a, b}
        self._lag_resps: list[int] = sorted(need_resp)
        self._lag_col_of = {r: i for i, r in enumerate(self._lag_resps)}

        # raw columns pulled from each incoming batch: corr cols + passthroughs
        raw_needed: list[str] = list(corr_cols)
        for _, c in raw_entries:
            if c not in raw_needed:
                raw_needed.append(c)
        self._raw_needed = raw_needed
        src_pos = {c: i for i, c in enumerate(raw_needed)}

        self._corr_cols = corr_cols
        self._corr_src = np.array([src_pos[c] for c in corr_cols], dtype=np.int64)
        self._raw_out = np.array([j for j, _ in raw_entries], dtype=np.int64)
        self._raw_src = np.array([src_pos[c] for _, c in raw_entries], dtype=np.int64)
        self._diff_out = np.array([j for j, _ in diff_entries], dtype=np.int64)
        self._diff_cidx = np.array([corr_pos[c] for _, c in diff_entries], dtype=np.int64)
        self._std_out = np.array([j for j, _ in std_entries], dtype=np.int64)
        self._std_cidx = np.array([corr_pos[c] for _, c in std_entries], dtype=np.int64)
        self._mavg_out = np.array([j for j, _ in mavg_entries], dtype=np.int64)
        self._mavg_cidx = np.array([corr_pos[c] for _, c in mavg_entries], dtype=np.int64)
        self._lag_out = np.array([j for j, _ in lag_entries], dtype=np.int64)
        self._lag_lidx = np.array(
            [self._lag_col_of[r] for _, r in lag_entries], dtype=np.int64
        )
        self._rsig = [(j, self._lag_col_of[a], self._lag_col_of[b])
                      for j, (a, b) in rsig_entries]
        self._time_out = time_out

        # ---- mutable state ----------------------------------------------
        self._roll: dict[int, _SymbolRoll] = {}
        self._lag_index: dict[tuple[int, int], int] = {}   # (sym, time) -> row
        self._lag_matrix: np.ndarray = np.zeros((0, len(self._lag_resps)), np.float64)
        self._steps: list[_Step] = []
        self.last_raw: np.ndarray | None = None   # v2 hook (vol model / XGB)
        self.last_symbols: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    # ------------------------------------------------------------------
    def seed_history(self, df_raw: pl.DataFrame) -> None:
        """Warm-start the rolling buffers from a raw training-tail frame.

        ``df_raw`` needs (date_id, time_id, symbol_id, <corr feature cols>);
        for exact parity feed >= ``rolling_window`` rows per symbol (~2 dates).
        Does NOT touch the lag lookup — the first ``new_date(lags)`` does.
        """
        from janestreet.config import COL_DATE  # local: keep top imports lean

        df_raw = df_raw.sort([COL_DATE, COL_TIME])
        w = self.rolling_window
        for part in df_raw.partition_by(COL_ID, maintain_order=True):
            sym = int(part[COL_ID][0])
            vals = (
                part.select(self._corr_cols)
                .tail(w)
                .to_numpy()
                .astype(np.float64)
            )
            st = _SymbolRoll(w, len(self._corr_cols))
            n = vals.shape[0]
            st.buf[:n] = vals
            st.count = n
            st.pos = n % w
            self._roll[sym] = st

    # ------------------------------------------------------------------
    def new_date(self, lags_df: pl.DataFrame | None) -> RefitBlock | None:
        """Start a new date.

        1. Assemble yesterday's (X, y, w) from the cached pushes + the lags
           frame (which carries yesterday's realized responders) and — iff a
           preprocessor is attached and ``refit_scaler`` — partial_fit the
           Welford scaler on yesterday's clipped block first, mirroring
           ``Preprocessor.transform(refit=True)`` in ``FullPipeline.update``.
        2. Install ``lags_df`` as today's ``responder_*_lag1d`` source.
        3. Clear the per-day cache.

        ``lags_df`` accepts Kaggle naming (``responder_{r}_lag_1``) or plain
        ``responder_{r}``; keyed on (symbol_id, time_id), date column ignored.
        Returns None when there is nothing to refit on (first date / no lags).
        """
        block = self._assemble_refit_block(lags_df)
        self._set_lags(lags_df)
        self._steps = []
        return block

    # ------------------------------------------------------------------
    def push(self, test_batch_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ingest one (date_id, time_id) batch; return (X_step, symbol_ids).

        ``X_step`` is float32 (S, K) with rows sorted by ascending
        ``symbol_id`` (a stable order the caller needs for RNN hidden-state
        carry); ``symbol_ids`` maps rows back to the batch. Standardized when
        a preprocessor is attached, else raw engineered values with NaN
        preserved.
        """
        time_id = int(test_batch_df[COL_TIME][0])
        syms = test_batch_df[COL_ID].to_numpy().astype(np.int64)
        order = np.argsort(syms, kind="stable")
        syms = syms[order]

        raw = (
            test_batch_df.select(self._raw_needed)
            .to_numpy()
            .astype(np.float64)[order]
        )
        s_count = raw.shape[0]
        n_corr = len(self._corr_cols)
        corr = raw[:, self._corr_src] if n_corr else np.zeros((s_count, 0))

        # market average per (date, time): null-ignoring mean over the batch
        if n_corr:
            finite = ~np.isnan(corr)
            cnt = finite.sum(axis=0)
            tot = np.where(finite, corr, 0.0).sum(axis=0)
            mavg = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        else:
            mavg = np.zeros(0)

        # per-symbol trailing window stats (include the current row, like
        # polars rolling_* which close the window on the current row)
        means = np.full((s_count, n_corr), np.nan)
        stds = np.full((s_count, n_corr), np.nan)
        w = self.rolling_window
        for i in range(s_count):
            sym = int(syms[i])
            st = self._roll.get(sym)
            if st is None:
                st = _SymbolRoll(w, n_corr)
                self._roll[sym] = st
            st.push(corr[i])
            if st.count == w:
                # plain (not nan-) mean/std: a NaN anywhere in the window
                # must poison the result, matching polars min_samples=T.
                means[i] = st.buf.mean(axis=0)
                stds[i] = st.buf.std(axis=0, ddof=1)

        # lagged responders (0.0 default == offline fill_null(0.0))
        lag = np.zeros((s_count, len(self._lag_resps)), dtype=np.float64)
        for i in range(s_count):
            idx = self._lag_index.get((int(syms[i]), time_id))
            if idx is not None:
                lag[i] = self._lag_matrix[idx]

        # ---- assemble in feature_cols order ------------------------------
        out = np.empty((s_count, self.n_features), dtype=np.float64)
        if self._raw_out.size:
            out[:, self._raw_out] = raw[:, self._raw_src]
        if self._diff_out.size:
            out[:, self._diff_out] = corr[:, self._diff_cidx] - means[:, self._diff_cidx]
        if self._std_out.size:
            out[:, self._std_out] = stds[:, self._std_cidx]
        if self._mavg_out.size:
            out[:, self._mavg_out] = np.broadcast_to(
                mavg[self._mavg_cidx], (s_count, self._mavg_out.size)
            )
        if self._time_out is not None:
            out[:, self._time_out] = float(time_id)
        if self._lag_out.size:
            out[:, self._lag_out] = lag[:, self._lag_lidx]
        for j, ai, bi in self._rsig:
            out[:, j] = lag[:, ai] - lag[:, bi]

        if COL_WEIGHT in test_batch_df.columns:
            weights = (
                test_batch_df[COL_WEIGHT].to_numpy().astype(np.float32)[order]
            )
        else:
            weights = np.full(s_count, np.nan, dtype=np.float32)

        raw32 = out.astype(np.float32)
        self._steps.append(_Step(time_id, syms, raw32, weights))
        self.last_raw = raw32
        self.last_symbols = syms

        if self.preprocessor is not None:
            x_step = self._transform_block(raw32, refit=False)
        else:
            x_step = raw32
        return x_step.astype(np.float32, copy=False), syms

    # ------------------------------------------------------------------
    def _transform_block(self, x: np.ndarray, refit: bool) -> np.ndarray:
        """Exact port of ``Preprocessor.transform`` on a numpy block."""
        pre = self.preprocessor
        xc = pre.clipper.transform(np.asarray(x, dtype=np.float32))
        if refit:
            pre.scaler.partial_fit(xc)
        return pre.scaler.transform(xc)

    # ------------------------------------------------------------------
    def _lag_col_names(self, lags_df: pl.DataFrame) -> dict[int, str]:
        cols = set(lags_df.columns)
        picked: dict[int, str] = {}
        for r in self._lag_resps:
            for cand in (f"responder_{r}_lag_1", f"responder_{r}"):
                if cand in cols:
                    picked[r] = cand
                    break
            else:
                raise ValueError(
                    f"lags frame is missing responder_{r} "
                    f"(looked for responder_{r}_lag_1 / responder_{r})"
                )
        return picked

    def _set_lags(self, lags_df: pl.DataFrame | None) -> None:
        if lags_df is None or lags_df.height == 0:
            self._lag_index = {}
            self._lag_matrix = np.zeros((0, len(self._lag_resps)), np.float64)
            return
        picked = self._lag_col_names(lags_df)
        mat = (
            lags_df.select([picked[r] for r in self._lag_resps])
            .to_numpy()
            .astype(np.float64)
        )
        # offline: missing/null joins are fill_null(0.0)-ed before use
        np.nan_to_num(mat, copy=False)
        syms = lags_df[COL_ID].to_numpy().astype(np.int64)
        times = lags_df[COL_TIME].to_numpy().astype(np.int64)
        self._lag_matrix = mat
        self._lag_index = {
            (int(s), int(t)): i for i, (s, t) in enumerate(zip(syms, times))
        }

    # ------------------------------------------------------------------
    def _assemble_refit_block(
        self, lags_df: pl.DataFrame | None
    ) -> RefitBlock | None:
        if not self._steps or lags_df is None or lags_df.height == 0:
            return None
        picked = self._lag_col_names(lags_df)
        y_vals = (
            lags_df[picked[self.target_responder]]
            .to_numpy()
            .astype(np.float64)
        )
        np.nan_to_num(y_vals, copy=False)
        syms_l = lags_df[COL_ID].to_numpy().astype(np.int64)
        times_l = lags_df[COL_TIME].to_numpy().astype(np.int64)
        y_lookup = {
            (int(s), int(t)): y_vals[i]
            for i, (s, t) in enumerate(zip(syms_l, times_l))
        }

        steps = sorted(self._steps, key=lambda st: st.time_id)
        # constant panel: symbols present at EVERY timestep with a y for each
        keep = set(steps[0].symbols.tolist())
        for st in steps[1:]:
            keep &= set(st.symbols.tolist())
        keep = {
            s for s in keep
            if all((s, st.time_id) in y_lookup for st in steps)
        }
        if not keep:
            return None
        panel = np.array(sorted(keep), dtype=np.int64)
        s_count = panel.size
        n_times = len(steps)
        k = self.n_features

        x_raw = np.empty((n_times * s_count, k), dtype=np.float32)
        y = np.empty(n_times * s_count, dtype=np.float32)
        w = np.empty(n_times * s_count, dtype=np.float32)
        for ti, st in enumerate(steps):
            # st.symbols is sorted ascending and contains every panel symbol
            pos = np.searchsorted(st.symbols, panel)
            lo = ti * s_count
            x_raw[lo:lo + s_count] = st.raw[pos]
            w[lo:lo + s_count] = st.weights[pos]
            for i, s in enumerate(panel):
                y[lo + i] = y_lookup[(int(s), st.time_id)]

        if self.preprocessor is not None:
            x = self._transform_block(x_raw, refit=self.refit_scaler)
            x = x.astype(np.float32, copy=False)
        else:
            x = x_raw
        return RefitBlock(
            X=x, y=y, w=w, n_times=n_times, symbol_ids=panel, X_raw=x_raw
        )
