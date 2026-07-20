"""Slate-based online feature engine for the Kaggle submission kernel.

Single source of truth for the ``FeatureState`` class that
``notebooks/kaggle_submission.ipynb`` serves with — the notebook imports it
from the staged weights dataset (which ships ``scripts/serving/**``, see
``scripts/pack_submission_weights.py``) and must NOT carry an inline copy.

Relationship to ``feature_state.py`` (the row-panel engine):

* ``feature_state.FeatureState`` emits rows only for the symbols present in
  the batch (ascending ``symbol_id``) and assembles a constant-panel
  ``RefitBlock`` on ``new_date`` — the reference replica of the offline path.
* ``kernel_state.FeatureState`` (this module) emits the FULL symbol slate
  (``n_slots`` rows, indexed by ``symbol_id``) every step so RNN hidden-state
  and xsec-attention batch dims stay aligned across a day. Absent symbols get
  NaN raw features (-> 0 after standardization, exactly like training NaNs)
  and weight 0. Refit assembly (padding, y lookup) lives in the notebook.

Both engines are validated against the offline ``prepare_dataset`` frame by
``scripts/serving/replay_check.py`` (``--engine {feature_state,kernel_state}``).

Extra state the kernel relies on:

* ``active``  — per-date boolean mask over slots, True once a symbol has
  appeared in any batch today. The xsec-attention stream must only attend
  over active symbols (training never saw all-NaN pad tokens).
* ``day_store`` records carry a ``present`` mask per step so the daily
  scaler refit (``_transform_np(..., refit=True, refit_mask=...)``) can
  exclude pad/absent slate rows from the Welford statistics, matching the
  offline protocol where ``partial_fit`` only ever sees real rows.
"""

from __future__ import annotations

import io
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COLS_FEATURES_CORR

ROLL_WINDOW = 1000        # cfg.rolling_window — per-symbol rolling stats
N_TIMES = 968             # time_ids per date (stable for date_id >= 700)
N_SLOTS0 = 39             # train symbol universe 0..38; the slate grows if needed


# ---------------------------------------------------------------------------
# CPU-safe checkpoint loading (shared with scripts/ship_catchup.py).
# The 500d members were trained/pickled on CUDA; the submission rerun is
# CPU-only, so every torch storage must be rerouted through
# map_location='cpu' at unpickle time. For CPU-pickled checkpoints this is a
# no-op passthrough. torch / janestreet.pipeline are imported lazily so that
# importing kernel_state for the feature engine alone stays torch-free.
# ---------------------------------------------------------------------------
class _CPUUnpickler(pickle.Unpickler):
    """Deserialize checkpoints pickled with CUDA tensors on a CUDA-less box."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pipe_cpu(path):
    """``FullPipeline.load`` with a map_location='cpu'-safe unpickler."""
    from janestreet.pipeline import FullPipeline

    with Path(path).open("rb") as f:
        blob = _CPUUnpickler(f).load()
    return FullPipeline(cfg=blob["cfg"], model=blob["model"],
                        feature_cols=blob["feature_cols"],
                        aux_cols=blob["aux_cols"],
                        target_col=blob["target_col"],
                        preprocessor=blob["preprocessor"])


class FeatureState:
    """Streaming replica of janestreet.data.features.FeatureBuilder (134-col recipe).

    Column groups (order fixed by the checkpoints' ``feature_cols``):
      feature_XX                    raw, straight from the test batch
      {c}_diff_rolling_avg_1000     raw - rolling mean of the last 1000 rows (per symbol);
                                    NaN until the window is full and null-free (polars
                                    ``rolling_mean(window_size=1000)`` semantics)
      {c}_rolling_std_1000          rolling std, ddof=1, same NaN rule
      {c}_avg_per_date_time         cross-symbol mean of the present rows in this batch
      feature_time_id               time_id as float
      responder_{r}_lag1d           yesterday's responder at the same (symbol, time),
                                    from the lags frame; missing -> raw 0.0 (as in training)

    Rows are emitted for the FULL symbol slate every step so the RNN hidden-state and
    xsec-attention batch dims stay aligned across the whole day. Absent symbols get NaN
    raw features (-> 0 after standardization, exactly like training NaNs) and weight 0
    (zero contribution to the refit loss).

    ``roll_window`` defaults to the production 1000 and exists so
    ``replay_check.py --rolling-window`` can exercise a fast validation gate
    (it must match the window suffix baked into ``feature_cols``).
    """

    def __init__(self, feature_cols, n_slots=N_SLOTS0, roll_window=ROLL_WINDOW):
        self.cols = list(feature_cols)
        self.K = len(self.cols)
        self.roll_window = int(roll_window)
        pos = {c: i for i, c in enumerate(self.cols)}
        self.corr = list(COLS_FEATURES_CORR)
        # NB: fullmatch — engineered names ('feature_06_diff_rolling_avg_1000',
        # 'feature_time_id', ...) also start with 'feature_'.
        self.raw_cols = [c for c in self.cols if re.fullmatch(r'feature_\d{2}', c)]
        self.i_raw = np.array([pos[c] for c in self.raw_cols])
        self.i_diff = np.array(
            [pos[f'{c}_diff_rolling_avg_{self.roll_window}'] for c in self.corr])
        self.i_std = np.array(
            [pos[f'{c}_rolling_std_{self.roll_window}'] for c in self.corr])
        self.i_avg = np.array([pos[f'{c}_avg_per_date_time'] for c in self.corr])
        self.i_tid = pos['feature_time_id']
        self.lag_rs = [r for r in range(9) if f'responder_{r}_lag1d' in pos]
        self.i_lag = np.array([pos[f'responder_{r}_lag1d'] for r in self.lag_rs])
        covered = len(self.raw_cols) + 3 * len(self.corr) + 1 + len(self.lag_rs)
        assert covered == self.K, f'unrecognized feature columns ({covered} != {self.K})'

        self.n_slots = n_slots
        self._alloc(n_slots)
        self.lag_today = np.zeros((N_TIMES, n_slots, len(self.lag_rs)), np.float32)
        self.day_store = []       # one dict per (date, time) batch — tomorrow's refit block

    # -- ring buffers --------------------------------------------------------
    def _alloc(self, n):
        c = len(self.corr)
        self.buf = np.full((n, c, self.roll_window), np.nan, np.float32)
        self.head = np.zeros(n, np.int64)      # next write position in the ring
        self.count = np.zeros(n, np.int64)     # rows seen, capped at roll_window
        self.pushes = np.zeros(n, np.int64)
        self.rsum = np.zeros((n, c), np.float64)
        self.rsumsq = np.zeros((n, c), np.float64)
        self.nancnt = np.zeros((n, c), np.int64)
        self.active = np.zeros(n, np.bool_)    # appeared today (per-date; see new_date)

    def _grow(self, n_new):
        old = self.n_slots
        for nm in ('buf', 'head', 'count', 'pushes', 'rsum', 'rsumsq', 'nancnt',
                   'active'):
            arr = getattr(self, nm)
            fresh = (np.full((n_new,) + arr.shape[1:], np.nan, arr.dtype) if nm == 'buf'
                     else np.zeros((n_new,) + arr.shape[1:], arr.dtype))
            fresh[:old] = arr
            setattr(self, nm, fresh)
        lag = np.zeros((self.lag_today.shape[0], n_new, self.lag_today.shape[2]), np.float32)
        lag[:, :old] = self.lag_today
        self.lag_today = lag
        self.n_slots = n_new

    # -- warmup from the train tail (the frozen test continues the timeline) --
    def warmup(self, comp_dir):
        tp = Path(comp_dir) / 'train.parquet'
        if not tp.exists():
            print('[featurestate] no train.parquet — rolling windows start cold')
            return
        src = str(tp / '**' / '*.parquet') if tp.is_dir() else str(tp)
        lf = pl.scan_parquet(src)
        dmax = lf.select(pl.col('date_id').max()).collect().item()
        need = self.roll_window // N_TIMES + 2   # dates giving > roll_window rows per symbol
        df = (
            lf.filter(pl.col('date_id') > dmax - need)
            .select(['date_id', 'time_id', 'symbol_id', *self.corr])
            .collect()
            .sort(['symbol_id', 'date_id', 'time_id'])
        )
        syms = df.get_column('symbol_id').to_numpy().astype(np.int64)
        if syms.size == 0:
            return
        vals = df.select(self.corr).to_numpy().astype(np.float32)
        if int(syms.max()) >= self.n_slots:
            self._grow(int(syms.max()) + 1)
        for sym in np.unique(syms):
            v = vals[syms == sym][-self.roll_window:]
            m = v.shape[0]
            self.buf[sym, :, :m] = v.T
            self.head[sym] = m % self.roll_window
            self.count[sym] = m
            v64 = v.astype(np.float64)
            nanm = np.isnan(v64)
            self.rsum[sym] = np.where(nanm, 0.0, v64).sum(axis=0)
            self.rsumsq[sym] = np.where(nanm, 0.0, v64 * v64).sum(axis=0)
            self.nancnt[sym] = nanm.sum(axis=0)
        print(f'[featurestate] rolling windows warmed from train dates > {dmax - need}')

    # -- per-day / per-batch updates -----------------------------------------
    def new_date(self, lags):
        """Install today's lagged-responder lookup (missing -> raw 0.0, as in training)."""
        self.day_store = []
        self.active[:] = False                 # per-date active-symbol set (xsec stream)
        n_lag = len(self.lag_rs)
        if lags is None or lags.height == 0:
            self.lag_today = np.zeros((N_TIMES, self.n_slots, n_lag), np.float32)
            return
        cols = [f'responder_{r}_lag_1' for r in self.lag_rs]
        if cols[0] not in lags.columns:        # tolerate unlagged column naming
            cols = [f'responder_{r}' for r in self.lag_rs]
        t = lags.get_column('time_id').to_numpy().astype(np.int64)
        s = lags.get_column('symbol_id').to_numpy().astype(np.int64)
        vals = np.nan_to_num(lags.select(cols).to_numpy().astype(np.float32), nan=0.0)
        if int(s.max()) >= self.n_slots:
            self._grow(int(s.max()) + 1)
        lut = np.zeros((max(N_TIMES, int(t.max()) + 1), self.n_slots, n_lag), np.float32)
        lut[t, s] = vals
        self.lag_today = lut

    def push(self, test):
        """Consume one (date_id, time_id) batch -> (X_slate, w_slate) in raw feature space."""
        syms = test.get_column('symbol_id').to_numpy().astype(np.int64)
        tid = int(test.get_column('time_id')[0])
        w = test.get_column('weight').to_numpy().astype(np.float32)
        raw = test.select(self.raw_cols).to_numpy().astype(np.float32)
        corr = test.select(self.corr).to_numpy().astype(np.float32)
        if int(syms.max()) >= self.n_slots:
            self._grow(int(syms.max()) + 1)
        n = self.n_slots
        self.active[syms] = True

        X = np.full((n, self.K), np.nan, np.float32)
        X[np.ix_(syms, self.i_raw)] = raw
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                  # all-NaN batch columns
            X[:, self.i_avg] = np.nanmean(corr, axis=0)      # market avg per (date, time)
        self._roll_update(syms, corr)
        mean, std = self._roll_read()
        X[np.ix_(syms, self.i_diff)] = corr - mean[syms]
        X[np.ix_(syms, self.i_std)] = std[syms]
        X[:, self.i_tid] = float(tid)
        X[:, self.i_lag] = (self.lag_today[tid]
                            if tid < self.lag_today.shape[0] else 0.0)
        w_slate = np.zeros(n, np.float32)
        w_slate[syms] = w
        present = np.zeros(n, np.bool_)        # real rows — the scaler-refit mask
        present[syms] = True
        self.day_store.append({'X': X, 'w': w_slate, 'tid': tid, 'present': present})
        return X, w_slate

    # -- rolling mean / std via ring buffer + running sums --------------------
    def _roll_update(self, syms, vals):
        isnan = np.isnan(vals)
        v0 = np.where(isnan, 0.0, vals).astype(np.float64)
        h = self.head[syms]
        old = self.buf[syms, :, h]                           # (S, 16) evicted values
        old_nan = np.isnan(old)
        full = (self.count[syms] >= self.roll_window)[:, None]
        old0 = np.where(old_nan, 0.0, old).astype(np.float64)
        self.rsum[syms] += v0 - np.where(full, old0, 0.0)
        self.rsumsq[syms] += v0 * v0 - np.where(full, old0 * old0, 0.0)
        self.nancnt[syms] += isnan.astype(np.int64) - np.where(full, old_nan, False).astype(np.int64)
        self.buf[syms, :, h] = vals
        self.head[syms] = (h + 1) % self.roll_window
        self.count[syms] = np.minimum(self.count[syms] + 1, self.roll_window)
        self.pushes[syms] += 1
        redo = syms[self.pushes[syms] % N_TIMES == 0]        # exact recompute ~once a day
        if redo.size:
            b = self.buf[redo].astype(np.float64)
            nanm = np.isnan(b)
            self.rsum[redo] = np.where(nanm, 0.0, b).sum(axis=2)
            self.rsumsq[redo] = np.where(nanm, 0.0, b * b).sum(axis=2)
            sentinel = (self.roll_window - np.minimum(self.count[redo], self.roll_window))[:, None]
            self.nancnt[redo] = nanm.sum(axis=2) - sentinel

    def _roll_read(self):
        ok = (self.count[:, None] >= self.roll_window) & (self.nancnt == 0)
        mean = np.where(ok, self.rsum / self.roll_window, np.nan)
        var = (self.rsumsq - self.rsum * self.rsum / self.roll_window) / (self.roll_window - 1)
        std = np.where(ok, np.sqrt(np.maximum(var, 0.0)), np.nan)
        return mean.astype(np.float32), std.astype(np.float32)


def _transform_np(pre, X, refit=False, refit_mask=None):
    """Preprocessor.transform for a numpy block (it expects a polars frame).

    X's columns are already in ``pre.feature_cols`` order (asserted by the
    kernel): clip to train quantiles -> optional Welford partial_fit ->
    z-score (scaler.transform ends with nan_to_num(0), so NaN slate rows
    become 0).

    ``refit_mask`` (bool, per row) restricts the ``partial_fit`` statistics to
    real rows — pad/absent slate rows carry batch-level values (market avg,
    time_id, zero lags) that would otherwise contaminate the Welford stats,
    which offline only ever see present rows. The transform itself is still
    applied to every row.
    """
    Xc = pre.clipper.transform(X.astype(np.float32, copy=False))
    if refit:
        Xf = Xc if refit_mask is None else Xc[refit_mask]
        if Xf.shape[0]:
            pre.scaler.partial_fit(Xf)
    return pre.scaler.transform(Xc)
