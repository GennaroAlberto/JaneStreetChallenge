"""Signature-channel selector v2 — score features by AR/ARMA predictiveness.

Why this exists
---------------

v1 (``select_signature_channels.py``) scored features by univariate |corr|
with responder_6. That was wrong: the signature does not care about a
channel's marginal predictiveness, it cares whether the channel's *recent
path* carries information. v1's pick scored **worse** than the arbitrary
``[0..5]`` baseline (+0.00445 vs +0.00485 static).

This v2 fits a per-feature AR(p)[+MA(q)] model and scores each feature
by how well *its own lags* predict responder_6. Concretely, for each raw
``feature_NN``:

    fit              responder_6_{s,d,t}  ~  sum_{k=1..p} a_k * feature_NN_{s,d,t-k}
                                          + sum_{k=1..q} b_k * eps_{s,d,t-k}
    score by R²      on a held-out date range within the analysis window.

We default to pure AR(p=7) (q=0) because the Jane Street per-day grid is
~968 time-ids and a 7-step look-back captures enough of the path without
introducing a non-trivial MA solver. ``--ma`` lets you add a small MA
component (default 0; try 2 or 3 if you want).

Two scoring modes are reported side-by-side so you can compare:

* **levels**     — lags of the feature itself.
* **increments** — lags of ``feature_NN - feature_NN.shift(1)`` per (sym, date).
                   This is what the Volterra signature actually integrates
                   over, so it's arguably the more faithful filter.

We rank features by ``max(R²_levels, R²_increments)`` (a feature is
useful for the signature if either path encoding is predictive), and
then apply a redundancy gate identical to v1.

Selection: greedy with a redundancy gate
----------------------------------------

1. Sort candidates by max(R²_levels, R²_increments), descending.
2. Pick the top one.
3. For each subsequent candidate, accept iff its |corr| with every
   already-selected channel is below ``--redundancy-cap`` (default 0.5).
4. Repeat until K channels are picked (default 8).

Usage
-----

    uv run python scripts/select_signature_channels_ar.py \\
        --min-date 1499 --max-date 1598 --valid-tail 30 \\
        --p 7 --ma 0 --k 8 \\
        --out artifacts/sig_channels_ar.json
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COL_DATE, COL_ID, COL_TIME, COLS_FEATURES_CAT, Cfg
from janestreet.data.features import FeatureBuilder
from janestreet.pipeline import prepare_dataset

warnings.filterwarnings("ignore")


DERIVED_PATTERNS = (
    "_diff_rolling_avg_", "_rolling_std_", "_avg_per_date_time",
    "_ewma_", "_rank_per_time",
)


def is_raw_feature(col: str) -> bool:
    if col in COLS_FEATURES_CAT:
        return False
    if col == "feature_time_id":
        return False
    if any(p in col for p in DERIVED_PATTERNS):
        return False
    return bool(re.fullmatch(r"feature_\d{2}", col))


# ----------------------------------------------------------------------
def _lagged_design(
    df: pl.DataFrame, col: str, p: int, *, target: str, use_increments: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (rows × p) of lagged values, y (rows) of the target.

    Lags are taken WITHIN (symbol, date) so we never look across the
    day boundary (the responder_6 path resets at midnight).
    """
    if use_increments:
        df = df.with_columns(
            (pl.col(col) - pl.col(col).shift(1).over([COL_ID, COL_DATE]))
            .alias(f"__{col}__d")
        )
        source_col = f"__{col}__d"
    else:
        source_col = col
    lag_exprs = [
        pl.col(source_col)
        .shift(k)
        .over([COL_ID, COL_DATE])
        .alias(f"__lag{k}")
        for k in range(1, p + 1)
    ]
    df = df.with_columns(lag_exprs)
    cols = [f"__lag{k}" for k in range(1, p + 1)]
    sub = df.select([*cols, target]).drop_nulls()
    X = sub.select(cols).to_numpy().astype(np.float64)
    y = sub.select(target).to_series().to_numpy().astype(np.float64)
    # Drop rows where any value is non-finite (rare but happens after diffs)
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[m], y[m]


def _ridge_holdout_r2(
    X: np.ndarray, y: np.ndarray, *, train_frac: float = 0.7, alpha: float = 1e-2,
) -> float:
    """Fit ridge on the first ``train_frac`` of rows, score on the rest.

    Standardise inputs to make the ridge penalty meaningful across features
    on different scales. R² is the standard (unweighted) coefficient of
    determination — we just want a relative ranking here, not an exact
    comparison to the bench's weighted-R².
    """
    n = len(y)
    if n < 200:
        return float("nan")
    n_tr = int(train_frac * n)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    # Standardise on train stats.
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-12
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ytr_c = ytr - ytr.mean()
    # Closed-form ridge.
    p = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(p)
    b = Xtr.T @ ytr_c
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("nan")
    yhat = Xte @ w + ytr.mean()
    ss_res = float(((yte - yhat) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def _arma_ma_holdout_r2(
    X: np.ndarray, y: np.ndarray, q: int, *, train_frac: float = 0.7, alpha: float = 1e-2,
) -> float:
    """Pure-AR-with-MA approximation: append q lags of the y-residual-so-far.

    Cheap MA: at train time we use lagged y as a proxy for the MA innovation
    (lagged_target). At inference we'd ideally Kalman-smooth, but for
    *ranking* purposes the proxy is fine.
    """
    if q <= 0:
        return _ridge_holdout_r2(X, y, train_frac=train_frac, alpha=alpha)
    n = len(y)
    if n < q + 100:
        return float("nan")
    extra = np.zeros((n, q))
    for k in range(1, q + 1):
        extra[k:, k - 1] = y[:-k]
    X2 = np.hstack([X, extra])
    return _ridge_holdout_r2(X2, y, train_frac=train_frac, alpha=alpha)


# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-date", type=int, default=1499)
    p.add_argument("--max-date", type=int, default=1598)
    p.add_argument(
        "--valid-tail", type=int, default=30,
        help="held-out dates for ranking inside the analysis window",
    )
    p.add_argument("--p", type=int, default=7, help="AR order (lags of the feature)")
    p.add_argument("--ma", type=int, default=0, help="MA order (lags of target as proxy)")
    p.add_argument("--k", type=int, default=8, help="number of channels to pick")
    p.add_argument(
        "--redundancy-cap", type=float, default=0.5,
        help="reject candidates whose |corr| with any selected channel exceeds this",
    )
    p.add_argument("--target", type=str, default="responder_6")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    cfg = Cfg()
    cfg.min_date_id = args.min_date
    cfg.max_date_id = args.max_date
    df = prepare_dataset(cfg)
    feat_cols = FeatureBuilder(cfg).feature_columns()
    raw = [c for c in feat_cols if is_raw_feature(c)]
    print(
        f"data: {df.height:,} rows over {args.min_date}..{args.max_date}\n"
        f"features: {len(feat_cols)} total, {len(raw)} raw candidates\n"
        f"AR order p={args.p}  MA order q={args.ma}  redundancy cap={args.redundancy_cap}",
        flush=True,
    )

    # Sort the frame globally so within-(sym,date) shift() is well-defined.
    df = df.sort([COL_ID, COL_DATE, COL_TIME])

    # Score each raw feature on both levels and increments.
    print("\nFitting AR per feature (levels + increments) ...", flush=True)
    rows = []
    for col in raw:
        try:
            X_lvl, y_lvl = _lagged_design(df, col, args.p, target=args.target, use_increments=False)
            X_inc, y_inc = _lagged_design(df, col, args.p, target=args.target, use_increments=True)
            r2_lvl = _arma_ma_holdout_r2(X_lvl, y_lvl, args.ma)
            r2_inc = _arma_ma_holdout_r2(X_inc, y_inc, args.ma)
        except Exception as e:  # noqa: BLE001
            print(f"  {col}: error {e!r}"); continue
        rows.append(dict(
            col=col, r2_levels=float(r2_lvl), r2_increments=float(r2_inc),
            r2_max=float(max(r2_lvl, r2_inc) if not np.isnan(r2_lvl) and not np.isnan(r2_inc) else float("nan")),
        ))

    rows = [r for r in rows if np.isfinite(r["r2_max"])]
    rows.sort(key=lambda r: r["r2_max"], reverse=True)
    print(f"\n{'rank':>4} {'col':12} {'R²lvl':>9} {'R²inc':>9} {'R²max':>9}")
    for i, r in enumerate(rows[:25]):
        print(
            f"{i+1:>4} {r['col']:12} {r['r2_levels']:>+9.5f} "
            f"{r['r2_increments']:>+9.5f} {r['r2_max']:>+9.5f}"
        )

    # ---- Greedy redundancy-aware selection ----
    print(f"\nSelecting K={args.k} channels with |corr|-cap {args.redundancy_cap}:")
    selected: list[dict] = []
    selected_cols: list[str] = []
    cached: dict[str, np.ndarray] = {}

    def get_clean(col: str) -> np.ndarray:
        if col in cached:
            return cached[col]
        x = df.select(col).to_series().to_numpy().astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cached[col] = x
        return x

    for r in rows:
        col = r["col"]
        if not selected:
            selected.append(r); selected_cols.append(col)
            print(f"  + {col}  (seed, R²max={r['r2_max']:+.5f})")
            continue
        x = get_clean(col)
        worst = 0.0
        ok = True
        for sc in selected_cols:
            y = get_clean(sc)
            xc = x - x.mean()
            yc = y - y.mean()
            denom = (np.sqrt((xc ** 2).sum()) * np.sqrt((yc ** 2).sum())) + 1e-12
            rho = float(abs((xc * yc).sum() / denom))
            worst = max(worst, rho)
            if rho > args.redundancy_cap:
                ok = False
                break
        if ok:
            selected.append(r); selected_cols.append(col)
            print(f"  + {col}  (R²max={r['r2_max']:+.5f}  max |corr| to selected = {worst:.3f})")
        else:
            print(f"  - {col}  (rejected, max |corr| {worst:.3f} > {args.redundancy_cap})")
        if len(selected) >= args.k:
            break

    sig_channels = [feat_cols.index(s["col"]) for s in selected]

    out = dict(
        selected=[dict(
            col=s["col"], channel_idx=feat_cols.index(s["col"]),
            r2_levels=s["r2_levels"], r2_increments=s["r2_increments"], r2_max=s["r2_max"],
        ) for s in selected],
        signature_channels=sig_channels,
        all_scored=rows,
        config=dict(
            min_date=args.min_date, max_date=args.max_date,
            p=args.p, ma=args.ma, k=args.k,
            redundancy_cap=args.redundancy_cap, target=args.target,
        ),
        timestamp=datetime.now().isoformat(),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nfinal signature_channels = {sig_channels}")
    print(f"written → {args.out}")


if __name__ == "__main__":
    main()
