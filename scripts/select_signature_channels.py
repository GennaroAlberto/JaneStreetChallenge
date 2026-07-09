"""Pick the subset of raw features that should feed the Volterra signature.

Why this exists
---------------

The signature compresses the *path* of a small set of channels into a
finite set of monomials. Two requirements for it to add signal:

1. The channels should be **raw** (carry the full high-frequency path).
   Feeding pre-smoothed features (rolling-mean differences, rolling
   std, market-average, any EMA) wastes the signature: most of what it
   would have extracted has already been baked in.

2. The channels should be **temporally rich and non-redundant** with each
   other. A signature of d channels has d + d² + d³ + … monomials, so
   adding a channel that's a near-duplicate of an existing one just
   inflates dimensionality without adding signal.

How we score
------------

For every raw ``feature_NN`` column (raw = not a derived
``*_diff_rolling_avg`` / ``*_rolling_std`` / ``*_avg_per_date_time`` /
``*_ewma`` / ``*_rank`` / ``feature_time_id`` column, and not in the
project's hand-flagged categorical list):

* ``target_abs_corr`` — |Pearson(feature, responder_6)|. Predictiveness.
* ``returns_var``    — variance of first-differences per symbol.
  Higher = richer path = a signature has more to compress.
* ``autocorr_lag1``  — lag-1 autocorr of first-differences. A signature
  shines on processes with *some* persistence (mean-reverting trends,
  not pure white noise).

We combine these into a composite score that prefers predictive,
temporally rich features:

    score = z(target_abs_corr) + 0.5 * z(returns_var) + 0.25 * abs(autocorr_lag1)

(z = standardize across surviving raw features.)

Selection: greedy with a redundancy gate
----------------------------------------

1. Sort candidates by score (descending).
2. Pick the top one.
3. For each candidate after that, accept it iff its |corr| with every
   already-selected channel is below ``--redundancy-cap`` (default 0.6).
4. Repeat until K channels are picked (default 6).

We also expose the ranked table + correlation matrix so you can sanity-check.

Usage
-----

    uv run python scripts/select_signature_channels.py \\
        --min-date 1499 --max-date 1598 --valid 0 \\
        --k 6 --out artifacts/sig_channels.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COL_DATE, COL_ID, COLS_FEATURES_CAT, Cfg
from janestreet.data.features import FeatureBuilder
from janestreet.pipeline import prepare_dataset

# Columns we know are derived from a feature (smoothed / cross-sectional /
# EMA-ish) — exclude them from the signature input pool.
DERIVED_PATTERNS = (
    "_diff_rolling_avg_", "_rolling_std_", "_avg_per_date_time",
    "_ewma_", "_rank_per_time",
)


def is_raw_feature(col: str) -> bool:
    """Predicate: is this column a *raw* feature_NN suitable for signature input?"""
    if col in COLS_FEATURES_CAT:
        return False
    if col == "feature_time_id":
        return False
    if any(p in col for p in DERIVED_PATTERNS):
        return False
    # We accept feature_00..feature_78 but not their derivatives.
    return bool(re.fullmatch(r"feature_\d{2}", col))


def per_symbol_returns_var(df: pl.DataFrame, col: str) -> float:
    """Mean across symbols of the within-symbol first-difference variance."""
    diff = (
        df.lazy()
        .sort([COL_ID, COL_DATE, "time_id"])
        .with_columns((pl.col(col) - pl.col(col).shift(1).over(COL_ID)).alias("_d"))
        .group_by(COL_ID)
        .agg(pl.col("_d").var().alias("_v"))
        .select(pl.col("_v").mean())
        .collect()
        .item()
    )
    return float(diff) if diff is not None and not np.isnan(diff) else 0.0


def per_symbol_autocorr_lag1(df: pl.DataFrame, col: str) -> float:
    """Mean across symbols of the lag-1 autocorrelation of first-differences."""
    res = (
        df.lazy()
        .sort([COL_ID, COL_DATE, "time_id"])
        .with_columns((pl.col(col) - pl.col(col).shift(1).over(COL_ID)).alias("_d"))
        .with_columns(pl.col("_d").shift(1).over(COL_ID).alias("_d_lag1"))
        .group_by(COL_ID)
        .agg(pl.corr("_d", "_d_lag1").alias("_rho"))
        .select(pl.col("_rho").mean())
        .collect()
        .item()
    )
    return float(res) if res is not None and not np.isnan(res) else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-date", type=int, default=1499)
    p.add_argument("--max-date", type=int, default=1598)
    p.add_argument(
        "--valid", type=int, default=0,
        help="reserved for symmetry with bench; unused here",
    )
    p.add_argument("--k", type=int, default=6, help="number of channels to pick")
    p.add_argument(
        "--redundancy-cap", type=float, default=0.6,
        help="reject a candidate whose |corr| with any selected channel exceeds this",
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
    print(f"data: {df.height:,} rows over {args.min_date}..{args.max_date}", flush=True)
    print(f"features: {len(feat_cols)} total, {len(raw)} raw candidates", flush=True)

    target = df.select(args.target).to_series().to_numpy().astype(np.float64)
    target_mask = np.isfinite(target)
    target = target[target_mask]
    target_centered = target - target.mean()
    target_norm = np.sqrt((target_centered ** 2).sum()) + 1e-12

    # ---- Score each raw feature ----
    print("\nScoring raw features:")
    rows = []
    for col in raw:
        x = df.select(col).to_series().to_numpy().astype(np.float64)
        x = x[target_mask]
        finite = np.isfinite(x)
        if finite.sum() < 100:
            print(f"  {col}: skipped — too few finite values"); continue
        x_c = x[finite] - np.nanmean(x[finite])
        t_c = target_centered[finite]
        corr_t = float((x_c * t_c).sum() / (np.sqrt((x_c ** 2).sum()) * target_norm + 1e-12))
        var_d = per_symbol_returns_var(df, col)
        ac_d = per_symbol_autocorr_lag1(df, col)
        rows.append(dict(col=col, target_abs_corr=abs(corr_t), returns_var=var_d, autocorr_lag1=ac_d))

    if not rows:
        raise SystemExit("no scorable raw features")

    tac = np.array([r["target_abs_corr"] for r in rows])
    rvr = np.array([r["returns_var"] for r in rows])
    aco = np.array([r["autocorr_lag1"] for r in rows])

    def z(a: np.ndarray) -> np.ndarray:
        s = a.std()
        return (a - a.mean()) / (s + 1e-12)

    composite = z(tac) + 0.5 * z(np.log1p(rvr)) + 0.25 * np.abs(aco)
    for r, s in zip(rows, composite, strict=True):
        r["composite"] = float(s)
    rows.sort(key=lambda r: r["composite"], reverse=True)

    print(f"\n{'rank':>4} {'col':12} {'|corr|':>8} {'ret_var':>10} {'ac_lag1':>8} {'score':>8}")
    for i, r in enumerate(rows[:20]):
        print(
            f"{i+1:>4} {r['col']:12} {r['target_abs_corr']:>8.4f} "
            f"{r['returns_var']:>10.3g} {r['autocorr_lag1']:>+8.3f} "
            f"{r['composite']:>+8.3f}"
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
            selected.append(r); selected_cols.append(col); print(f"  + {col}  (seed)")
            continue
        x = get_clean(col)
        ok = True
        worst = 0.0
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
            print(f"  + {col}  (max |corr| to selected = {worst:.3f})")
        else:
            print(f"  - {col}  (rejected, max |corr| {worst:.3f} > {args.redundancy_cap})")
        if len(selected) >= args.k:
            break

    # Channel indices in the final feature vector ordering
    sig_channels = [feat_cols.index(s["col"]) for s in selected]

    out = dict(
        selected=[dict(col=s["col"], channel_idx=feat_cols.index(s["col"]),
                       target_abs_corr=s["target_abs_corr"],
                       returns_var=s["returns_var"], autocorr_lag1=s["autocorr_lag1"],
                       composite=s["composite"]) for s in selected],
        signature_channels=sig_channels,
        all_scored=rows,
        config=dict(
            min_date=args.min_date, max_date=args.max_date,
            k=args.k, redundancy_cap=args.redundancy_cap,
            target=args.target,
        ),
        timestamp=datetime.now().isoformat(),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nfinal signature_channels = {sig_channels}")
    print(f"written → {args.out}")


if __name__ == "__main__":
    main()
