"""XGB ablation: innovation features from the pre-smoothed feature trio.

The feature atlas (docs/FEATURE_RESEARCH.md) found feature_12/67/70 carry a
triangular ACF ramp with cutoff ~160 (ramp linearity R² ≈ 0.98): they are
themselves ~SMA-160 of some underlying series. Their recent increments —
plausibly the informative part — are attenuated ~160x relative to the stale
level. This ablation adds short-lag differences (the cheap anti-smoothing
remedy; see the smoothing chapter) and measures the marginal value:

  base                — the pool's 134 columns
  base + innovations  — + x_t − x_{t−δ} for c ∈ {12, 67, 70}, δ ∈ {1, 8, 40}

Diffs are computed within (symbol, day) blocks (strictly trailing; first δ
rows of each block zero-filled). Memmap layout: date-major, symbol-major
within date, 968 contiguous times per (date, symbol).

Usage
-----
    uv run python scripts/ablate_innovations.py \\
        --data artifacts/precomputed/pool700_lags \\
        --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \\
        --out artifacts/bench/ablation_innov.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

TRIO = ["feature_12", "feature_67", "feature_70"]
DELTAS = [1, 8, 40]
N_TIMES = 968


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi):
    ranges = manifest["date_row_ranges"]
    dates = [d for d in range(lo, hi + 1) if str(d) in ranges]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64) for d in dates]
    return dates, parts


def innovation_features(X_mm, manifest, dates, trio_idx):
    """Trailing diffs per (date, symbol) block for the trio columns."""
    ranges = manifest["date_row_ranges"]
    out_parts = []
    for d in dates:
        r0, r1 = ranges[str(d)]
        n = r1 - r0
        if n % N_TIMES:
            raise ValueError(f"date {d}: {n} rows not divisible by {N_TIMES}")
        block = np.ascontiguousarray(X_mm[r0:r1][:, trio_idx]).astype(np.float32)
        b = block.reshape(n // N_TIMES, N_TIMES, len(trio_idx))
        cols = []
        for delta in DELTAS:
            diff = np.zeros_like(b)
            diff[:, delta:, :] = b[:, delta:, :] - b[:, :-delta, :]
            cols.append(diff.reshape(n, len(trio_idx)))
        out_parts.append(np.concatenate(cols, axis=1))   # (n, 9)
    return np.concatenate(out_parts, axis=0)


def fit_eval(Xtr, ytr, wtr, Xva, yva, wva, n_estimators, seed):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", max_bin=128, n_jobs=1,
        early_stopping_rounds=50, random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(Xtr, ytr, sample_weight=wtr,
              eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
    pred = model.predict(Xva)
    best_it = int(getattr(model, "best_iteration", n_estimators) or n_estimators)
    return r2_weighted(yva, pred, wva), best_it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/ablation_innov.json")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["feature_cols"]
    K = manifest["K"]
    trio_idx = [names.index(c) for c in TRIO]
    print(f"trio at indices {trio_idx}; deltas {DELTAS}", flush=True)

    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    tr_dates, tr_parts = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va_parts = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    tr = np.concatenate(tr_parts); va = np.concatenate(va_parts)
    print(f"train rows={len(tr):,}  valid rows={len(va):,}", flush=True)

    innov_tr = innovation_features(X_mm, manifest, tr_dates, trio_idx)
    innov_va = innovation_features(X_mm, manifest, va_dates, trio_idx)
    Xtr = np.ascontiguousarray(X_mm[tr]).astype(np.float32)
    Xva = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    ytr, wtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr])
    yva, wva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va])

    results = []
    for name, xt, xv in (
        ("base", Xtr, Xva),
        ("base+innovations", np.hstack([Xtr, innov_tr]), np.hstack([Xva, innov_va])),
    ):
        r2, best_it = fit_eval(xt, ytr, wtr, xv, yva, wva, args.n_estimators, args.seed)
        print(f"  {name:18s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
              f"R²={r2:+.5f}", flush=True)
        results.append({"variant": name, "n_features": int(xt.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        if name != "base":
            del xt, xv
        gc.collect()

    results[1]["delta_vs_base"] = results[1]["r2_weighted"] - results[0]["r2_weighted"]
    Path(args.out).write_text(json.dumps({
        "data": str(data), "trio": TRIO, "deltas": DELTAS,
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"delta vs base: {results[1]['delta_vs_base']:+.5f} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
