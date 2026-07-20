"""XGB ablation: market/idio decomposition of the reversal block.

The atlas found the reversal features (39..60 family) carry strong negative
nowcast correlation with the realized signal and modest positive alpha.
Their cross-sectional market share is ~0.3 — i.e. each is a blend of a
market-wide reversal component and an idiosyncratic one, which plausibly
carry different predictive timing. This ablation splits the strongest
reversal columns into per-(date,time) cross-symbol mean (market) and
residual (idio) and measures the marginal value of the explicit split:

  base           — the pool's 134 columns
  base + decomp  — + market mean and idio residual for each of
                   {45, 46, 47, 56, 57, 58, 60}  (14 new columns)

Both parts are contemporaneous (cross-sectional at the same timestamp) —
legal under the streaming protocol. Memmap layout: date-major, then
symbol-major, 968 contiguous times per (date, symbol) block.

Usage
-----
    uv run python scripts/ablate_reversal_decomp.py \\
        --data artifacts/precomputed/pool700_lags \\
        --out artifacts/bench/ablation_revdecomp.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

BLOCK = ["feature_45", "feature_46", "feature_47",
         "feature_56", "feature_57", "feature_58", "feature_60"]
N_TIMES = 968


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi):
    ranges = manifest["date_row_ranges"]
    dates = [d for d in range(lo, hi + 1) if str(d) in ranges]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64) for d in dates]
    return dates, np.concatenate(parts)


def decomp_features(X_mm, manifest, dates, block_idx):
    """Per (date,time) cross-symbol mean + residual for the block columns."""
    ranges = manifest["date_row_ranges"]
    parts = []
    for d in dates:
        r0, r1 = ranges[str(d)]
        n = r1 - r0
        if n % N_TIMES:
            raise ValueError(f"date {d}: {n} rows not divisible by {N_TIMES}")
        s = n // N_TIMES
        b = np.ascontiguousarray(X_mm[r0:r1][:, block_idx]).astype(np.float32)
        b = b.reshape(s, N_TIMES, len(block_idx))       # (sym, time, col)
        mkt = b.mean(axis=0, keepdims=True)             # (1, time, col)
        idio = b - mkt
        mkt_rows = np.broadcast_to(mkt, b.shape).reshape(n, len(block_idx))
        parts.append(np.concatenate([mkt_rows, idio.reshape(n, len(block_idx))], axis=1))
    return np.concatenate(parts, axis=0)                # (N, 2*len(block))


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
    p.add_argument("--out", default="artifacts/bench/ablation_revdecomp.json")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["feature_cols"]
    K = manifest["K"]
    block_idx = [names.index(c) for c in BLOCK]
    print(f"reversal block at indices {block_idx}", flush=True)

    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    tr_dates, tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    print(f"train rows={len(tr):,}  valid rows={len(va):,}", flush=True)

    dec_tr = decomp_features(X_mm, manifest, tr_dates, block_idx)
    dec_va = decomp_features(X_mm, manifest, va_dates, block_idx)
    Xtr = np.ascontiguousarray(X_mm[tr]).astype(np.float32)
    Xva = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    ytr, wtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr])
    yva, wva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va])

    results = []
    for name, xt, xv in (
        ("base", Xtr, Xva),
        ("base+decomp", np.hstack([Xtr, dec_tr]), np.hstack([Xva, dec_va])),
    ):
        r2, best_it = fit_eval(xt, ytr, wtr, xv, yva, wva,
                               args.n_estimators, args.seed)
        print(f"  {name:12s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
              f"R²={r2:+.5f}", flush=True)
        results.append({"variant": name, "n_features": int(xt.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        if name != "base":
            del xt, xv
        gc.collect()

    results[1]["delta_vs_base"] = (results[1]["r2_weighted"]
                                   - results[0]["r2_weighted"])
    Path(args.out).write_text(json.dumps({
        "data": str(data), "block": BLOCK,
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"delta vs base: {results[1]['delta_vs_base']:+.5f} → {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
