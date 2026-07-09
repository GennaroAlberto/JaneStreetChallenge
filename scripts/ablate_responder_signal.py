"""XGB ablation: do lagged responders / responder-signal features help?

Reads the ``pool700_lags`` memmap directly (no prepare_dataset, no torch — so
no libomp conflict and a small memory footprint) and fits XGB on three
feature sets to isolate the marginal value of each block:

  1. base            — the 125 engineered features (no lagged responders)
  2. base + lags     — + the 9 previous-day responder columns
  3. base + lags + signal — + the 5 responder-signal features
       (multi-scale momentum + venue spread, computed here as differences of
        the lag columns — exactly what FeatureBuilder._add_responder_signal
        produces, up to the frozen standardization which is irrelevant to a
        scale-invariant tree model)

Weighted-R² on the validation tail (competition metric) is reported per
variant. Same XGB hyperparameters and early-stopping across variants.

Usage
-----

    uv run python scripts/ablate_responder_signal.py \\
        --data artifacts/precomputed/pool700_lags \\
        --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \\
        --out artifacts/bench/ablation_rsig.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb


def r2_weighted(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest: dict, lo: int, hi: int) -> np.ndarray:
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    return np.concatenate(parts)


def signal_features(lag_block: np.ndarray) -> np.ndarray:
    """Compute the 5 responder-signal features from the 9-col lag block.

    lag_block columns are responder_0..8 _lag1d in order. Mapping (Kaggle
    #555562): r6=SMA20/A, r7=SMA120/A, r8=SMA4/A, r3=SMA20/B, r4=SMA120/B,
    r5=SMA4/B. Offsets within the block: r{k} at column k.
    """
    r = {k: lag_block[:, k] for k in range(9)}
    return np.column_stack([
        r[8] - r[6],   # rsig_momA_short  (SMA4 - SMA20, exch A)
        r[6] - r[7],   # rsig_momA_med    (SMA20 - SMA120, exch A)
        r[5] - r[3],   # rsig_momB_short  (exch B)
        r[3] - r[4],   # rsig_momB_med    (exch B)
        r[8] - r[5],   # rsig_spread_fine (A - B at SMA4)
    ]).astype(np.float32)


def fit_eval(
    Xtr: np.ndarray, ytr: np.ndarray, wtr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray, wva: np.ndarray,
    n_estimators: int, seed: int,
) -> tuple[float, int]:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    K = manifest["K"]
    n_base = K - len(manifest.get("lag_cols", []))   # base = everything before the lags
    print(f"K={K}  base cols=0..{n_base - 1}  lag cols={n_base}..{K - 1}", flush=True)
    assert n_base == 125, f"expected 125 base cols, got {n_base}"

    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    print(f"train rows={len(tr):,} ({args.train_lo}..{args.train_hi})  "
          f"valid rows={len(va):,} ({args.valid_lo}..{args.valid_hi})", flush=True)

    # Materialise once (float32); slice/derive per variant.
    Xtr_all = np.ascontiguousarray(X_mm[tr]).astype(np.float32)
    Xva_all = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    ytr, wtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr])
    yva, wva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va])

    sig_tr = signal_features(Xtr_all[:, n_base:K])
    sig_va = signal_features(Xva_all[:, n_base:K])

    def build(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "base":
            return (np.ascontiguousarray(Xtr_all[:, :n_base]),
                    np.ascontiguousarray(Xva_all[:, :n_base]))
        if name == "base+lags":
            return (np.ascontiguousarray(Xtr_all[:, :K]),
                    np.ascontiguousarray(Xva_all[:, :K]))
        return (np.hstack([Xtr_all[:, :K], sig_tr]),
                np.hstack([Xva_all[:, :K], sig_va]))

    # Build/fit/free one variant at a time so we never hold all three
    # feature matrices in memory simultaneously (16 GB Mac).
    results = []
    for name in ("base", "base+lags", "base+lags+signal"):
        xtr, xva = build(name)
        r2, best_it = fit_eval(xtr, ytr, wtr, xva, yva, wva, args.n_estimators, args.seed)
        print(f"  {name:22s} n_features={xtr.shape[1]:>3}  best_iter={best_it:>4}  R²={r2:+.5f}", flush=True)
        results.append({"variant": name, "n_features": int(xtr.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        del xtr, xva
        gc.collect()

    base_r2 = results[0]["r2_weighted"]
    for r in results:
        r["delta_vs_base"] = r["r2_weighted"] - base_r2

    out = {
        "data": str(data),
        "train": [args.train_lo, args.train_hi], "valid": [args.valid_lo, args.valid_hi],
        "n_estimators": args.n_estimators, "seed": args.seed,
        "results": results, "timestamp": datetime.now().isoformat(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nΔ vs base:  +lags {results[1]['delta_vs_base']:+.5f}   "
          f"+lags+signal {results[2]['delta_vs_base']:+.5f}", flush=True)
    print(f"written → {args.out}", flush=True)


if __name__ == "__main__":
    main()
