"""XGB ablation: does exposing symbol IDENTITY help?

The models currently never see *which* symbol they predict — only rolling,
per-symbol-normalized features — so they can't learn a per-symbol effect. Two
findings motivate testing that gap (see ``cluster_symbols.py``): symbol
*co-movement* clusters are temporally unstable (useless for routing), but
symbol *feature-profile* clusters are stable (ARI ~0.46) — a symbol's feature
signature persists. So identity carries durable signal.

This isolates the marginal value of identity on top of our current best
feature set (base = all memmap columns, incl. lagged responders):

  1. base                    — current features, no identity
  2. base + symbol one-hot   — finest identity (one column per symbol)
  3. base + cluster one-hot  — coarse identity via feature-profile clusters
       (clusters fit on TRAIN rows only — a symbol->cluster map keyed on the
        stable feature signature; no leakage, it's an identity map)

Weighted-R² on the validation tail (competition metric) per variant.

Usage
-----

    uv run python scripts/ablate_symbol_identity.py \\
        --data artifacts/precomputed/pool700_lags \\
        --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \\
        --k 6 --out artifacts/bench/ablation_symbol.json
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def r2_weighted(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest: dict, lo: int, hi: int) -> np.ndarray:
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    return np.concatenate(parts)


def feature_profile_clusters(
    X: np.ndarray, sym: np.ndarray, n_base: int, k: int,
) -> dict[int, int]:
    """Cluster symbols by their standardized mean base-feature vector (train only)."""
    uniq = np.unique(sym)
    prof = np.vstack([np.nanmean(X[sym == s, :n_base], axis=0) for s in uniq])
    prof = np.nan_to_num(prof)
    prof = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)
    lab = fcluster(linkage(pdist(prof), method="ward"), k, criterion="maxclust")
    return {int(s): int(la) for s, la in zip(uniq, lab, strict=True)}


def onehot(sym: np.ndarray, keys: list[int]) -> np.ndarray:
    idx = {kk: i for i, kk in enumerate(keys)}
    out = np.zeros((len(sym), len(keys)), dtype=np.float32)
    for row, s in enumerate(sym):
        j = idx.get(int(s))
        if j is not None:
            out[row, j] = 1.0
    return out


def fit_eval(Xtr, ytr, wtr, Xva, yva, wva, n_estimators, seed) -> tuple[float, int]:
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
    p.add_argument("--k", type=int, default=6, help="feature-profile clusters")
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    K = manifest["K"]
    n_base = K - len(manifest.get("lag_cols", []))
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")
    sym = np.load(data / "symbols.i16.npy", mmap_mode="r")

    tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    Xtr = np.ascontiguousarray(X_mm[tr]).astype(np.float32)
    Xva = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    ytr, wtr, symtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr]), np.ascontiguousarray(sym[tr])
    yva, wva, symva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va]), np.ascontiguousarray(sym[va])
    print(f"K={K} (base {n_base}+lags {K - n_base})  train={len(tr):,}  valid={len(va):,}", flush=True)

    # Feature-profile clusters fit on TRAIN rows only, then apply to valid.
    cmap = feature_profile_clusters(Xtr, symtr, n_base, args.k)
    all_syms = sorted(set(cmap))
    clusters = sorted(set(cmap.values()))
    print(f"feature-profile k={args.k}: sizes="
          f"{[sum(1 for v in cmap.values() if v == c) for c in clusters]}", flush=True)

    sym_keys = all_syms
    def build(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "base":
            return Xtr, Xva
        if name == "base+symbol":
            return (np.hstack([Xtr, onehot(symtr, sym_keys)]),
                    np.hstack([Xva, onehot(symva, sym_keys)]))
        cl_tr = np.array([cmap.get(int(s), -1) for s in symtr])
        cl_va = np.array([cmap.get(int(s), -1) for s in symva])
        return (np.hstack([Xtr, onehot(cl_tr, clusters)]),
                np.hstack([Xva, onehot(cl_va, clusters)]))

    results = []
    for name in ("base", "base+symbol", "base+cluster"):
        xtr, xva = build(name)
        r2, best_it = fit_eval(xtr, ytr, wtr, xva, yva, wva, args.n_estimators, args.seed)
        print(f"  {name:14s} n_features={xtr.shape[1]:>3}  best_iter={best_it:>4}  R²={r2:+.5f}", flush=True)
        results.append({"variant": name, "n_features": int(xtr.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        if name != "base":
            del xtr, xva
            gc.collect()

    base_r2 = results[0]["r2_weighted"]
    for r in results:
        r["delta_vs_base"] = r["r2_weighted"] - base_r2

    out = {"data": str(data), "k": args.k,
           "train": [args.train_lo, args.train_hi], "valid": [args.valid_lo, args.valid_hi],
           "cluster_map": {str(s): cmap[s] for s in all_syms},
           "results": results, "timestamp": datetime.now().isoformat()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nΔ vs base:  +symbol {results[1]['delta_vs_base']:+.5f}   "
          f"+cluster {results[2]['delta_vs_base']:+.5f}", flush=True)
    print(f"written → {args.out}", flush=True)


if __name__ == "__main__":
    main()
