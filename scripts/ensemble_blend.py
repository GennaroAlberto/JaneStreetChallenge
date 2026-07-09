"""Blend predictions from multiple model runs using their .npz dumps.

The bench writes ``<out>/preds/<model>_<static|online>.npz`` for every
model it evaluates. This script loads N such files, evaluates each on the
weighted-R² metric, then reports:

  1. each model individually
  2. simple average
  3. weight-optimized blend with non-negative weights summing to 1 (fit on
     the first half of the validation tail, scored on the second half so
     the fit doesn't peek at the held-out score)

No checkpoints are reloaded — works for any model, XGB included.

Usage::

    uv run python scripts/ensemble_blend.py \\
        --pred artifacts/bench/main/preds/xgb_static.npz \\
        --pred artifacts/bench/main/preds/lstm_modelr_online.npz \\
        --pred artifacts/bench/main/preds/gru_modelr_online.npz \\
        --out artifacts/bench/main/ensemble.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from janestreet.training.metrics import r2_weighted


def fit_blend_weights(
    preds: np.ndarray, y: np.ndarray, w: np.ndarray, n_iter: int = 4000, lr: float = 1e-2
) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise weighted MSE.

    Projected gradient: at each step, take a gradient step on the weighted
    squared loss, clip negatives, normalise to the simplex.
    """
    n_models = preds.shape[0]
    a = np.ones(n_models) / n_models
    for _ in range(n_iter):
        blend = a @ preds
        resid = blend - y
        grad = (w * resid) @ preds.T
        a = a - lr * grad
        a = np.clip(a, 0.0, None)
        s = a.sum()
        if s > 1e-12:
            a = a / s
    return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", action="append", required=True, help="repeatable; path to <model>_<static|online>.npz")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    names: list[str] = []
    P_list: list[np.ndarray] = []
    y_arr: np.ndarray | None = None
    w_arr: np.ndarray | None = None
    individual: list[dict] = []

    for pf in args.pred:
        npz = np.load(pf)
        preds = np.asarray(npz["preds"]).astype(np.float64)
        y = np.asarray(npz["y"]).astype(np.float64)
        w = np.asarray(npz["w"]).astype(np.float64)
        if y_arr is None:
            y_arr, w_arr = y, w
        # Sanity: all predictions must be aligned to the same valid slice.
        elif y_arr.shape != y.shape:
            raise ValueError(
                f"Shape mismatch: {Path(pf).name} has y.shape={y.shape}, "
                f"baseline has {y_arr.shape}"
            )
        r2 = r2_weighted(y, preds, w)
        names.append(Path(pf).stem)
        P_list.append(preds)
        individual.append({"name": Path(pf).stem, "path": str(pf), "r2": r2})
        print(f"  {Path(pf).stem:30s}  R²={r2:+.5f}", flush=True)

    P = np.stack(P_list)
    assert y_arr is not None and w_arr is not None

    # Simple unweighted average
    r2_avg = r2_weighted(y_arr, P.mean(axis=0), w_arr)
    print(f"\n  simple average        R²={r2_avg:+.5f}", flush=True)

    # Held-out blend
    n_obs = P.shape[1]
    split = n_obs // 2
    weights = fit_blend_weights(P[:, :split], y_arr[:split], w_arr[:split])
    r2_blend_train = r2_weighted(y_arr[:split], weights @ P[:, :split], w_arr[:split])
    r2_blend_test = r2_weighted(y_arr[split:], weights @ P[:, split:], w_arr[split:])
    r2_blend_full = r2_weighted(y_arr, weights @ P, w_arr)
    print(f"  optimal (fit on first half) R²[first ½]={r2_blend_train:+.5f}", flush=True)
    print(f"                              R²[2nd  ½]={r2_blend_test:+.5f}", flush=True)
    print(f"                              R²[full ]={r2_blend_full:+.5f}", flush=True)
    weight_map = dict(zip(names, weights.round(4).tolist(), strict=True))
    print(f"  weights = {weight_map}", flush=True)

    out = {
        "preds": [str(p) for p in args.pred],
        "individual": individual,
        "simple_avg_r2": r2_avg,
        "optimal_blend": {
            "weights": dict(zip(names, weights.tolist(), strict=True)),
            "r2_blend_train_half": r2_blend_train,
            "r2_blend_test_half": r2_blend_test,
            "r2_blend_full_valid": r2_blend_full,
        },
        "timestamp": datetime.now().isoformat(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten → {args.out}")


if __name__ == "__main__":
    main()
