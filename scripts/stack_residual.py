"""Non-linear residual stacker over the per-model .npz prediction dumps.

The plain ``ensemble_blend.py`` fits a linear (convex) combination of the
predictions. That's optimal iff the residuals are uncorrelated across
models — which we've seen is *not* quite true here (adding a signature
stream to the RNN ensemble slightly *hurts* the simple average even
though the model has non-trivial standalone R²).

This script fits a **non-linear stacker** (small XGB by default) on the
first half of the validation tail and evaluates on the second half, so the
comparison to the linear blender is apples-to-apples. Framing it as
residual boosting: pick a designated "base" model, subtract it, and let
XGB find the residual structure the other streams collectively encode.

Two modes:

* ``--mode residual`` (default): target = ``y − base_pred``; features are
  each stream's prediction *and* its deviation from base. XGB tries to
  correct the base using the ensemble's disagreement signal. Final
  prediction = base + xgb_correction.
* ``--mode direct``: target = ``y``; features are the raw predictions.
  This is straight non-linear stacking, no residual framing.

Both are trained on the first half and evaluated on the second half of
the validation tail.

Usage
-----

    uv run python scripts/stack_residual.py \\
        --pred artifacts/bench/run2_280d/preds/xgb_static.npz \\
        --pred artifacts/bench/run2_280d/preds/gru_modelr_online.npz \\
        --pred artifacts/bench/run2_280d/preds/lstm_modelr_online.npz \\
        --pred artifacts/bench/run3_sig_v3/preds/mlp_logsig_min_ar_k8_d3_online.npz \\
        --base lstm_modelr_online \\
        --mode residual \\
        --out artifacts/bench/run3_sig_v3/stack_residual.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


# Inline the weighted-R² helper instead of importing from
# ``janestreet.training.metrics`` — that module unconditionally imports
# torch, which on macOS arm64 collides with xgboost's libomp and segfaults
# on any subsequent ``xgb.XGBRegressor.fit`` call.
def r2_weighted(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    num = float(np.average((y_pred - y_true) ** 2, weights=w))
    den = float(np.average(y_true ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]


def load_streams(paths: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load N .npz prediction dumps; return (name -> preds), y, w."""
    preds: dict[str, np.ndarray] = {}
    y_ref: np.ndarray | None = None
    w_ref: np.ndarray | None = None
    for p in paths:
        name = Path(p).stem
        npz = np.load(p)
        preds[name] = np.asarray(npz["preds"]).astype(np.float64)
        y = np.asarray(npz["y"]).astype(np.float64)
        w = np.asarray(npz["w"]).astype(np.float64)
        if y_ref is None:
            y_ref, w_ref = y, w
        elif y_ref.shape != y.shape:
            raise ValueError(f"Shape mismatch: {name} has {y.shape}, expected {y_ref.shape}")
    assert y_ref is not None and w_ref is not None
    return preds, y_ref, w_ref


def make_features(
    preds: dict[str, np.ndarray], base_name: str | None, mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix for the stacker.

    * If ``mode == 'residual'`` and ``base_name`` is set, include base_pred,
      each other prediction, and each other's deviation from base.
    * Otherwise (``direct``), include only the raw predictions.
    """
    names = list(preds.keys())
    if mode == "residual":
        assert base_name in preds, f"base {base_name!r} not in preds"
        base = preds[base_name]
        cols: list[np.ndarray] = [base]
        feat_names = [base_name]
        for n in names:
            if n == base_name:
                continue
            cols.append(preds[n])
            feat_names.append(n)
            cols.append(preds[n] - base)
            feat_names.append(f"{n}_minus_base")
    else:
        cols = [preds[n] for n in names]
        feat_names = names
    return np.column_stack(cols).astype(np.float32), feat_names


def fit_xgb_stacker(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
    n_estimators: int, max_depth: int, learning_rate: float, seed: int,
):
    if xgb is None:
        raise ImportError("xgboost is required for the stacker; already declared in pyproject")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.9,
        tree_method="hist",
        max_bin=64,
        n_jobs=1,
        objective="reg:squarederror",
        random_state=seed,
    )
    model.fit(X, y, sample_weight=w)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", action="append", required=True, help="repeatable; path to <model>_<static|online>.npz")
    p.add_argument("--base", type=str, default=None, help="base model name (stem of its npz path); needed for --mode residual")
    p.add_argument("--mode", choices=["residual", "direct"], default="residual")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    preds, y, w = load_streams(args.pred)
    # Simple average and any per-model R²s (for reporting)
    per_model = {n: float(r2_weighted(y, v, w)) for n, v in preds.items()}
    avg = np.mean(list(preds.values()), axis=0)
    r2_simple_avg = float(r2_weighted(y, avg, w))

    if args.mode == "residual":
        base_name = args.base
        if base_name is None:
            # Default: use the best individual as base.
            base_name = max(per_model, key=per_model.get)
            print(f"[stack] --base not set — auto-picking best individual: {base_name}", flush=True)
        base = preds[base_name]
        target = y - base
    else:
        base_name = None
        target = y

    X, feat_names = make_features(preds, base_name, args.mode)

    # Chronological split — first half of valid = train, second half = held-out test.
    n = len(y)
    split = n // 2
    Xtr, Xte = X[:split], X[split:]
    ytr_target = target[:split]
    wtr, wte = w[:split], w[split:]

    model = fit_xgb_stacker(
        Xtr, ytr_target, wtr,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    pred_train_target = model.predict(Xtr)
    pred_test_target = model.predict(Xte)

    # Reconstruct predictions of y depending on framing.
    if args.mode == "residual":
        base_tr, base_te = preds[base_name][:split], preds[base_name][split:]
        pred_train = base_tr + pred_train_target
        pred_test = base_te + pred_test_target
    else:
        pred_train = pred_train_target
        pred_test = pred_test_target

    r2_train = float(r2_weighted(y[:split], pred_train, wtr))
    r2_test = float(r2_weighted(y[split:], pred_test, wte))

    # Simple-average baseline on the same halves so numbers are comparable.
    r2_avg_train = float(r2_weighted(y[:split], avg[:split], wtr))
    r2_avg_test = float(r2_weighted(y[split:], avg[split:], wte))

    print("Per-model R² (full valid):", flush=True)
    for n_, r2 in sorted(per_model.items(), key=lambda kv: -kv[1]):
        print(f"  {n_:40s}  {r2:+.5f}")

    print(f"\nStacker (mode={args.mode}, base={base_name}):", flush=True)
    print(f"  first-half  train  R² = {r2_train:+.5f}  (simple avg on same slice = {r2_avg_train:+.5f})")
    print(f"  second-half TEST   R² = {r2_test :+.5f}  (simple avg on same slice = {r2_avg_test :+.5f})  ← held-out")
    print(f"\nBaseline (simple average on full valid): {r2_simple_avg:+.5f}")

    # Feature importances — useful diagnostic.
    fi = dict(zip(feat_names, model.feature_importances_.tolist(), strict=True))
    print("\nFeature importances (gain):")
    for n_, imp in sorted(fi.items(), key=lambda kv: -kv[1]):
        print(f"  {n_:40s}  {imp:.4f}")

    out = {
        "mode": args.mode,
        "base": base_name,
        "streams": [str(p) for p in args.pred],
        "per_model_r2": per_model,
        "simple_avg_r2_full_valid": r2_simple_avg,
        "simple_avg_r2_train_half": r2_avg_train,
        "simple_avg_r2_test_half": r2_avg_test,
        "stacker_r2_train_half": r2_train,
        "stacker_r2_test_half": r2_test,
        "feature_importances": fi,
        "xgb_kwargs": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "timestamp": datetime.now().isoformat(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten → {args.out}")


if __name__ == "__main__":
    main()
