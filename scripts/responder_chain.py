"""Responder prediction chain: predict the easy targets, feed them to r6.

Two questions, one script:

1. **How predictable is each responder?** Per the reverse-engineered
   construction, the nine responders are SMAs (windows 4/20/120) of one
   signal on two venues plus their differences — their predictability
   should differ systematically (longer windows integrate persistent
   effects; venue spreads may mean-revert). Nobody has measured this here.
2. **Does a prediction chain help?** ModelR's aux branches already blend
   *internal* predictions of other responders into y; this is the explicit
   version: stage-1 XGBs predict all nine responders from features, and
   stage-2 receives [features, r̂0..r̂8] for the r6 target. Deployable —
   predicted (not true) same-time responders are functions of legal inputs.

Leak hygiene: stage-1 fits on an EARLIER window (1399–1498) than stage-2's
train (1499–1598), so stage-2 never consumes in-sample stage-1 fits; both
stages are evaluated on the untouched tail (1599–1698).

Usage
-----
    uv run python scripts/responder_chain.py --out artifacts/bench/chain_lab
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from janestreet.config import Cfg
from janestreet.data.ingest import scan_train_dates

RESPONDERS = [f"responder_{i}" for i in range(9)]


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi, step=1):
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    rows = np.concatenate(parts)
    return rows[::step] if step > 1 else rows


def load_raw_responders(lo, hi, step=1):
    """All nine responders for the date range, in memmap row order
    (date-major, then symbol, then time)."""
    cfg = Cfg()
    df = (scan_train_dates(cfg, lo, hi)
          .select(["date_id", "symbol_id", "time_id", *RESPONDERS])
          .collect()
          .sort(["date_id", "symbol_id", "time_id"]))
    R = df.select(RESPONDERS).to_numpy().astype(np.float32)
    return R[::step] if step > 1 else R


def make_xgb(n_estimators, seed):
    return xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", max_bin=128, n_jobs=1,
        early_stopping_rounds=50, random_state=seed,
        objective="reg:squarederror",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--s1-lo", type=int, default=1399)
    p.add_argument("--s1-hi", type=int, default=1498)
    p.add_argument("--s2-lo", type=int, default=1499)
    p.add_argument("--s2-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--row-step", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/chain_lab")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg):
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    K = manifest["K"]
    step = args.row_step
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y_all = np.load(data / "y.f32.npy", mmap_mode="r")
    w_all = np.load(data / "w.f32.npy", mmap_mode="r")

    r1 = rows_for_dates(manifest, args.s1_lo, args.s1_hi, step)
    r2_ = rows_for_dates(manifest, args.s2_lo, args.s2_hi, step)
    rv = rows_for_dates(manifest, args.valid_lo, args.valid_hi, step)
    X1 = np.ascontiguousarray(X_mm[r1]).astype(np.float32)
    X2 = np.ascontiguousarray(X_mm[r2_]).astype(np.float32)
    Xv = np.ascontiguousarray(X_mm[rv]).astype(np.float32)
    w1, w2, wv = (np.ascontiguousarray(w_all[r]) for r in (r1, r2_, rv))
    y2, yv = np.ascontiguousarray(y_all[r2_]), np.ascontiguousarray(y_all[rv])

    log("loading raw responders (all nine, memmap order)…")
    R1 = load_raw_responders(args.s1_lo, args.s1_hi, step)
    R2 = load_raw_responders(args.s2_lo, args.s2_hi, step)
    Rv = load_raw_responders(args.valid_lo, args.valid_hi, step)

    # alignment assertion: raw responder_6 must equal memmap y at f16 tol
    if not np.allclose(R2[:, 6].astype(np.float16).astype(np.float32), y2, atol=2e-3):
        raise ValueError("raw/memmap alignment failed on stage-2 rows")
    if not np.allclose(Rv[:, 6].astype(np.float16).astype(np.float32), yv, atol=2e-3):
        raise ValueError("raw/memmap alignment failed on valid rows")
    log("alignment OK")

    # ---- stage 1: per-responder predictability + prediction columns --------
    pred_2 = np.empty((len(X2), 9), np.float32)
    pred_v = np.empty((len(Xv), 9), np.float32)
    table = []
    for i in range(9):
        m = make_xgb(args.n_estimators, args.seed)
        m.fit(X1, R1[:, i], sample_weight=w1,
              eval_set=[(X2, R2[:, i])], sample_weight_eval_set=[w2],
              verbose=False)
        pred_2[:, i] = m.predict(X2)
        pred_v[:, i] = m.predict(Xv)
        r2_mid = r2_weighted(R2[:, i], pred_2[:, i], w2)
        r2_tail = r2_weighted(Rv[:, i], pred_v[:, i], wv)
        table.append({"responder": i, "r2_stage2win": r2_mid, "r2_tail": r2_tail})
        log(f"  r{i}: R²(1499-1598)={r2_mid:+.5f}   R²(tail)={r2_tail:+.5f}")
        del m
        gc.collect()

    # ---- stage 2: r6 with vs without chained predictions -------------------
    results = []
    for name, xt, xv in (
        ("base", X2, Xv),
        ("base+chain", np.hstack([X2, pred_2]), np.hstack([Xv, pred_v])),
    ):
        m = make_xgb(1500, args.seed)
        m.fit(xt, y2, sample_weight=w2,
              eval_set=[(xv, yv)], sample_weight_eval_set=[wv], verbose=False)
        r2v = r2_weighted(yv, m.predict(xv), wv)
        best_it = int(getattr(m, "best_iteration", 0) or 0)
        log(f"  stage2 {name:12s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
            f"R²={r2v:+.5f}")
        results.append({"variant": name, "n_features": int(xt.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2v})
        del m
        gc.collect()
    results[1]["delta_vs_base"] = (results[1]["r2_weighted"]
                                   - results[0]["r2_weighted"])

    (out / "chain.json").write_text(json.dumps({
        "stage1": table, "stage2": results,
        "windows": {"s1": [args.s1_lo, args.s1_hi], "s2": [args.s2_lo, args.s2_hi],
                    "valid": [args.valid_lo, args.valid_hi]},
        "row_step": step, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log(f"chain delta vs base: {results[1]['delta_vs_base']:+.5f} → {out}/chain.json")


if __name__ == "__main__":
    main()
