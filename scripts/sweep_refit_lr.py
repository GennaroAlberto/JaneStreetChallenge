"""Reload a trained pipeline checkpoint and re-evaluate online refit at several LRs.

The training is done; we only need to vary the daily-refit learning rate
and walk forward over the validation tail. Cost is ~one validation pass
per LR, so the sweep typically finishes in a few minutes.

Usage::

    uv run python scripts/sweep_refit_lr.py \\
        --ckpt artifacts/bench/main/checkpoints/lstm_modelr.pkl \\
        --min-date 1399 --max-date 1698 --valid 100 \\
        --lrs 1e-4 3e-4 1e-3 3e-3
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COL_DATE, COL_WEIGHT, Cfg
from janestreet.pipeline import FullPipeline, prepare_dataset
from janestreet.training.metrics import r2_weighted


def walk_forward(
    pipe: FullPipeline, df: pl.DataFrame, valid_dates: np.ndarray, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One walk-forward pass: predict day, then update on the previous day."""
    pipe = copy.deepcopy(pipe)
    preds, ys, ws = [], [], []
    for i, dt in enumerate(valid_dates):
        day = df.filter(pl.col(COL_DATE) == dt)
        if i > 0:
            prev = df.filter(pl.col(COL_DATE) == int(dt) - 1)
            if prev.height > 0:
                pipe.update(prev)
        preds.append(pipe.predict(day))
        ys.append(day.select(target_col).to_series().to_numpy())
        ws.append(day.select(COL_WEIGHT).to_series().to_numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--min-date", type=int, default=1399)
    p.add_argument("--max-date", type=int, default=1698)
    p.add_argument("--valid", type=int, default=100)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--dump-preds-at-lr", type=float, default=None,
        help="if set, also dump <ckpt-name>_online_lr<X>.npz with preds at this LR",
    )
    p.add_argument(
        "--preds-out-dir", type=str, default=None,
        help="directory for the .npz dump (defaults to ckpt.parent.parent / 'preds')",
    )
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    out_path = Path(args.out) if args.out else (ckpt.parent / f"{ckpt.stem}_lr_sweep.json")

    cfg = Cfg(); cfg.min_date_id = args.min_date; cfg.max_date_id = args.max_date
    df = prepare_dataset(cfg)
    dates = np.sort(df.select(pl.col(COL_DATE).unique()).to_series().to_numpy())
    valid_dates = dates[-args.valid:]

    print(f"loaded ckpt = {ckpt}", flush=True)
    print(f"valid dates: {valid_dates[0]}..{valid_dates[-1]} ({len(valid_dates)})", flush=True)
    print(f"LR sweep: {args.lrs}", flush=True)

    results = []
    base = FullPipeline.load(ckpt)
    target_col = base.target_col

    # Baseline: static (no refit)
    t0 = time.time()
    static = copy.deepcopy(base)
    if hasattr(static.model, "lr_refit"):
        static.model.lr_refit = 0.0
    preds, ys, ws = walk_forward(static, df, valid_dates, target_col)
    r2 = r2_weighted(ys, preds, ws)
    results.append({"lr_refit": 0.0, "r2": r2, "seconds": time.time() - t0})
    print(f"  lr=0.0          R²={r2:+.5f}  ({time.time()-t0:.1f}s)", flush=True)

    for lr in args.lrs:
        pipe = copy.deepcopy(base)
        if not hasattr(pipe.model, "lr_refit"):
            print(f"  skip lr={lr}: model has no refit hook")
            continue
        pipe.model.lr_refit = float(lr)
        t0 = time.time()
        preds, ys, ws = walk_forward(pipe, df, valid_dates, target_col)
        r2 = r2_weighted(ys, preds, ws)
        results.append({"lr_refit": float(lr), "r2": r2, "seconds": time.time() - t0})
        print(f"  lr={lr:>9.0e}    R²={r2:+.5f}  ({time.time()-t0:.1f}s)", flush=True)

    out = {
        "ckpt": str(ckpt),
        "min_date": args.min_date, "max_date": args.max_date, "valid": args.valid,
        "results": results,
        "best": max(results, key=lambda r: r["r2"]),
        "timestamp": datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nbest: lr={out['best']['lr_refit']}  R²={out['best']['r2']:+.5f}")
    print(f"written → {out_path}")

    # Optional one-off prediction dump at a chosen LR (e.g. the winner of the sweep).
    if args.dump_preds_at_lr is not None:
        pipe = copy.deepcopy(base)
        if not hasattr(pipe.model, "lr_refit"):
            print("WARNING: model has no refit hook; --dump-preds-at-lr is a no-op")
        else:
            pipe.model.lr_refit = float(args.dump_preds_at_lr)
            preds, ys, ws = walk_forward(pipe, df, valid_dates, target_col)
            r2 = r2_weighted(ys, preds, ws)
            preds_dir = (
                Path(args.preds_out_dir)
                if args.preds_out_dir
                else ckpt.parent.parent / "preds"
            )
            preds_dir.mkdir(parents=True, exist_ok=True)
            lr_tag = f"{args.dump_preds_at_lr:g}".replace("-", "m")  # filename-safe
            npz_path = preds_dir / f"{ckpt.stem}_online_lr{lr_tag}.npz"
            np.savez_compressed(
                npz_path,
                preds=preds.astype(np.float32),
                y=ys.astype(np.float32),
                w=ws.astype(np.float32),
                valid_dates=np.asarray(valid_dates, dtype=np.int32),
            )
            print(f"dumped preds @ lr={args.dump_preds_at_lr} → {npz_path}  (R²={r2:+.5f})")


if __name__ == "__main__":
    main()
