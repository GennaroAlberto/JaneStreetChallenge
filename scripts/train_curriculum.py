"""Curriculum training — one model, warm-started across successive date batches.

Exposes a single model to *more* dates than fits in one training pass, without
ever holding more than one batch in RAM: train on batch 1 (cold), then
warm-start on batch 2, batch 3, … keeping the same weights in memory between
batches. A checkpoint + validation eval is saved after every batch, so you get
the whole trajectory (batch-1-only → all-batches) and can pick the best.

Chronological batches ending near the validation window exploit recency: the
final, most-influential batch is the one closest to the tail we score on.

All batches read from the same precomputed memmap (so standardization stats and
the feature set — base + lagged responders + responder-signal features — are
identical across batches and match the eval path).

Usage
-----

    uv run python scripts/train_curriculum.py \\
        --data artifacts/precomputed/pool700_full --model gru_modelr \\
        --batches 700-899,900-1099,1100-1299,1398-1597 \\
        --valid-lo 1599 --valid-hi 1698 \\
        --out artifacts/bench/gru_curriculum
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

# Warm torch's OpenMP pool before heavy numpy (macOS libomp guard).
torch.nn.functional.relu(torch.zeros(1024))

from janestreet.config import COL_DATE, COL_WEIGHT, Cfg  # noqa: E402
from janestreet.models.base import FitData  # noqa: E402
from janestreet.pipeline import (  # noqa: E402
    FullPipeline,
    Preprocessor,
    make_pipeline,
    prepare_dataset,
)
from janestreet.training.metrics import r2_weighted  # noqa: E402

# Curriculum model configs. Fewer epochs on warm batches (weights already good).
MODEL_KWARGS: dict[str, dict] = {
    "gru_modelr": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu", num_aux=4,
    ),
    "lstm_modelr": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        lr=1e-3, weight_decay=1e-2, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu", num_aux=4,
    ),
    "gru": dict(
        hidden_sizes=[64, 64], dropout_rates=[0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu",
    ),
}


def build_fitdata(data_dir: Path, manifest: dict, lo: int, hi: int) -> FitData:
    K = manifest["K"]
    X_mm = np.memmap(data_dir / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    resp = np.load(data_dir / "resp.f16.npy", mmap_mode="r")
    y = np.load(data_dir / "y.f32.npy", mmap_mode="r")
    w = np.load(data_dir / "w.f32.npy", mmap_mode="r")
    symbols = np.load(data_dir / "symbols.i16.npy", mmap_mode="r")
    dates = np.load(data_dir / "dates.i16.npy", mmap_mode="r")
    times = np.load(data_dir / "times.i16.npy", mmap_mode="r")
    ranges = manifest["date_row_ranges"]
    idx = np.concatenate([np.arange(*ranges[str(d)], dtype=np.int64)
                          for d in range(lo, hi + 1) if str(d) in ranges])
    return FitData(
        X=np.ascontiguousarray(X_mm[idx]).astype(np.float32),
        resp=(np.ascontiguousarray(resp[idx]).astype(np.float32)
              if resp.shape[1] else np.zeros((len(idx), 0), np.float32)),
        y=np.ascontiguousarray(y[idx]).astype(np.float32),
        w=np.ascontiguousarray(w[idx]).astype(np.float32),
        symbols=np.ascontiguousarray(symbols[idx]).astype(np.int64),
        dates=np.ascontiguousarray(dates[idx]).astype(np.int64),
        times=np.ascontiguousarray(times[idx]).astype(np.int64),
    )


def walk_forward(pipe: FullPipeline, df: pl.DataFrame, valid_dates: np.ndarray, target: str):
    pipe = copy.deepcopy(pipe)
    preds, ys, ws = [], [], []
    for i, dt in enumerate(valid_dates):
        day = df.filter(pl.col(COL_DATE) == dt)
        if i > 0:
            prev = df.filter(pl.col(COL_DATE) == int(dt) - 1)
            if prev.height > 0:
                pipe.update(prev)
        preds.append(pipe.predict(day))
        ys.append(day.select(target).to_series().to_numpy())
        ws.append(day.select(COL_WEIGHT).to_series().to_numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", choices=list(MODEL_KWARGS), default="gru_modelr")
    p.add_argument("--batches", type=str, required=True,
                   help="comma-separated lo-hi date windows, e.g. 700-899,900-1099,...")
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--cold-epochs", type=int, default=12)
    p.add_argument("--warm-epochs", type=int, default=8)
    p.add_argument("--warm-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    batches = []
    for tok in args.batches.split(","):
        lo, hi = (int(x) for x in tok.split("-"))
        assert hi <= manifest["train_until"], f"batch {tok} past train_until (leakage)"
        batches.append((lo, hi))

    print(f"[curriculum] model={args.model} batches={batches} "
          f"valid={args.valid_lo}..{args.valid_hi}  features K={manifest['K']} "
          f"(lags={manifest.get('lagged_responders')} sig={manifest.get('responder_signal_features')})",
          flush=True)

    # Build the pipeline once; keep the model in memory across batches.
    cfg = Cfg()
    cfg.model_name = args.model
    kw = MODEL_KWARGS[args.model].copy()
    kw["seed"] = args.seed
    cfg.model_kwargs = kw
    pipe = make_pipeline(cfg, feature_cols=manifest["feature_cols"], aux_cols=manifest["aux_cols"])
    with (data_dir / "preprocessor.pkl").open("rb") as f:
        pipe.preprocessor = pickle.load(f)  # noqa: S301  trusted local artifact
    assert isinstance(pipe.preprocessor, Preprocessor)

    # Eval frame (built once): valid tail + lookback, same feature config as train.
    cfg_eval = Cfg()
    cfg_eval.lagged_responders = list(manifest.get("lagged_responders", []))
    cfg_eval.responder_signal_features = bool(manifest.get("responder_signal_features", False))
    lookback = (cfg_eval.rolling_window + 967) // 968 + 1
    cfg_eval.min_date_id = args.valid_lo - lookback
    cfg_eval.max_date_id = args.valid_hi
    df_eval = prepare_dataset(cfg_eval, storage_precision="float32")
    valid_dates = np.arange(args.valid_lo, args.valid_hi + 1)
    valid_dates = valid_dates[np.isin(
        valid_dates, df_eval.select(pl.col(COL_DATE).unique()).to_series().to_numpy())]

    results = []
    for bi, (lo, hi) in enumerate(batches):
        warm = bi > 0
        pipe.model.epochs = args.warm_epochs if warm else args.cold_epochs
        if warm:
            pipe.model.lr = args.warm_lr
        t0 = time.time()
        fd = build_fitdata(data_dir, manifest, lo, hi)
        print(f"\n=== batch {bi+1}/{len(batches)}  dates {lo}..{hi}  "
              f"rows={fd.X.shape[0]:,}  {'WARM' if warm else 'COLD'} "
              f"(lr={pipe.model.lr}, epochs={pipe.model.epochs}) ===", flush=True)
        pipe.model.fit(fd, None, verbose=True, warm_start=warm)
        fit_s = time.time() - t0
        del fd

        # Eval this checkpoint (static + online) on the tail.
        static = copy.deepcopy(pipe)
        if hasattr(static.model, "lr_refit"):
            static.model.lr_refit = 0.0
        p_s, y_s, w_s = walk_forward(static, df_eval, valid_dates, cfg.target)
        r2_static = r2_weighted(y_s, p_s, w_s)
        p_o, y_o, w_o = walk_forward(pipe, df_eval, valid_dates, cfg.target)
        r2_online = r2_weighted(y_o, p_o, w_o)
        print(f"  [after batch {bi+1}] fit={fit_s/60:.1f}min  "
              f"static R²={r2_static:+.5f}  online R²={r2_online:+.5f}", flush=True)

        tag = f"b{bi+1}_{lo}_{hi}"
        with contextlib.suppress(NotImplementedError):
            pipe.save(out / "checkpoints" / f"{args.model}_{tag}.pkl")
        np.savez_compressed(out / "preds" / f"{args.model}_{tag}_online.npz",
            preds=p_o.astype(np.float32), y=y_o.astype(np.float32),
            w=w_o.astype(np.float32), valid_dates=valid_dates.astype(np.int32))
        results.append({"batch": bi + 1, "lo": lo, "hi": hi, "warm": warm,
                        "fit_s": fit_s, "r2_static": r2_static, "r2_online": r2_online})
        (out / "curriculum.json").write_text(json.dumps(
            {"model": args.model, "seed": args.seed, "batches": results,
             "data": str(data_dir), "timestamp": datetime.now().isoformat()}, indent=2))

    print("\n=== curriculum summary (online R² after each batch) ===", flush=True)
    for r in results:
        print(f"  batch {r['batch']} ({r['lo']}..{r['hi']}): {r['r2_online']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
