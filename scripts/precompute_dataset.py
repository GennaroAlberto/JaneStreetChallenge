"""Precompute a standardized feature memmap for temporal-bagging training.

Motivation
----------

Feature engineering (per-symbol rolling stats) needs the whole date range in
one frame, but a 500+ date TRAINING run OOMs on a 16 GB Mac — not because of
the polars frame (~5 GB Float16) but because ``_to_fitdata`` materialises a
~9 GB Float32 numpy ``X`` copy per fit. This script pays the feature-build
cost ONCE and writes the standardized ``X`` to a Float16 memmap on disk. A
downstream trainer (scripts/train_from_memmap.py) then streams per-date
slices straight from the memmap — peak RAM ~5-6 GB regardless of how many
dates it trains on.

It also enables **temporal bagging**: because the standardization statistics
are fit once on the full train range, many models can be trained on
different (overlapping) date subsets from the same memmap. Overlapping
windows give decorrelated models — the ensemble diversity our single-window
models lacked.

Leakage boundary
----------------

The preprocessor (quantile clipper + standardizer) is fit on dates
``[min_date, train_until]`` ONLY. The validation tail
``(train_until, max_date]`` is written to the memmap using those frozen
train-time statistics, never contributing to them. Get this wrong and you
leak the validation distribution into training.

Layout written to ``--out``
---------------------------

    X.f16.mmap        Float16 memmap, shape (N, K) — standardized features
    resp.f16.npy      Float16 (N, A) — aux-responder targets
    y.f32.npy         Float32 (N,)   — primary target (responder_6)
    w.f32.npy         Float32 (N,)   — sample weights
    symbols.i16.npy   int16 (N,)
    dates.i16.npy     int16 (N,)
    times.i16.npy     int16 (N,)
    preprocessor.pkl  the fitted Preprocessor (for the eval-time path)
    manifest.json     K, N, feature_cols, per-date row offsets, config

Usage
-----

    uv run python scripts/precompute_dataset.py \\
        --min-date 1097 --max-date 1698 --train-until 1598 \\
        --out artifacts/precomputed/tail602
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COL_DATE, COL_ID, COL_TIME, COL_WEIGHT, Cfg
from janestreet.data.features import FeatureBuilder
from janestreet.data.scaler import OnlineStandardizer, QuantileClipper
from janestreet.pipeline import Preprocessor, prepare_dataset


def fit_preprocessor_streaming(
    df: pl.DataFrame, feature_cols: list[str], train_dates: np.ndarray,
    clip_subsample: int, log,
) -> Preprocessor:
    """Fit clipper + standardizer on train dates only, without materialising full X.

    * Clipper quantiles: fit on a subsample of train dates (every
      ``clip_subsample``-th). 0.1%/99.9% quantiles are stable on a few tens
      of dates and this keeps the peak memory tiny.
    * Standardizer: streaming Welford over all train dates (clip each chunk
      first so the stats match the clipped feature space).
    """
    sub_dates = train_dates[::clip_subsample]
    log(f"  fitting clipper on {len(sub_dates)} subsampled train dates")
    X_sub = (
        df.filter(pl.col(COL_DATE).is_in(sub_dates))
        .select(feature_cols)
        .to_numpy()
        .astype(np.float32)
    )
    clipper = QuantileClipper().fit(X_sub)
    del X_sub

    log(f"  fitting standardizer via streaming Welford over {len(train_dates)} train dates")
    scaler = OnlineStandardizer()
    first = True
    for i, dt in enumerate(train_dates):
        X_dt = (
            df.filter(pl.col(COL_DATE) == dt)
            .select(feature_cols)
            .to_numpy()
            .astype(np.float32)
        )
        X_dt = clipper.transform(X_dt)
        if first:
            scaler.fit(X_dt)
            first = False
        else:
            scaler.partial_fit(X_dt)
        del X_dt
        if (i + 1) % 100 == 0:
            log(f"    standardizer: {i + 1}/{len(train_dates)} dates")
    return Preprocessor(feature_cols=feature_cols, clipper=clipper, scaler=scaler)


def main() -> None:
    p = argparse.ArgumentParser()
    # Default floor 700: the 8th-place solution's choice — n_times stabilises
    # at 968 from date 677, and they found earlier data didn't help. Gives a
    # ~900-date training pool (700..1598) to resample from.
    p.add_argument("--min-date", type=int, default=700)
    p.add_argument("--max-date", type=int, default=1698)
    p.add_argument(
        "--train-until", type=int, default=1598,
        help="preprocessor stats fit on dates <= this; dates above are the "
             "validation tail (frozen stats, no leakage)",
    )
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--precision", choices=["float16", "float32"], default="float16")
    p.add_argument("--clip-subsample", type=int, default=8,
                   help="use every Nth train date to fit the clip quantiles")
    p.add_argument("--aux-targets", type=str, default=None,
                   help="comma-separated aux responder cols; default = cfg.aux_targets")
    p.add_argument("--lagged-responders", type=str, default=None,
                   help="comma-separated responder ids to add as previous-day "
                        "input features (e.g. '0,1,2,3,4,5,6,7,8'). Needed for "
                        "the TimeXer endogenous stream; also helps other models.")
    p.add_argument("--responder-signal-features", action="store_true",
                   help="add the engineered multi-scale-momentum + venue-spread "
                        "features (needs lagged responders {3..8}). Standardized "
                        "consistently here so train and eval match.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "precompute_log.txt"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    t_start = time.time()
    cfg = Cfg()
    cfg.min_date_id = args.min_date
    cfg.max_date_id = args.max_date
    if args.lagged_responders:
        cfg.lagged_responders = [int(r) for r in args.lagged_responders.split(",")]
    cfg.responder_signal_features = bool(args.responder_signal_features)
    aux_cols = (
        args.aux_targets.split(",") if args.aux_targets else list(cfg.aux_targets)
    )

    log(f"precompute → {out}")
    log(f"range {args.min_date}..{args.max_date}  train_until={args.train_until}  "
        f"precision={args.precision}  lagged_responders={cfg.lagged_responders}  "
        f"responder_signal_features={cfg.responder_signal_features}")

    # 1. Feature engineering on the full range (Float16 storage — peak ~5 GB).
    t0 = time.time()
    df = prepare_dataset(cfg, storage_precision="float16")
    log(f"feature build: {df.height:,} rows × {df.width} cols in {time.time()-t0:.1f}s")

    feature_cols = FeatureBuilder(cfg).feature_columns()
    K = len(feature_cols)
    N = df.height

    dates_all = np.sort(df.select(pl.col(COL_DATE).unique()).to_series().to_numpy())
    train_dates = dates_all[dates_all <= args.train_until]
    log(f"train dates for stats: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)})")

    # 2. Fit the preprocessor on train dates only (leakage boundary).
    pre = fit_preprocessor_streaming(df, feature_cols, train_dates, args.clip_subsample, log)
    with (out / "preprocessor.pkl").open("wb") as f:
        pickle.dump(pre, f, protocol=pickle.HIGHEST_PROTOCOL)
    log("preprocessor fit + saved")

    # 3. Sort the whole frame by (symbol, date, time) so each (symbol,date) is a
    #    contiguous 968-row block, and dates are grouped. This is the row order
    #    the memmap will use — the trainer slices contiguous date windows.
    df = df.sort([COL_ID, COL_DATE, COL_TIME])

    # 4. Allocate the memmap and companion arrays; fill per date.
    np_dtype = np.float16 if args.precision == "float16" else np.float32
    x_filename = "X.f16.mmap" if args.precision == "float16" else "X.f32.mmap"
    X_mm = np.memmap(out / x_filename, dtype=np_dtype, mode="w+", shape=(N, K))

    resp_all = np.empty((N, len(aux_cols)), dtype=np.float16) if aux_cols else np.zeros((N, 0), np.float16)
    y_all = np.empty(N, dtype=np.float32)
    w_all = np.empty(N, dtype=np.float32)
    sym_all = np.empty(N, dtype=np.int16)
    date_all = np.empty(N, dtype=np.int16)
    time_all = np.empty(N, dtype=np.int16)

    manifest_dates: dict[str, list[int]] = {}
    cursor = 0
    t0 = time.time()
    for di, dt in enumerate(dates_all):
        day = df.filter(pl.col(COL_DATE) == dt)
        n = day.height
        Xd = day.select(feature_cols).to_numpy().astype(np.float32)
        # clip → standardize → nan_to_num, using the frozen train-time stats.
        # (Preprocessor.transform expects a polars frame; apply the fitted
        # clipper + scaler to the numpy array directly instead.)
        Xd = pre.scaler.transform(pre.clipper.transform(Xd))
        X_mm[cursor:cursor + n] = Xd.astype(np_dtype)
        if aux_cols:
            resp_all[cursor:cursor + n] = day.select(aux_cols).to_numpy().astype(np.float16)
        y_all[cursor:cursor + n] = day.select(cfg.target).to_series().to_numpy().astype(np.float32)
        w_all[cursor:cursor + n] = day.select(COL_WEIGHT).to_series().to_numpy().astype(np.float32)
        sym_all[cursor:cursor + n] = day.select(COL_ID).to_series().to_numpy().astype(np.int16)
        date_all[cursor:cursor + n] = day.select(COL_DATE).to_series().to_numpy().astype(np.int16)
        time_all[cursor:cursor + n] = day.select(COL_TIME).to_series().to_numpy().astype(np.int16)
        manifest_dates[str(int(dt))] = [cursor, cursor + n]
        cursor += n
        del Xd, day
        if (di + 1) % 100 == 0:
            log(f"  wrote {di + 1}/{len(dates_all)} dates  ({cursor:,}/{N:,} rows, "
                f"{time.time()-t0:.0f}s)")

    assert cursor == N, f"row cursor {cursor} != N {N}"
    X_mm.flush()
    del X_mm

    np.save(out / "resp.f16.npy", resp_all)
    np.save(out / "y.f32.npy", y_all)
    np.save(out / "w.f32.npy", w_all)
    np.save(out / "symbols.i16.npy", sym_all)
    np.save(out / "dates.i16.npy", date_all)
    np.save(out / "times.i16.npy", time_all)

    lag_cols = FeatureBuilder(cfg).lagged_responder_columns()
    manifest = {
        "N": int(N), "K": int(K),
        "precision": args.precision,
        "X_file": x_filename,
        "feature_cols": feature_cols,
        "aux_cols": aux_cols,
        "lagged_responders": cfg.lagged_responders,
        "lag_cols": lag_cols,
        "lag_col_indices": [feature_cols.index(c) for c in lag_cols],
        "responder_signal_features": cfg.responder_signal_features,
        "target": cfg.target,
        "min_date": args.min_date, "max_date": args.max_date,
        "train_until": args.train_until,
        "n_times": 968,
        "date_row_ranges": manifest_dates,
        "row_order": "sorted by (symbol, date, time)",
        "timestamp": datetime.now().isoformat(),
    }
    with (out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    disk_gb = (out / manifest["X_file"]).stat().st_size / 1e9
    log(f"\nDONE in {(time.time()-t_start)/60:.1f} min")
    log(f"  memmap: {manifest['X_file']}  ({disk_gb:.2f} GB, {N:,} × {K} {args.precision})")
    log(f"  manifest: {len(manifest_dates)} dates indexed")


if __name__ == "__main__":
    main()
