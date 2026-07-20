"""Train one bagged model on a resample of the precomputed date pool.

Reads the standardized-feature memmap produced by
``scripts/precompute_dataset.py`` and trains a single model on a *resample*
of the training-date pool (dates >= 700, matching the 8th-place floor where
``n_times`` stabilises at 968). Each ``--seed`` draws a different set of
dates, so N runs give N decorrelated models for the ensemble. The
preprocessor (global standardization stats) is shared across all bags; only
the training dates differ. Because ``X`` is read per-date from the memmap,
training never materialises the full pool — a resampled model uses only its
own dates' rows.

Resample modes (``--resample-mode``)
------------------------------------

* ``subsample`` (default): ``--resample-frac`` of the pool WITHOUT
  replacement (0.63 ≈ a bootstrap's unique fraction). RAM-light and clean.
* ``bootstrap``: |pool| draws WITH replacement (classic bagging; heavier).
* ``window``: explicit contiguous ``--train-lo/--train-hi`` (no resampling).

Evaluation uses the in-memory path on the validation tail with the frozen
preprocessor, so the online-refit standardizer still adapts day-to-day.

Example bag (5 members, same architecture, different resamples)
---------------------------------------------------------------

    for S in 1 2 3 4 5; do
      uv run python scripts/train_from_memmap.py \\
        --data artifacts/precomputed/pool700 \\
        --model lstm_modelr --resample-mode subsample --resample-frac 0.63 \\
        --seed $S --tag bag$S --out artifacts/bench/bag700
    done
    # then blend the 5 *_online.npz with scripts/ensemble_blend.py
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pickle
import sys
import time
from datetime import datetime
import os
from pathlib import Path

TSLAB_SRC = Path(os.environ.get(
    "TSLAB_SRC", Path.home() / "Desktop" / "code" / "tslab" / "src"))

import numpy as np
import polars as pl
import torch

# macOS libomp guard: the first *multithreaded* torch op initialises torch's
# OpenMP threadpool. If heavy numpy work (memmap fancy-indexing) spins up
# numpy's OMP first, torch's first parallel op collides and segfaults. Warm
# torch's pool here — before any numpy — so the pools initialise in a safe
# order. bench.py avoids this incidentally by building the model (torch ops)
# before its numpy; this script touches the memmap first, so we're explicit.
torch.nan_to_num(torch.zeros(1024))

from janestreet.config import COL_DATE, COL_WEIGHT, Cfg  # noqa: E402
from janestreet.data.resample import resolve_bootstrap_groups, sample_dates  # noqa: E402
from janestreet.models.base import FitData  # noqa: E402
from janestreet.pipeline import (  # noqa: E402
    FullPipeline,
    Preprocessor,
    make_pipeline,
    prepare_dataset,
)
from janestreet.training.metrics import r2_weighted  # noqa: E402

# Per-model kwargs mirror scripts/bench.py's tuned configs. Keep in sync.
MODEL_KWARGS: dict[str, dict] = {
    "lstm_modelr": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        lr=1e-3, weight_decay=1e-2, epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=1e-3, device="cpu", num_aux=4,
    ),
    "gru_modelr": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        # rewalk 2026-07-18: plain GRU also prefers 3e-4 refits
        # (+0.00811 → +0.00845 online); LSTM keeps 1e-3 (unswept — rewalk
        # cheap via scripts/rewalk_checkpoint.py before changing it)
        lr=1e-3, weight_decay=1e-2, epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-4, device="cpu", num_aux=4,
    ),
    # ModelR + cross-sectional symbol attention (EDA: weak symbols should
    # borrow strength; the RNNs otherwise never mix symbols).
    # refit-lr swept 2026-07-18: attention params are brittle under 1e-3
    # daily steps (online +0.00679 → +0.00870 at 3e-4; 3e-3 catastrophic).
    # The plain-RNN cliff does NOT transfer across the attention boundary.
    "gru_modelr_xsec": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-4, device="cpu", num_aux=4,
        xsec_heads=4,
    ),
    "lstm_modelr_xsec": dict(
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-4, device="cpu", num_aux=4,
        xsec_heads=4,
    ),
    # Supervised denoising AE + MLP (2021 JS winner family; row-wise).
    "ae_mlp": dict(
        latent=16, enc_hidden=128, mlp_hidden=[256, 256, 128],
        dropout=0.2, noise_std=0.1, num_aux=4, aux_weight=1.0,
        recon_weight=0.5, lr=1e-3, weight_decay=1e-2, epochs=20,
        batch_size=8192, early_stopping_patience=3, lr_refit=1e-4,
        device="cpu",
    ),
    # ---- small-intensity models (fast; good for validating the bag loop
    #      and cheap ensemble members) ----
    # Single-branch GRU/LSTM: no aux heads, 2 layers × 64 — ~4× lighter than
    # the modelr variants. Still sequence models, still online-refit.
    "gru": dict(
        hidden_sizes=[64, 64], dropout_rates=[0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, epochs=12, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu",
    ),
    "lstm": dict(
        hidden_sizes=[64, 64], dropout_rates=[0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, epochs=12, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu",
    ),
    # Plain MLP — non-sequence, flat row batches. Fastest DL baseline. No
    # online refit (it has no lr_refit hook), so only a static stream.
    "mlp": dict(
        hidden_sizes=[256, 256], dropout=0.1, lr=1e-3, weight_decay=1e-2,
        epochs=20, batch_size=16384, early_stopping_patience=4, device="cpu",
    ),
    # MLP + within-day signature (AR-selected channels, depth 3, minimal
    # log-sig if iisignature present else classical). Stateless per-timestep.
    "mlp_sig": dict(
        hidden_sizes=[256, 256], dropout=0.1, lr=1e-3, weight_decay=1e-2,
        epochs=15, batch_size=1, grad_clip=1.0, early_stopping_patience=3,
        lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="signature",
    ),
    # Causal iTransformer/TimeXer-inspired. Inverted attention over the
    # AR-selected variate features + endogenous global token + causal temporal
    # attention. Heavier than the small models; a distinct ensemble member.
    "itransformer": dict(
        d_model=96, n_heads=4, n_inv_layers=2, n_temporal_layers=2,
        ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2, epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        variate_channels=[13, 63, 42, 33, 70, 22, 20, 56],
    ),
    # TimeXer-for-JS. endo_channels are injected from the manifest at run
    # time (the lagged-responder columns) — requires a --lagged-responders
    # precompute. patch_len must divide into 968 with padding handled.
    "timexer": dict(
        d_model=96, n_heads=4, n_endo_layers=2, n_exo_layers=2, patch_len=44,
        ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2, epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
    ),
}


def load_manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def _build_fitdata_parquet(
    data_dir: Path, manifest: dict, date_list: np.ndarray,
    group_override: np.ndarray | None,
    aux_cols: list[str] | None = None,
) -> FitData:
    """Parquet-backed twin of ``build_fitdata_for_dates`` (see its docstring).

    Reads one ``data/date_id=<d>/part.parquet`` partition per requested date and
    slices out the feature/aux/target columns — so, like the memmap path, only
    the selected dates are ever materialised. Partitions are cached per call so a
    bootstrap that repeats a date reads it from disk once.
    """
    feature_cols = manifest["feature_cols"]
    aux_cols = aux_cols if aux_cols is not None else manifest["aux_cols"]
    target = manifest["target"]
    K, A = len(feature_cols), len(aux_cols)

    # Preallocate from the manifest's row counts and fill in place. The old
    # implementation cached every partition's DataFrame and concatenated
    # per-date numpy parts at the end — peak ~3x the final size, which OOMs
    # a free-tier Colab (12.7 GB) on a 280-date window. This path peaks at
    # final size + one date's frame.
    ranges = manifest["date_row_ranges"]

    def path_for(d: int) -> Path:
        return data_dir / "data" / f"date_id={int(d)}" / "part.parquet"

    counts = [
        (ranges[str(int(d))][1] - ranges[str(int(d))][0])
        if str(int(d)) in ranges and path_for(d).exists() else 0
        for d in date_list
    ]
    n_total = int(sum(counts))
    X = np.empty((n_total, K), np.float32)
    resp = np.empty((n_total, A), np.float32) if A else np.zeros((n_total, 0), np.float32)
    y = np.empty(n_total, np.float32)
    w = np.empty(n_total, np.float32)
    sym = np.empty(n_total, np.int64)
    dat = np.empty(n_total, np.int64)
    tim = np.empty(n_total, np.int64)

    c = 0
    for occ, d in enumerate(date_list):
        n = counts[occ]
        if n == 0:
            continue
        df = pl.read_parquet(path_for(d))
        if df.height != n:
            raise ValueError(
                f"date {int(d)}: partition has {df.height} rows, manifest says {n}")
        X[c:c + n] = df.select(feature_cols).to_numpy()
        if A:
            resp[c:c + n] = df.select(aux_cols).to_numpy()
        y[c:c + n] = df.get_column(target).to_numpy()
        w[c:c + n] = df.get_column("weight").to_numpy()
        sym[c:c + n] = df.get_column("symbol_id").to_numpy()
        tim[c:c + n] = df.get_column("time_id").to_numpy()
        dat[c:c + n] = (int(group_override[occ]) if group_override is not None
                        else int(d))
        c += n
        del df
    assert c == n_total
    return FitData(X=X, resp=resp, y=y, w=w, symbols=sym, dates=dat, times=tim)


def build_fitdata_for_dates(
    data_dir: Path, manifest: dict, date_list: np.ndarray,
    group_override: np.ndarray | None = None,
    aux_cols: list[str] | None = None,
) -> FitData:
    """Assemble a FitData for an arbitrary list of dates, reading only those.

    ``date_list`` may contain duplicates (bootstrap). For each occurrence we
    include that date's rows; ``group_override`` (one id per occurrence, same
    length as ``date_list``) is broadcast over the date's rows and used as the
    ``dates`` array so ``DateBatchDataset`` keeps duplicates as separate
    batches. If ``group_override`` is None (subsample / contiguous window),
    the real date_ids are used.

    Reads from whichever artifact format the manifest declares. For ``memmap``
    (default): the memmap is sorted by (symbol, date, time); we fancy-index each
    date's contiguous row range, so only the selected dates are materialised in
    Float32 — never the full pool. A ~400-date selection peaks ~12 GB, which
    fits a 16 GB Mac. For ``parquet`` (portable, for the Colab mount): read one
    partition per date. Either way the returned FitData is identical.
    """
    if manifest.get("format") == "parquet":
        return _build_fitdata_parquet(data_dir, manifest, date_list,
                                      group_override, aux_cols)
    K = manifest["K"]
    X_mm = np.memmap(data_dir / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    resp = np.load(data_dir / "resp.f16.npy", mmap_mode="r")
    if aux_cols is not None and aux_cols != manifest["aux_cols"]:
        # the memmap's aux matrix is frozen at precompute time — a requested
        # subset/reorder is fine; anything else needs a parquet pool exported
        # with --include-responders (or a fresh precompute)
        missing = [c for c in aux_cols if c not in manifest["aux_cols"]]
        if missing:
            raise ValueError(
                f"aux targets {missing} not in this memmap's aux matrix "
                f"({manifest['aux_cols']}). Use a parquet pool exported with "
                "--include-responders, or re-run precompute_dataset with "
                f"--aux-targets including them.")
        sel = [manifest["aux_cols"].index(c) for c in aux_cols]
        resp = resp[:, sel]
    y = np.load(data_dir / "y.f32.npy", mmap_mode="r")
    w = np.load(data_dir / "w.f32.npy", mmap_mode="r")
    symbols = np.load(data_dir / "symbols.i16.npy", mmap_mode="r")
    dates = np.load(data_dir / "dates.i16.npy", mmap_mode="r")
    times = np.load(data_dir / "times.i16.npy", mmap_mode="r")

    ranges = manifest["date_row_ranges"]
    idx_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for occ, d in enumerate(date_list):
        key = str(int(d))
        if key not in ranges:
            continue
        s, e = ranges[key]
        rows = np.arange(s, e, dtype=np.int64)
        idx_parts.append(rows)
        if group_override is not None:
            group_parts.append(np.full(len(rows), int(group_override[occ]), dtype=np.int64))
    idx = np.concatenate(idx_parts)

    X_win = np.ascontiguousarray(X_mm[idx]).astype(np.float32)
    resp_win = (np.ascontiguousarray(resp[idx]).astype(np.float32)
                if resp.shape[1] > 0 else np.zeros((len(idx), 0), np.float32))
    dates_out = (np.concatenate(group_parts) if group_override is not None
                 else np.ascontiguousarray(dates[idx]).astype(np.int64))
    return FitData(
        X=X_win,
        resp=resp_win,
        y=np.ascontiguousarray(y[idx]).astype(np.float32),
        w=np.ascontiguousarray(w[idx]).astype(np.float32),
        symbols=np.ascontiguousarray(symbols[idx]).astype(np.int64),
        dates=dates_out,
        times=np.ascontiguousarray(times[idx]).astype(np.int64),
    )


def walk_forward(pipe: FullPipeline, df: pl.DataFrame, valid_dates: np.ndarray, target_col: str):
    pipe = copy.deepcopy(pipe)
    # The recurrent predict path reshapes flat rows assuming TIME-MAJOR
    # (time, symbol) order — true for raw-path frames, FALSE for parquet
    # pool partitions (exported symbol-major). Sorting here is idempotent
    # for the raw path and fixes the parquet path (scrambled sequences:
    # static ~0, online ~half — measured on Kaggle 2026-07-19).
    df = df.sort([COL_DATE, "time_id", "symbol_id"])
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
    p.add_argument("--data", type=str, required=True, help="precompute output dir")
    # Training-date selection: either an explicit window, or a resample of the
    # pool. Resampling is the bagging path — each seed → a different draw.
    p.add_argument("--resample-mode", choices=["window", "subsample", "bootstrap"],
                   default="subsample",
                   help="how to pick this model's training dates from the pool")
    p.add_argument("--pool-lo", type=int, default=700,
                   help="low end of the training-date pool (8th place used 700)")
    p.add_argument("--pool-hi", type=int, default=None,
                   help="high end of the pool; default = manifest train_until")
    p.add_argument("--resample-frac", type=float, default=0.63,
                   help="fraction of the pool per model in subsample mode "
                        "(0.63 ≈ bootstrap's unique fraction)")
    # Window mode only:
    p.add_argument("--train-lo", type=int, default=None)
    p.add_argument("--train-hi", type=int, default=None)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument(
        "--embargo-dates", type=int, default=1,
        help="drop this many training dates immediately before valid-lo. The "
             "aux targets (responder_9/10) use cross-date FUTURE shifts of up "
             "to 40 steps (< 1 date), so the last training date otherwise leaks "
             "validation responders into its aux target. 1 date fully removes it.",
    )
    p.add_argument("--model", choices=list(MODEL_KWARGS), default="lstm_modelr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decay-halflife", type=float, default=0.0,
                   help="if >0, exp-decay training sample weights by date age with "
                        "this half-life (in dates): recent dates weigh more in the "
                        "loss. Eval weights are unchanged. 0 = off.")
    # Overrides of the per-model defaults (for GPU / longer-training runs).
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="dates per batch. 1 = per-date (default). Use 8-16 on GPU "
                        "to actually utilise it (the RNN barely benefits at bs=1).")
    p.add_argument("--patience", type=int, default=None, help="early-stopping patience")
    p.add_argument("--hidden", type=str, default=None,
                   help="comma-separated hidden sizes overriding the model "
                        "config (capacity probes, e.g. '128,128,128')")
    p.add_argument("--xsec-heads", type=int, default=None,
                   help="override the cross-sectional attention head count "
                        "(xsec models only). The attention embed dim follows "
                        "the LAST hidden size (ModelRBase: n_in = "
                        "hidden_sizes[-1]), so heads must divide it — e.g. "
                        "hidden 96 admits 2/4/8 heads, 128 admits 2/4/8/16.")
    p.add_argument("--aux-targets", type=str, default=None,
                   help="comma-separated aux responder columns (default: the "
                        "pool manifest's). Spread-aux example: "
                        "'responder_0,responder_2,responder_7,responder_8' — "
                        "the venue spreads are ~20x more predictable "
                        "(r2 tail R²=0.17), i.e. far stronger aux "
                        "supervision. Memmap pools only carry the aux cols "
                        "frozen at precompute; parquet pools exported with "
                        "--include-responders carry all nine.")
    p.add_argument("--tag", type=str, required=True, help="label for this bag member")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--watch", action="store_true",
                   help="attach the tslab SETOL TrainingWatcher: per-epoch "
                        "layer-spectrum snapshots, verdicts json + dashboard "
                        "png next to the member outputs")
    p.add_argument("--keep-epochs", action="store_true",
                   help="save every epoch's state dict under {out}/epochs/ — "
                        "enables SETOL-picked stopping + epoch bagging "
                        "post hoc (~4-8MB per epoch)")
    args = p.parse_args()

    data_dir = Path(args.data)
    manifest = load_manifest(data_dir)
    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Effective pool upper bound: never past the stats boundary, and embargoed
    # below valid-lo so no training date's future-shifted aux target reaches
    # into the validation range.
    pool_hi = args.pool_hi if args.pool_hi is not None else manifest["train_until"]
    embargo_cap = args.valid_lo - 1 - args.embargo_dates
    pool_hi = min(pool_hi, manifest["train_until"], embargo_cap)
    assert pool_hi < args.valid_lo, "pool overlaps validation"

    # --- Choose this bag member's training dates ---
    all_pool = np.array([int(d) for d in manifest["date_row_ranges"]], dtype=np.int64)
    pool = np.sort(all_pool[(all_pool >= args.pool_lo) & (all_pool <= pool_hi)])
    group_override = None
    if args.resample_mode == "window":
        assert args.train_lo is not None and args.train_hi is not None, \
            "window mode needs --train-lo and --train-hi"
        assert args.train_hi <= pool_hi, (
            f"train-hi {args.train_hi} past the embargoed pool bound {pool_hi} "
            f"(would leak future aux targets into validation)"
        )
        train_dates = np.arange(args.train_lo, args.train_hi + 1)
        train_dates = train_dates[np.isin(train_dates, pool)]
    else:
        train_dates = sample_dates(pool, args.seed, args.resample_mode, args.resample_frac)
        if args.resample_mode == "bootstrap":
            group_override = resolve_bootstrap_groups(train_dates)

    n_unique = len(np.unique(train_dates))
    print(f"[{args.tag}] model={args.model} seed={args.seed} "
          f"resample={args.resample_mode} pool=[{args.pool_lo},{pool_hi}]({len(pool)} dates) "
          f"→ {len(train_dates)} draws / {n_unique} unique  valid={args.valid_lo}..{args.valid_hi}",
          flush=True)

    aux_req = (args.aux_targets.split(",") if args.aux_targets
               else manifest["aux_cols"])

    # --- Build FitData for this bag's resampled dates (from memmap) ---
    t0 = time.time()
    train_fd = build_fitdata_for_dates(data_dir, manifest, train_dates,
                                       group_override, aux_cols=aux_req)
    print(f"  train FitData: {train_fd.X.shape[0]:,} rows × {train_fd.X.shape[1]} "
          f"in {time.time()-t0:.1f}s", flush=True)

    # --- Optional recency weighting: exp-decay the TRAIN sample weights so more
    # recent dates dominate the loss. Addresses the non-stationary conditional
    # (old dates encode a stale X->y mapping). Only training weights are scaled;
    # the eval weights (competition metric) are untouched. Half-life is in dates.
    if args.decay_halflife and args.decay_halflife > 0:
        anchor = int(train_dates.max())
        age = (anchor - train_fd.dates).astype(np.float64)   # 0 for the most recent date
        mult = np.power(0.5, age / args.decay_halflife).astype(np.float32)
        train_fd.w = train_fd.w * mult
        eff = float(mult.mean())
        print(f"  recency weighting: half-life={args.decay_halflife}d  "
              f"anchor_date={anchor}  mean weight-multiplier={eff:.3f}", flush=True)

    # --- Validation FitData for early stopping (from memmap) ---
    valid_dates_pool = np.arange(args.valid_lo, args.valid_hi + 1)
    valid_fd = build_fitdata_for_dates(data_dir, manifest, valid_dates_pool,
                                       aux_cols=aux_req)

    # --- Build model + fit ---
    cfg = Cfg()
    cfg.model_name = args.model
    kw = MODEL_KWARGS[args.model].copy()
    kw["seed"] = args.seed
    # CLI overrides — let the same script run longer/bigger on a GPU without
    # editing MODEL_KWARGS. Only override when the flag was actually passed.
    if args.device is not None:
        kw["device"] = args.device
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.batch_size is not None:
        kw["batch_size"] = args.batch_size
    if args.patience is not None:
        kw["early_stopping_patience"] = args.patience
    if args.hidden is not None:
        sizes = [int(s) for s in args.hidden.split(",")]
        kw["hidden_sizes"] = sizes
        kw["dropout_rates"] = [kw["dropout_rates"][0]] * len(sizes)
    if args.xsec_heads is not None:
        # Only meaningful where the model config already carries attention.
        # Silently accepting it elsewhere would "run" a capacity probe that
        # never existed, so fail loudly instead (mirrors the --hidden policy
        # of overriding, never inventing, kwargs).
        if "xsec_heads" not in kw:
            raise SystemExit(
                f"--xsec-heads only applies to xsec models "
                f"(*_modelr_xsec), not {args.model}")
        kw["xsec_heads"] = args.xsec_heads
    # nn.MultiheadAttention needs embed_dim % num_heads == 0, and the embed
    # dim is NOT independent: ModelRBase builds the attention on
    # n_in = hidden_sizes[-1]. So --hidden scales the attention width
    # automatically, and the only way the pair can break is a non-divisible
    # (last hidden, heads) combo — catch it here with the arithmetic spelled
    # out rather than deep inside torch's constructor.
    if kw.get("xsec_heads"):
        n_in, heads = kw["hidden_sizes"][-1], kw["xsec_heads"]
        if n_in % heads != 0:
            raise SystemExit(
                f"xsec attention: embed dim = last hidden size = {n_in} is "
                f"not divisible by xsec_heads = {heads} "
                f"(head_dim would be {n_in / heads:.2f}); adjust --hidden "
                f"or --xsec-heads")
    # TimeXer needs the endogenous channel indices = the lagged-responder
    # feature columns. Derive them from the precompute manifest so the model
    # config doesn't hard-code indices that shift with the feature set.
    if args.model == "timexer":
        endo = manifest.get("lag_col_indices", [])
        if not endo:
            raise SystemExit(
                "timexer needs lagged responders, but this precompute has none. "
                "Rebuild with: precompute_dataset.py --lagged-responders 0,1,...,8"
            )
        kw["endo_channels"] = endo
        print(f"  timexer endo_channels = {endo} ({manifest.get('lag_cols')})", flush=True)
    cfg.model_kwargs = kw
    pipe = make_pipeline(cfg, feature_cols=manifest["feature_cols"], aux_cols=aux_req)
    # Attach the frozen preprocessor so predict()/update() re-standardize raw
    # eval data with the same train-time stats.
    with (data_dir / "preprocessor.pkl").open("rb") as f:
        pipe.preprocessor = pickle.load(f)  # noqa: S301  trusted local artifact
    assert isinstance(pipe.preprocessor, Preprocessor)

    watch_state: dict = {}
    if args.watch or args.keep_epochs:
        # tslab is a sibling package (source import — no env mutation);
        # snapshot layer spectra each epoch via the model's callback hook
        if args.watch:
            sys.path.insert(0, str(TSLAB_SRC))
            from tslab.monitor import TrainingWatcher  # noqa: E402
        epochs_dir = out / "epochs" / args.tag
        if args.keep_epochs:
            epochs_dir.mkdir(parents=True, exist_ok=True)

        def _watch_cb(net, epoch, val_r2):
            if args.watch:
                w = watch_state.get("w")
                if w is None:
                    w = TrainingWatcher(net)
                    watch_state["w"] = w
                w.snapshot(epoch, val_metric=val_r2)
            if args.keep_epochs:
                torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                           epochs_dir / f"epoch_{epoch:02d}.pt")

        pipe.model.epoch_callback = _watch_cb

    t0 = time.time()
    pipe.model.fit(train_fd, valid_fd, verbose=True)
    fit_s = time.time() - t0
    print(f"  fit: {fit_s/60:.1f} min", flush=True)

    if watch_state.get("w") is not None:
        watcher = watch_state["w"]
        pipe.model.epoch_callback = None  # don't pickle the closure
        # one extra snapshot AFTER fit restored the best checkpoint: the
        # in-loop snapshots include patience-overshoot epochs, so without
        # this the "current" verdicts describe weights we discarded
        watcher.snapshot(getattr(pipe.model, "best_epoch", None) or -1)
        (out / f"{args.model}_{args.tag}_health.json").write_text(
            json.dumps({"verdicts_restored_model": watcher.verdicts(),
                        "history": watcher.history()},
                       indent=2, default=str))
        print(f"  [watch] (restored best) {watcher.summary()}", flush=True)
        try:
            from tslab.monitor import report as tslab_report
            health_png = out / f"{args.model}_{args.tag}_health.png"
            tslab_report(watcher, health_png)
            print(f"  [watch] dashboard → {health_png}", flush=True)
        except Exception as e:  # dashboard is a nice-to-have, never fatal
            print(f"  [watch] dashboard skipped: {e}", flush=True)

    ckpt = out / "checkpoints" / f"{args.model}_{args.tag}.pkl"
    with contextlib.suppress(NotImplementedError):
        pipe.save(ckpt)  # xgb raises — not applicable here

    # --- Eval on the validation tail via the in-memory path ---
    # Include several lookback dates so the 1000-step (~1 day) rolling features
    # are fully populated on the FIRST scored validation date. 3 dates (~2900
    # steps) comfortably covers the window; these lookback dates are used only
    # to warm the rolling stats, never scored (valid_dates filters to >=valid_lo).
    # (Not a leakage concern — lookback dates are past training data, available
    # at inference via the lag mechanism.)
    if manifest.get("format") == "parquet":
        # Portable path (Colab/Kaggle: no raw competition data available).
        # The pool partitions already hold the standardized features, target,
        # weight and aux columns — build the eval frame straight from them
        # and disable re-standardization (the values are already in model
        # space). Protocol note vs the raw path: the online-scaler
        # adaptation is unavailable here — eval uses frozen train-time
        # stats, a slightly more conservative variant of the walk.
        part_dir = data_dir / "data"
        frames = [
            pl.read_parquet(part_dir / f"date_id={d}" / "part.parquet")
            for d in range(args.valid_lo, args.valid_hi + 1)
            if (part_dir / f"date_id={d}" / "part.parquet").exists()
        ]
        df_eval = pl.concat(frames)
        pipe.preprocessor = None
    else:
        cfg_eval = Cfg()
        # CRITICAL: the eval frame must have the SAME feature columns as
        # training, including the lagged responders — otherwise the input
        # width won't match the trained model. Mirror the precompute's lag
        # config.
        cfg_eval.lagged_responders = list(manifest.get("lagged_responders", []))
        # dates needed to fill a rolling_window-step trailing window, +1 margin
        lookback = (cfg_eval.rolling_window + 967) // 968 + 1
        cfg_eval.min_date_id = args.valid_lo - lookback
        cfg_eval.max_date_id = args.valid_hi
        df_eval = prepare_dataset(cfg_eval, storage_precision="float32")
    valid_dates = np.arange(args.valid_lo, args.valid_hi + 1)
    valid_dates = valid_dates[np.isin(valid_dates,
                                      df_eval.select(pl.col(COL_DATE).unique()).to_series().to_numpy())]

    # static (freeze refit)
    static = copy.deepcopy(pipe)
    if hasattr(static.model, "lr_refit"):
        static.model.lr_refit = 0.0
    p_s, y_s, w_s = walk_forward(static, df_eval, valid_dates, cfg.target)
    r2_static = r2_weighted(y_s, p_s, w_s)
    print(f"  static R² = {r2_static:+.5f}", flush=True)

    # online refit
    p_o, y_o, w_o = walk_forward(pipe, df_eval, valid_dates, cfg.target)
    r2_online = r2_weighted(y_o, p_o, w_o)
    print(f"  online R² = {r2_online:+.5f}", flush=True)

    base = f"{args.model}_{args.tag}"
    np.savez_compressed(out / "preds" / f"{base}_static.npz",
        preds=p_s.astype(np.float32), y=y_s.astype(np.float32), w=w_s.astype(np.float32),
        valid_dates=valid_dates.astype(np.int32))
    np.savez_compressed(out / "preds" / f"{base}_online.npz",
        preds=p_o.astype(np.float32), y=y_o.astype(np.float32), w=w_o.astype(np.float32),
        valid_dates=valid_dates.astype(np.int32))

    res = {
        "tag": args.tag, "model": args.model, "seed": args.seed,
        "resample_mode": args.resample_mode,
        "pool_lo": args.pool_lo, "pool_hi": int(pool_hi),
        "resample_frac": args.resample_frac,
        "n_train_dates_unique": int(n_unique),
        "train_lo": args.train_lo, "train_hi": args.train_hi,
        "valid_lo": args.valid_lo, "valid_hi": args.valid_hi,
        "n_train_rows": int(train_fd.X.shape[0]),
        "fit_s": fit_s, "r2_static": r2_static, "r2_online": r2_online,
        "timestamp": datetime.now().isoformat(),
    }
    (out / f"{base}.json").write_text(json.dumps(res, indent=2))
    print(f"  written → {out / f'{base}.json'}", flush=True)


if __name__ == "__main__":
    main()
