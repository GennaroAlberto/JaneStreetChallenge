"""Long-running benchmark: train every model in the registry on the same
single-fold data slice, score it under both static and online-refit eval.

Designed to run unattended for 15-20 hours. Each model writes its result to
its own JSON file under ``artifacts/bench/<run>/<model>.json`` so partial
progress survives crashes; a final ``results.md`` aggregates everything.

Usage::

    uv run python scripts/bench.py                         # default 500-date run
    uv run python scripts/bench.py --min-date 1399 --max-date 1698 --valid 60
    uv run python scripts/bench.py --only gru_modelr,sig_transformer
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from janestreet.config import COL_DATE, COL_WEIGHT, Cfg
from janestreet.pipeline import make_pipeline, prepare_dataset
from janestreet.training.metrics import r2_weighted

# Filter noisy warnings from polars-on-NaN-only columns
warnings.filterwarnings("ignore")

# Per-model configurations — sized so the whole sweep finishes ~15-18h on
# 400 training dates / 100 validation dates on a Mac laptop CPU.
#
# Spec keys are *experiment* names; each maps to (base_model_name, kwargs).
# This lets us add variants (e.g. "lstm_modelr_h128") without touching the
# model registry. The bench picks the spec by name, then asks the registry
# to build the base model with the per-variant kwargs.
MODEL_SPECS: dict[str, dict] = {
    # XGB: hist tree method + max_bin to cap memory footprint of the
    # internal QuantileDMatrix on a 16-GB Mac.
    "xgb": dict(
        _base="xgb",
        n_estimators=3000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        max_bin=128, tree_method="hist",
        early_stopping_rounds=50, n_jobs=1,
    ),
    "mlp": dict(
        _base="mlp",
        hidden_sizes=[512, 512, 256], dropout=0.1,
        lr=1e-3, weight_decay=1e-2,
        epochs=50, batch_size=16384,
        early_stopping_patience=5, device="cpu",
    ),
    "gru": dict(
        _base="gru",
        hidden_sizes=[96, 96], dropout_rates=[0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2,
        epochs=20, batch_size=1, grad_clip=1.0,
        early_stopping_patience=5, lr_refit=3e-4, device="cpu",
    ),
    # NOTE: tightened to fit the 15-20h budget after seeing GRU take 90 min
    # at 20 epochs. ModelR is a 4-branch (3 layers each) variant — ~6x compute
    # vs the single-branch GRU above — so we halve epochs to keep ~6h.
    "gru_modelr": dict(
        _base="gru_modelr",
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2,
        epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4,
        device="cpu", num_aux=4,
    ),
    "lstm_modelr": dict(
        _base="lstm_modelr",
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        lr=1e-3, weight_decay=1e-2,
        epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4,
        device="cpu", num_aux=4,
    ),
    # Mamba (selective SSM). Third architecture family — data-dependent
    # linear dynamics, unbounded latent. Compute is O(T · d_state · d_inner)
    # per layer, similar to LSTM in wall-clock on our T=968 sequences.
    # Refit-LR mirrors the RNN family (Mamba behaves closer to RNN than
    # transformer under refit — smooth gradient, small param count).
    "mamba": dict(
        _base="mamba",
        d_model=96, n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=1e-3, device="cpu",
    ),
    # Transformer attention is O(T^2) per layer; the 968-step sequence makes
    # each epoch ~12x more expensive than GRU on this slice. Cut hard.
    "transformer": dict(
        _base="transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
    ),
    "sig_transformer": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        # 6 channels × depth=2 + time → sig_dim = 7 + 49 = 56 extra features
        signature_channels=[0, 1, 2, 3, 4, 5],
        signature_window=16, signature_depth=2, signature_hurst=0.1,
    ),
    # ============================================================
    # run-2 variants
    # ============================================================
    # Phase B: bigger lstm_modelr (hidden 128 vs 96; 1.78x param count).
    "lstm_modelr_h128": dict(
        _base="lstm_modelr",
        hidden_sizes=[128, 128, 128], dropout_rates=[0.1, 0.1, 0.1],
        lr=1e-3, weight_decay=1e-2,
        epochs=10, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4,
        device="cpu", num_aux=4,
    ),
    # Phase C₁: richer Volterra signature (depth=3 → much larger sig_dim).
    "sig_transformer_d3": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        signature_channels=[0, 1, 2, 3, 4, 5],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
    ),
    # Phase F: signature on a *deliberately selected* subset of raw features.
    # Picked by scripts/select_signature_channels.py with a composite
    # score (target-corr + path-richness + autocorr) and a 0.5 redundancy
    # cap on inter-channel |corr|. Five of six picks coincide with
    # Volkova's hand-curated CORR list; feature_16 is the standout new find.
    # Channel indices below are positions in FeatureBuilder.feature_columns().
    # NOTE: this selector underperformed in Phase F — univariate target-corr
    # is the wrong filter for signature inputs. Kept for the record.
    "sig_transformer_selected": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        # feature_06, feature_16, feature_36, feature_19, feature_56, feature_04
        signature_channels=[6, 13, 33, 16, 53, 4],
        signature_window=16, signature_depth=2, signature_hurst=0.1,
    ),
    # Phase G: signature on channels selected by AR(7) ridge fit on each
    # raw feature's lags predicting responder_6 — picked by
    # scripts/select_signature_channels_ar.py. K=8, redundancy cap=0.5.
    # Channels = [feature_16, 66, 45, 36, 73, 25, 23, 59] →
    # indices = [13, 63, 42, 33, 70, 22, 20, 56].
    "sig_transformer_ar_k8": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=2, signature_hurst=0.1,
    ),
    # Phase I (planned): log-signature at the project-standard depth=3 on
    # AR-selected channels. With d=9 (8 channels + time):
    #   * standard signature dim   = 9 + 81 + 729 = 819
    #   * log-signature dim (here) = 9 + 36 + 729 = 774  (5% smaller)
    # The depth-2 → depth-3 saving comes mostly from the L₂ antisym
    # projection (81 → 36); the L₃ component is kept as a full d³ tensor
    # for now (see _logsig_dim TODO — proper Lyndon projection would cut
    # this further to 9 + 36 + 240 = 285). The de-correlation argument
    # still holds at L₃ since the shuffle products are subtracted out.
    # lr_refit=3e-5 is the Phase H winner for the depth-2 ar_k8 ckpt; at
    # depth=3 the gradient distribution shifts so this may need a fresh
    # sweep — bench will reveal it. Not run yet.
    "sig_transformer_logsig_ar_k8_d3": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="log_signature",
    ),
    # Standard signature at depth=3 on the AR-selected channels — direct
    # apples-to-apples for the log-signature variant above. sig_dim=819.
    "sig_transformer_ar_k8_d3": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
    ),
    # Minimal (Lyndon-basis) log-signature via iisignature. Realises the
    # full dim reduction at depth=3: 9 + 36 + 240 = 285 dims (vs the naive
    # log-sig's 774 and the classical signature's 819). Every output
    # component is linearly independent — no shuffle / Jacobi redundancies
    # remain. Requires `uv sync --extra logsig-minimal` (iisignature).
    # If iisignature isn't installed the bench's pipeline build will fail
    # fast with a clear ImportError at SignatureBlock construction.
    "sig_transformer_logsig_min_ar_k8_d3": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="log_signature_minimal",
    ),
    # ---- MLP + signature family (stateless in time, non-recurrent) ----
    # Purpose: an ensemble diversity candidate. The MLP head can only see
    # what the signature encodes about the recent path, plus the current
    # feature vector. Very different geometry from the RNN implicit-path
    # story — hopefully uncorrelated enough to lift the 3-way blend.
    # lr_refit=3e-5 mirrors the Phase H winner for sig_transformer.
    "mlp_sig_ar_k8_d3": dict(
        _base="mlp_sig",
        hidden_sizes=[512, 256, 128], dropout=0.1,
        lr=1e-3, weight_decay=1e-2,
        epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="signature",
    ),
    "mlp_logsig_ar_k8_d3": dict(
        _base="mlp_sig",
        hidden_sizes=[512, 256, 128], dropout=0.1,
        lr=1e-3, weight_decay=1e-2,
        epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="log_signature",
    ),
    "mlp_logsig_min_ar_k8_d3": dict(
        _base="mlp_sig",
        hidden_sizes=[512, 256, 128], dropout=0.1,
        lr=1e-3, weight_decay=1e-2,
        epochs=15, batch_size=1, grad_clip=1.0,
        early_stopping_patience=4, lr_refit=3e-5, device="cpu",
        signature_channels=[13, 63, 42, 33, 70, 22, 20, 56],
        signature_window=16, signature_depth=3, signature_hurst=0.1,
        signature_mode="log_signature_minimal",
    ),
    # Phase C₂: longer signature window (more memory, same depth).
    "sig_transformer_w64": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=3e-4, device="cpu",
        signature_channels=[0, 1, 2, 3, 4, 5],
        signature_window=64, signature_depth=2, signature_hurst=0.1,
    ),
    # Phase C₀: control — same architecture as run-1's sig_transformer, but
    # with refit OFF. If this beats run-1's online R², the online regression
    # is real and is caused by the refit step (not the architecture).
    "sig_transformer_norefit": dict(
        _base="sig_transformer",
        d_model=96, n_heads=4, n_layers=2, ff_mult=2, dropout=0.1,
        lr=5e-4, weight_decay=1e-2,
        epochs=8, batch_size=1, grad_clip=1.0,
        early_stopping_patience=3, lr_refit=0.0, device="cpu",
        signature_channels=[0, 1, 2, 3, 4, 5],
        signature_window=16, signature_depth=2, signature_hurst=0.1,
    ),
}


def n_params(model_wrapper) -> int:
    inner = getattr(model_wrapper, "model", None)
    if inner is None or not isinstance(inner, torch.nn.Module):
        return -1
    return sum(p.numel() for p in inner.parameters())


def eval_online_refit(
    pipe, df: pl.DataFrame, valid_dates: np.ndarray, target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward eval: predict each valid date, then refit on its responder_6 truth.

    Returns (preds, y_true, weights) concatenated in date order.
    """
    pipe = copy.deepcopy(pipe)  # don't mutate caller
    preds_list, y_list, w_list = [], [], []
    for i, dt in enumerate(valid_dates):
        day = df.filter(pl.col(COL_DATE) == dt)
        if i > 0:
            prev = df.filter(pl.col(COL_DATE) == int(dt) - 1)
            if prev.height > 0:
                pipe.update(prev)
        preds_list.append(pipe.predict(day))
        y_list.append(day.select(target_col).to_series().to_numpy())
        w_list.append(day.select(COL_WEIGHT).to_series().to_numpy())
    return (
        np.concatenate(preds_list),
        np.concatenate(y_list),
        np.concatenate(w_list),
    )


def eval_static(
    pipe, df: pl.DataFrame, valid_dates: np.ndarray, target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_va = df.filter(pl.col(COL_DATE).is_in(valid_dates))
    # Predict each date individually (the sequence models need n_times per date).
    preds_list, y_list, w_list = [], [], []
    for dt in valid_dates:
        day = df_va.filter(pl.col(COL_DATE) == dt)
        preds_list.append(pipe.predict(day))
        y_list.append(day.select(target_col).to_series().to_numpy())
        w_list.append(day.select(COL_WEIGHT).to_series().to_numpy())
    return (
        np.concatenate(preds_list),
        np.concatenate(y_list),
        np.concatenate(w_list),
    )


def run_one_model(
    name: str,
    cfg: Cfg,
    df_train: pl.DataFrame,
    df_valid_plus: pl.DataFrame,
    train_dates: np.ndarray,
    valid_dates: np.ndarray,
    out_dir: Path,
    log,
) -> dict:
    """Train one model, evaluate it twice (static + online-refit), write JSON.

    ``df_train`` covers train_dates; ``df_valid_plus`` covers valid_dates + the
    last training date (kept so the online-refit eval can update on the day
    before the first valid date). Splitting up-front means we only carry
    ~9 GB through training instead of ~16 GB.
    """
    kw = MODEL_SPECS[name].copy()
    base_model = kw.pop("_base", name)  # variant_name → registry model_name
    cfg.model_name = base_model
    cfg.model_kwargs = kw

    log(f"\n{'=' * 70}\n[{datetime.now():%H:%M:%S}] starting model: {name} (base: {base_model})")
    log(f"  kwargs: {json.dumps(kw, default=str)}")

    df_va = df_valid_plus.filter(pl.col(COL_DATE).is_in(valid_dates))
    n_tr_rows, n_va_rows = df_train.height, df_va.height
    log(f"  rows: train={n_tr_rows}  valid={n_va_rows}")

    pipe = make_pipeline(cfg)

    # --- Train ---
    t0 = time.time()
    try:
        pipe.fit(df_train, df_va, verbose=False)
    except Exception as e:
        log(f"  FIT FAILED: {e!r}")
        log(traceback.format_exc())
        return {"model": name, "error": repr(e), "trace": traceback.format_exc()}
    fit_s = time.time() - t0

    # Caller still holds ``df_train``; we can't drop it from here. The numpy
    # arrays inside the model's FitData are float32 and survive eval, but
    # any transient buffers from .to_numpy() during fit should be GC-able.
    gc.collect()

    params = n_params(pipe.model)
    log(f"  fit time = {fit_s:.1f}s  params = {params}")

    # --- Checkpoint (so we can replay eval / LR sweep later) ---
    # XGB intentionally skipped — see FullPipeline.save docstring.
    ckpt_path = out_dir / "checkpoints" / f"{name}.pkl"
    try:
        pipe.save(ckpt_path)
        log(f"  checkpoint saved → {ckpt_path.relative_to(out_dir.parent)}")
    except NotImplementedError:
        log(f"  checkpoint skipped for {name} (XGB; using .npz preds instead)")
    except Exception as e:  # noqa: BLE001
        log(f"  WARN: checkpoint save failed: {e!r}")

    # --- Static eval ---
    t0 = time.time()
    p_s, y_s, w_s = eval_static(pipe, df_valid_plus, valid_dates, cfg.target)
    eval_static_s = time.time() - t0
    r2_static = r2_weighted(y_s, p_s, w_s)
    log(f"  static eval = {eval_static_s:.1f}s  R²={r2_static:+.5f}")

    # Save prediction arrays alongside the JSON — these power downstream
    # ensembling without needing to reload checkpoints (works for XGB too).
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_dir / f"{name}_static.npz",
        preds=p_s.astype(np.float32),
        y=y_s.astype(np.float32),
        w=w_s.astype(np.float32),
        valid_dates=np.asarray(valid_dates, dtype=np.int32),
    )

    # --- Online-refit eval ---
    if getattr(pipe.model, "lr_refit", 0.0) > 0.0:
        t0 = time.time()
        p_o, y_o, w_o = eval_online_refit(pipe, df_valid_plus, valid_dates, cfg.target)
        eval_online_s = time.time() - t0
        r2_online = r2_weighted(y_o, p_o, w_o)
        log(f"  online-refit eval = {eval_online_s:.1f}s  R²={r2_online:+.5f}")
        np.savez_compressed(
            preds_dir / f"{name}_online.npz",
            preds=p_o.astype(np.float32),
            y=y_o.astype(np.float32),
            w=w_o.astype(np.float32),
            valid_dates=np.asarray(valid_dates, dtype=np.int32),
        )
    else:
        eval_online_s, r2_online = None, None

    res = {
        "model": name,
        "base_model": base_model,
        "n_params": params,
        "fit_s": fit_s,
        "eval_static_s": eval_static_s,
        "eval_online_s": eval_online_s,
        "r2_static": r2_static,
        "r2_online": r2_online,
        "kwargs": kw,
        "n_train_dates": int(len(train_dates)),
        "n_valid_dates": int(len(valid_dates)),
        "n_train_rows": int(n_tr_rows),
        "n_valid_rows": int(n_va_rows),
        "timestamp": datetime.now().isoformat(),
    }
    with (out_dir / f"{name}.json").open("w") as f:
        json.dump(res, f, indent=2, default=str)

    # Release model + intermediate tensors before the next iteration so
    # we don't stack two model footprints in memory on a 16 GB Mac.
    del pipe, df_va, p_s, y_s, w_s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def write_markdown(out_dir: Path, results: list[dict]) -> None:
    lines = [
        "# Benchmark results",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "| Model | params | fit (s) | R² static | R² online | Δ online |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: (x.get("r2_online") or x.get("r2_static") or -9), reverse=True):
        if "error" in r:
            lines.append(f"| {r['model']} | — | — | **error**: {r['error']} | | |")
            continue
        r2s = r["r2_static"]
        r2o = r["r2_online"]
        delta = (r2o - r2s) if (r2o is not None and r2s is not None) else None
        lines.append(
            f"| {r['model']} | {r['n_params']:,} | {r['fit_s']:.0f} | "
            f"{r2s:+.5f} | {(f'{r2o:+.5f}' if r2o is not None else '—')} | "
            f"{(f'{delta:+.5f}' if delta is not None else '—')} |"
        )
    lines.append("")
    lines.append("R² = competition weighted-R² (1 − ΣwΔ² / Σwy²). Higher is better. The 8th place LB score was 0.0112.")
    (out_dir / "results.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-date", type=int, default=1399)
    p.add_argument("--max-date", type=int, default=1698)
    p.add_argument("--valid", type=int, default=100, help="dates in validation tail")
    p.add_argument("--only", type=str, default=None, help="comma-separated subset of model names")
    p.add_argument("--out", type=str, default=None, help="output dir under artifacts/bench (default = timestamp)")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    cfg = Cfg()
    cfg.min_date_id = args.min_date
    cfg.max_date_id = args.max_date

    run_id = args.out or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.artifacts_root / "bench" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    log(f"Run id: {run_id}  out={out_dir}")
    log(f"date range: {args.min_date}..{args.max_date}  valid_tail={args.valid}")

    t0 = time.time()
    df = prepare_dataset(cfg)
    log(f"data loaded: {df.height:,} rows × {df.width} cols in {time.time()-t0:.1f}s")
    dates = np.sort(df.select(pl.col(COL_DATE).unique()).to_series().to_numpy())
    valid_dates = dates[-args.valid:]
    train_dates = dates[: -args.valid]
    log(f"train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)} dates)")
    log(f"valid: {valid_dates[0]}..{valid_dates[-1]} ({len(valid_dates)} dates)")

    # Materialise the slices we actually need and drop the full frame.
    # The full frame plus a fit-time numpy copy peaks >16 GB on the 350-day
    # slice — this trims ~10 GB without losing anything we use downstream.
    # For the online-refit lookback we also keep the last training date.
    df_train = df.filter(pl.col(COL_DATE).is_in(train_dates))
    online_dates = np.concatenate([train_dates[-1:], valid_dates])
    df_valid_plus = df.filter(pl.col(COL_DATE).is_in(online_dates))
    del df
    gc.collect()
    log(
        f"slices held: train={df_train.height:,} rows  "
        f"valid+1={df_valid_plus.height:,} rows"
    )

    selected = args.only.split(",") if args.only else list(MODEL_SPECS)
    log(f"models: {selected}")

    results: list[dict] = []
    for name in selected:
        out_path = out_dir / f"{name}.json"
        if args.skip_existing and out_path.exists():
            log(f"[{datetime.now():%H:%M:%S}] {name} already done — loading cached result")
            results.append(json.loads(out_path.read_text()))
            write_markdown(out_dir, results)
            continue
        res = run_one_model(
            name, cfg, df_train, df_valid_plus, train_dates, valid_dates, out_dir, log,
        )
        results.append(res)
        # Re-render the leaderboard after every model so partial progress is visible
        write_markdown(out_dir, results)

    log(f"\nALL DONE. total = {(time.time()-t0)/3600:.2f} h")
    write_markdown(out_dir, results)


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
    main()
