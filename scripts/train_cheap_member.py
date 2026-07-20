"""Cheap ModelR ensemble member — the [64,64] config found in the
innovation-aux A/B (its A arm hit +0.00725 online at 200d in ~16 min of
fit), scaled to the 280d production window for seed bagging.

Standard Volkova aux targets (r7, r8, forward-SMA synthetics); the walk
drives the nn.Module directly with symbol-major day tensors, so the
online predictions land in MEMMAP row order. They are saved under the
``pred`` key and slot into ``regen_ensemble_blend.py --xgb-variants``
(any stream in that dir is treated as memmap-ordered — no permutation).

Usage
-----
    uv run python scripts/train_cheap_member.py --seed 42 \\
        --out artifacts/bench/pool_rebuild_280/preds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from aux_innovation_lab import (  # noqa: E402
    r2_weighted, rows_for_dates, synth_forward, walk_forward_direct,
)
from responder_chain import load_raw_responders  # noqa: E402

from janestreet.models.base import FitData  # noqa: E402
from janestreet.models.recurrent import RecurrentModel  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1318)
    p.add_argument("--train-hi", type=int, default=1597)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=str, default="64,64",
                   help="comma-separated LSTM hidden sizes (capacity ladder)")
    p.add_argument("--pool-lo", type=int, default=None,
                   help="with --n-draws: subsample-bagging pool lower bound")
    p.add_argument("--n-draws", type=int, default=None,
                   help="draw this many dates (no replacement, seeded) from "
                        "[pool-lo, train-hi] instead of the contiguous window")
    p.add_argument("--decay-halflife", type=float, default=0.0,
                   help="train-side sample-weight decay 0.5^(age/halflife) "
                        "in days from train-hi; 0 = off")
    p.add_argument("--watch", action="store_true",
                   help="attach the tslab SETOL watcher: per-epoch layer "
                        "spectra + val recorded to {tag}_health.json")
    p.add_argument("--keep-epochs", action="store_true",
                   help="save EVERY epoch's state dict to {tag}_epochs/ — "
                        "enables post-hoc stopping-rule tests and epoch "
                        "bagging without retraining (small nets: ~3MB/epoch)")
    p.add_argument("--aux-mode", choices=["sma", "nowcast"], default="sma",
                   help="aux targets: Volkova forward-SMA synthetics (sma) "
                        "or realized/nowcast heads — the just-completed "
                        "r6 SMA and r8 window, backward shifts (nowcast)")
    p.add_argument("--out", default="artifacts/bench/pool_rebuild_280/preds")
    args = p.parse_args()

    hidden = [int(s) for s in args.hidden.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = ("lstm64" if hidden == [64, 64]
            else f"lstm{'x'.join(map(str, hidden))}")
    mods = ""
    if args.n_draws:
        mods += f"_sub{args.n_draws}"
    if args.decay_halflife:
        mods += f"_hl{args.decay_halflife:g}"
    if args.aux_mode != "sma":
        mods += f"_{args.aux_mode}"
    tag = f"{base}{mods}_s{args.seed}"

    def log(msg):
        line = str(msg) if str(msg).endswith("\n") else str(msg) + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with (out / f"{tag}.log").open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    ranges = manifest["date_row_ranges"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    y_all = np.load(data / "y.f32.npy", mmap_mode="r")
    w_all = np.load(data / "w.f32.npy", mmap_mode="r")
    sym = np.load(data / "symbols.i16.npy", mmap_mode="r")
    dat = np.load(data / "dates.i16.npy", mmap_mode="r")
    tim = np.load(data / "times.i16.npy", mmap_mode="r")

    def synth_backward(col, k):
        """within-(date,symbol)-block backward shift: nowcast target x[t-k]."""
        n = len(col)
        b = col.reshape(n // 968, 968).astype(np.float64)
        sh = np.zeros_like(b)
        sh[:, k:] = b[:, :-k]
        return sh.reshape(n).astype(np.float32)

    def aux_from(R):
        if args.aux_mode == "nowcast":
            return np.column_stack([
                R[:, 7], R[:, 8],
                synth_backward(R[:, 6], 20),   # the just-completed r6 SMA
                synth_backward(R[:, 8], 4),    # the just-completed r8 window
            ]).astype(np.float32)
        return np.column_stack([
            R[:, 7], R[:, 8],
            synth_forward(R[:, 8], [4]),
            synth_forward(R[:, 6], [20, 40]),
        ]).astype(np.float32)

    # --- training-date selection: contiguous window or seeded pool draw ----
    if args.n_draws:
        pool_lo = args.pool_lo if args.pool_lo is not None else args.train_lo
        tr_full = rows_for_dates(ranges, pool_lo, args.train_hi)
        pool_dates = np.array(sorted(int(d) for d in ranges
                                     if pool_lo <= int(d) <= args.train_hi))
        rng = np.random.default_rng(args.seed)
        drawn = np.sort(rng.choice(pool_dates, size=args.n_draws, replace=False))
        mask = np.isin(np.asarray(dat[tr_full]), drawn)
        tr = tr_full[mask]
        R_pool = load_raw_responders(pool_lo, args.train_hi)
        resp_tr = aux_from(R_pool[mask])
        del R_pool
        log(f"{tag}: {args.n_draws} draws from pool [{pool_lo},{args.train_hi}] "
            f"({len(pool_dates)} dates)")
    else:
        tr = rows_for_dates(ranges, args.train_lo, args.train_hi)
        resp_tr = aux_from(load_raw_responders(args.train_lo, args.train_hi))
    va = rows_for_dates(ranges, args.valid_lo, args.valid_hi)
    resp_va = aux_from(load_raw_responders(args.valid_lo, args.valid_hi))
    log(f"{tag}: train rows {len(tr):,}  valid rows {len(va):,}")

    def fitdata(rows_, resp, decay=False):
        X = np.nan_to_num(np.ascontiguousarray(X_mm[rows_]).astype(np.float32))
        w_out = w_all[rows_].astype(np.float32)
        if decay and args.decay_halflife > 0:
            age = (args.train_hi - dat[rows_].astype(np.float32))
            w_out = w_out * np.power(0.5, age / args.decay_halflife).astype(np.float32)
        return FitData(X=X, resp=resp,
                       y=y_all[rows_].astype(np.float32),
                       w=w_out,
                       symbols=sym[rows_].astype(np.int64),
                       dates=dat[rows_].astype(np.int64),
                       times=tim[rows_].astype(np.int64))

    model = RecurrentModel(
        model_type="lstm", aux_branches=True, num_aux=4,
        hidden_sizes=hidden, dropout_rates=[0.1] * len(hidden),
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, batch_size=1, epochs=args.epochs,
        early_stopping_patience=3, grad_clip=1.0, lr_refit=1e-3,
        seed=args.seed, device=args.device)
    watch_state: dict = {}
    if args.watch or args.keep_epochs:
        import os as _os
        import torch as _torch
        tslab_src = Path(_os.environ.get(
            "TSLAB_SRC", Path.home() / "Desktop" / "code" / "tslab" / "src"))
        epochs_dir = out / f"{tag}_epochs"
        if args.keep_epochs:
            epochs_dir.mkdir(exist_ok=True)

        def _cb(net, epoch, val_r2):
            if args.watch:
                w = watch_state.get("w")
                if w is None:
                    sys.path.insert(0, str(tslab_src))
                    from tslab.monitor import TrainingWatcher
                    w = TrainingWatcher(net)
                    watch_state["w"] = w
                w.snapshot(epoch, val_metric=val_r2)
            if args.keep_epochs:
                _torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                            epochs_dir / f"epoch_{epoch:02d}.pt")

        model.epoch_callback = _cb

    t0 = time.time()
    model.fit(fitdata(tr, resp_tr, decay=True), fitdata(va, resp_va), verbose=True)
    log(f"fit {((time.time() - t0) / 60):.1f} min")
    if watch_state.get("w") is not None:
        model.epoch_callback = None
        wtch = watch_state["w"]
        wtch.snapshot(getattr(model, "best_epoch", None) or -1)  # restored best
        (out / f"{tag}_health.json").write_text(json.dumps(
            {"verdicts_restored_model": wtch.verdicts(),
             "history": wtch.history()}, indent=2, default=str))
        log(f"[watch] (restored best) {wtch.summary()}")
    # save BEFORE the walk: walk_forward_direct refits daily through the
    # tail, so post-walk weights are tail-contaminated (never rewalk them —
    # measured +0.015 "scores" from exactly that mistake)
    import torch
    torch.save(model.model.state_dict(), out / f"{tag}.pt")

    p_, y_, w_ = walk_forward_direct(
        model, X_mm, ranges, y_all, w_all, K,
        args.valid_lo, args.valid_hi, model.device, log)
    r2 = r2_weighted(y_, p_, w_)
    log(f"{tag} online R² = {r2:+.5f}")
    np.savez_compressed(out / f"{tag}.npz", pred=p_.astype(np.float32),
                        y=y_.astype(np.float32), w=w_.astype(np.float32))
    (out / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "seed": args.seed, "hidden": hidden,
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "r2_online": r2, "fit_s": time.time() - t0,
        "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
