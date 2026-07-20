"""Log-signature of the AE latent path — signatures, resurrected.

The raw-feature signature models died of redundancy with the RNNs'
hidden state (see WHAT_DIDNT_WORK.md §1) and of channel selection: picking
8 of 130 noisy columns was the real bottleneck. This experiment removes
both objections at once:

  * channels = the AE bottleneck (8 dims, denoised, optionally
    target-aware) — the channel-selection problem disappears;
  * consumer = XGBoost (stateless) — the one family whose ensemble seat
    is NOT already carrying intraday path information, so any path
    geometry the signature encodes is genuinely new to it.

Per (symbol, day) we compute, at every timestep, the depth-2 log-signature
(level 1 + Lévy areas, optional Volterra/Hurst reweighting) of the trailing
window of the 8-dim latent path, then ablate on the standard harness:

  base           — the pool's 134 columns
  base+sig       — + the ~45 log-signature columns
  base+z+sig     — + the raw latents too (is the path geometry additive
                    to the pointwise latent state?)

Usage
-----
    uv run python scripts/sig_latents.py \\
        --latents artifacts/bench/ae_lab_sup/latents.npz \\
        --data artifacts/precomputed/pool700_lags \\
        --out artifacts/bench/sig_latents
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from janestreet.theory.signatures import SignatureBlock

N_TIMES = 968


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi):
    ranges = manifest["date_row_ranges"]
    dates = [d for d in range(lo, hi + 1) if str(d) in ranges]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64) for d in dates]
    return dates, np.concatenate(parts)


def sig_features(z, dates, manifest, block, log):
    """Per-timestep trailing-window log-signature of the latent path.

    ``z`` is (N_range, 8) aligned with the concatenated per-date row ranges
    (date-major, symbol-major within date, times contiguous).
    """
    ranges = manifest["date_row_ranges"]
    torch.set_num_threads(2)
    out = np.empty((len(z), block.sig_dim), dtype=np.float32)
    cursor = 0
    t0 = time.time()
    for i, d in enumerate(dates):
        r0, r1 = ranges[str(d)]
        n = r1 - r0
        if n % N_TIMES:
            raise ValueError(f"date {d}: {n} rows not divisible by {N_TIMES}")
        s = n // N_TIMES
        zd = torch.from_numpy(z[cursor:cursor + n].reshape(s, N_TIMES, -1))
        with torch.no_grad():
            full = block(zd)                       # (s, T, 8 + sig_dim)
        out[cursor:cursor + n] = (
            full[..., zd.shape[-1]:].reshape(n, block.sig_dim).numpy()
        )
        cursor += n
        if (i + 1) % 25 == 0:
            log(f"  sig: {i + 1}/{len(dates)} dates ({time.time() - t0:.0f}s)")
    assert cursor == len(z)
    return out


def fit_eval(Xtr, ytr, wtr, Xva, yva, wva, n_estimators, seed):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", required=True,
                   help="latents.npz from an ae_features.py run")
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mode", default="log_signature",
                   choices=["signature", "log_signature", "log_signature_minimal"])
    p.add_argument("--hurst", type=float, default=0.1,
                   help="Volterra reweighting exponent; negative to disable")
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--row-step", type=int, default=2,
                   help="subsample XGB fit/score rows by this step. The "
                        "full-row ablation peaks ~16 GB (134-col matrix + "
                        "hstack copies) and OOMs a 16 GB machine alongside "
                        "other jobs; step 2 quarters the peak. Deltas stay "
                        "internally valid (all variants share the rows), "
                        "but the base score is not comparable to full-row "
                        "harness numbers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/sig_latents")
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
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    blob = np.load(args.latents)
    z_tr, z_va = blob["z_train"], blob["z_valid"]
    latent = z_tr.shape[1]

    tr_dates, tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    if len(tr) != len(z_tr) or len(va) != len(z_va):
        raise ValueError(
            f"latents/rows mismatch: {len(z_tr)} vs {len(tr)} train, "
            f"{len(z_va)} vs {len(va)} valid — was the AE run on the same "
            "date range?")
    if not (np.array_equal(blob["rows_train"], tr)
            and np.array_equal(blob["rows_valid"], va)):
        raise ValueError("latent row indices do not match the harness rows")

    hurst = None if args.hurst < 0 else args.hurst
    block = SignatureBlock(channels=list(range(latent)), window=args.window,
                           depth=args.depth, hurst=hurst, mode=args.mode)
    log(f"latents: {args.latents} ({latent} dims)  sig: mode={args.mode} "
        f"depth={args.depth} window={args.window} hurst={hurst} "
        f"→ {block.sig_dim} columns")

    sig_tr = sig_features(z_tr, tr_dates, manifest, block, log)
    sig_va = sig_features(z_va, va_dates, manifest, block, log)

    step = args.row_step
    tr_s, va_s = tr[::step], va[::step]
    sig_tr, sig_va = np.ascontiguousarray(sig_tr[::step]), np.ascontiguousarray(sig_va[::step])
    z_tr, z_va = np.ascontiguousarray(z_tr[::step]), np.ascontiguousarray(z_va[::step])
    log(f"XGB rows (step {step}): train {len(tr_s):,}  valid {len(va_s):,}")

    Xtr = np.ascontiguousarray(X_mm[tr_s]).astype(np.float32)
    Xva = np.ascontiguousarray(X_mm[va_s]).astype(np.float32)
    ytr, wtr = np.ascontiguousarray(y[tr_s]), np.ascontiguousarray(w[tr_s])
    yva, wva = np.ascontiguousarray(y[va_s]), np.ascontiguousarray(w[va_s])

    results = []
    for name, xt, xv in (
        ("base", Xtr, Xva),
        ("base+sig", np.hstack([Xtr, sig_tr]), np.hstack([Xva, sig_va])),
        ("base+z+sig", np.hstack([Xtr, z_tr, sig_tr]),
         np.hstack([Xva, z_va, sig_va])),
    ):
        r2, best_it = fit_eval(xt, ytr, wtr, xv, yva, wva,
                               args.n_estimators, args.seed)
        log(f"  {name:12s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
            f"R²={r2:+.5f}")
        results.append({"variant": name, "n_features": int(xt.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        if name != "base":
            del xt, xv
        gc.collect()

    base_r2 = results[0]["r2_weighted"]
    for r in results:
        r["delta_vs_base"] = r["r2_weighted"] - base_r2
    (out / "ablation.json").write_text(json.dumps({
        "latents": str(args.latents), "data": str(data),
        "sig": {"mode": args.mode, "depth": args.depth,
                "window": args.window, "hurst": hurst,
                "sig_dim": block.sig_dim},
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log(f"deltas vs base: sig {results[1]['delta_vs_base']:+.5f}  "
        f"z+sig {results[2]['delta_vs_base']:+.5f} → {out}/ablation.json")


if __name__ == "__main__":
    main()
