"""Causal Fourier band-energy block — the frequency-domain angle.

For each (date, symbol) day path of the five atlas features, compute the
rfft of the TRAILING 64-step window at every timestep (strided, causal)
and reduce to 3 log band-energy shares: low (1-4 cyc), mid (5-16),
high (17-31), normalized by total window energy so the block carries
SHAPE not scale (scale lives in the rolling-std features already).
15 columns. Ablated base vs base+block on the standard harness.

Prior is honest-low: the target is an SMA (low-pass) of near-white
innovations and the pool's rolling stats already span low-frequency
content — this tests whether mid/high-band structure adds anything.

Usage
-----
    uv run python scripts/ablate_fourier.py --out artifacts/bench/ablation_fourier.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from rebuild_pool_ablation import fit_eval, r2_weighted, rows_for_dates  # noqa: E402

FEATS = ["feature_37", "feature_38", "feature_60", "feature_45", "feature_06"]
WIN = 64
BANDS = [(1, 5), (5, 17), (17, 32)]   # rfft bin ranges [lo, hi)
N_TIMES = 968


def fourier_block(X_mm, manifest, dates, feat_idx):
    """(n_rows, 5*3) log band-energy shares, causal, NaN for t < WIN-1."""
    ranges = manifest["date_row_ranges"]
    parts = []
    for d in dates:
        s, e = ranges[str(d)]
        n_sym = (e - s) // N_TIMES
        block = np.nan_to_num(np.ascontiguousarray(
            X_mm[s:e]).astype(np.float32))[:, feat_idx]
        block = block.reshape(n_sym, N_TIMES, len(feat_idx))
        out = np.full((n_sym, N_TIMES, len(feat_idx) * len(BANDS)), np.nan,
                      np.float32)
        for f in range(len(feat_idx)):
            paths = block[:, :, f]                                # (S, T)
            wins = np.lib.stride_tricks.sliding_window_view(
                paths, WIN, axis=1)                               # (S, T-W+1, W)
            spec = np.abs(np.fft.rfft(wins, axis=-1)) ** 2        # (S, T-W+1, 33)
            tot = spec[..., 1:].sum(-1) + 1e-12                   # drop DC
            for b, (lo, hi) in enumerate(BANDS):
                share = spec[..., lo:hi].sum(-1) / tot
                out[:, WIN - 1:, f * len(BANDS) + b] = np.log(share + 1e-6)
        parts.append(out.reshape(n_sym * N_TIMES, -1))
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--row-step", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/ablation_fourier.json")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["feature_cols"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")
    feat_idx = [names.index(f) for f in FEATS]

    tr_dates, tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    step = args.row_step

    print("computing fourier block…", flush=True)
    F_tr = fourier_block(X_mm, manifest, tr_dates, feat_idx)[::step]
    F_va = fourier_block(X_mm, manifest, va_dates, feat_idx)[::step]

    Xtr = np.ascontiguousarray(X_mm[tr[::step]]).astype(np.float32)
    Xva = np.ascontiguousarray(X_mm[va[::step]]).astype(np.float32)
    ytr, wtr = y[tr[::step]], w[tr[::step]]
    yva, wva = y[va[::step]], w[va[::step]]

    r2_base, it_b, _ = fit_eval(Xtr, ytr, wtr, Xva, yva, wva,
                                args.n_estimators, args.seed)
    print(f"  base({K})          R2 = {r2_base:+.5f}  (iter {it_b})", flush=True)
    Xtr2 = np.hstack([Xtr, F_tr])
    Xva2 = np.hstack([Xva, F_va])
    r2_f, it_f, _ = fit_eval(Xtr2, ytr, wtr, Xva2, yva, wva,
                             args.n_estimators, args.seed)
    print(f"  base+fourier({Xtr2.shape[1]}) R2 = {r2_f:+.5f}  (iter {it_f})",
          flush=True)
    print(f"delta: {r2_f - r2_base:+.5f}", flush=True)
    Path(args.out).write_text(json.dumps({
        "base": r2_base, "fourier": r2_f, "delta": r2_f - r2_base,
        "win": WIN, "bands": BANDS, "feats": FEATS,
        "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
