"""Vol-bagged XGB specialist streams — the confirmed-but-never-productionized
decorrelation source (graveyard §11's kept twist: specialists averaged WITH
the generalist beat the generalist, +0.0006).

Regime label: each (date, symbol) day is assigned yesterday's realized vol
(mean |responder_8| of that symbol's previous day), bucketed into terciles
by thresholds FIT ON TRAIN. Two specialists — calm (bottom two terciles)
and storm (top tercile) — train on their subset and predict the FULL
tail, producing blendable streams in memmap order.

Usage
-----
    uv run python scripts/volbag_members.py --out artifacts/bench/volbag
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
from responder_chain import load_raw_responders  # noqa: E402

N_TIMES = 968


def day_vol_labels(lo, hi):
    """{(date, symbol): mean |r8| of that (date, symbol) day}."""
    R = load_raw_responders(lo, hi)          # rows in (date, symbol, time) order
    absr8 = np.abs(R[:, 8])
    n_days = len(absr8) // N_TIMES
    return absr8.reshape(n_days, N_TIMES).mean(1)   # one value per (date,symbol) block


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1098)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--row-step", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/volbag")
    args = p.parse_args()

    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    # regime label per (date,symbol) block, lagged one day: block i of date d
    # gets date d-1's realized vol for the same symbol; day-edge blocks with
    # no yesterday fall back to the train median (neutral).
    print("building lagged vol labels…", flush=True)
    tr_dates, tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    vol_tr_today = day_vol_labels(args.train_lo, args.train_hi)
    vol_va_today = day_vol_labels(args.valid_lo, args.valid_hi)

    def lag_by_symbol(dates, rows, vols_today, lo, hi):
        """map (date,symbol) → yesterday's vol via a dict keyed on (d,s)."""
        ranges = manifest["date_row_ranges"]
        syms_all = np.load(data / "symbols.i16.npy", mmap_mode="r")
        table = {}
        i = 0
        for d in range(lo, hi + 1):
            if str(d) not in ranges:
                continue
            s, e = ranges[str(d)]
            n_sym = (e - s) // N_TIMES
            day_syms = syms_all[s:e][::N_TIMES]
            for j in range(n_sym):
                table[(d, int(day_syms[j]))] = vols_today[i]
                i += 1
        assert i == len(vols_today)
        # emit lagged per-block labels aligned to rows
        out_lab = []
        for d in range(lo, hi + 1):
            if str(d) not in ranges:
                continue
            s, e = ranges[str(d)]
            n_sym = (e - s) // N_TIMES
            day_syms = syms_all[s:e][::N_TIMES]
            for j in range(n_sym):
                out_lab.append(table.get((d - 1, int(day_syms[j])), np.nan))
        return np.repeat(np.array(out_lab, np.float64), N_TIMES)

    lab_tr = lag_by_symbol(tr_dates, tr, vol_tr_today, args.train_lo, args.train_hi)
    lab_va = lag_by_symbol(va_dates, va, vol_va_today, args.valid_lo, args.valid_hi)
    med = np.nanmedian(lab_tr)
    lab_tr = np.nan_to_num(lab_tr, nan=med)
    lab_va = np.nan_to_num(lab_va, nan=med)
    t_hi = np.quantile(lab_tr, 2 / 3)
    print(f"storm threshold (train 67th pct): {t_hi:.4f}", flush=True)

    step = args.row_step
    # valid stays FULL-ROW (134 cols x 3.7M = 2 GB) so the streams are
    # blend-ready without a rerun; only train is subsampled
    Xva = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    yva, wva = y[va], w[va]

    results = {}
    for name, mask in (("calm", lab_tr < t_hi), ("storm", lab_tr >= t_hi)):
        rows_sub = tr[mask][::step]
        Xtr = np.ascontiguousarray(X_mm[rows_sub]).astype(np.float32)
        print(f"[{name}] train rows {len(rows_sub):,}", flush=True)
        r2, best_it, pred = fit_eval(Xtr, y[rows_sub], w[rows_sub],
                                     Xva, yva, wva,
                                     args.n_estimators, args.seed)
        print(f"[{name}] full-tail R² = {r2:+.5f} (iter {best_it})", flush=True)
        results[name] = r2
        np.savez_compressed(out / "preds" / f"xgb_{name}.npz",
                            pred=pred.astype(np.float32))
        del Xtr

    (out / "volbag.json").write_text(json.dumps({
        "results": results, "storm_threshold": float(t_hi),
        "train": [args.train_lo, args.train_hi], "row_step": step,
        "note": "valid preds are FULL-ROW (blend-ready, memmap order)",
        "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
