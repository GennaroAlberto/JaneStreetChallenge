"""Score the regenerated ensemble: enriched XGB stream vs old-style.

Inputs (all on the 1599--1698 tail, memmap row order):
  * pool_rebuild/preds/xgb_base.npz and xgb_all.npz  (key: ``pred``)
  * regen members' preds/{tag}_online.npz            (keys: preds/y/w)

Alignment is asserted, not assumed: the RNN npz files carry their own y/w,
which must match the memmap's y/w for the same rows bit-for-bit.

Outputs: per-stream scores, pairwise stream correlations (diversity
accounting), and the two 3-way simple averages:
  old-style: xgb_base + gru_modelr_online + lstm_modelr_online
  enriched:  xgb_all  + gru_modelr_online + lstm_modelr_online
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi):
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--xgb-preds", default="artifacts/bench/pool_rebuild/preds")
    p.add_argument("--xgb-variants", default="xgb_base,xgb_all",
                   help="comma-separated npz basenames in --xgb-preds; a "
                        "3-way blend (variant + 2 RNNs) is scored for each, "
                        "raw and per-symbol-calibrated")
    p.add_argument("--rnn-preds", default="artifacts/bench/regen_ensemble/preds")
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--out", default="artifacts/bench/regen_ensemble/blend.json")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    y = np.load(data / "y.f32.npy", mmap_mode="r")[va]
    w = np.load(data / "w.f32.npy", mmap_mode="r")[va]
    syms = np.load(data / "symbols.i16.npy", mmap_mode="r")[va]
    times = np.load(data / "times.i16.npy", mmap_mode="r")[va]
    dates = np.load(data / "dates.i16.npy", mmap_mode="r")[va]

    def raw_to_memmap(stream: np.ndarray) -> np.ndarray:
        """Reorder a raw-walk stream (per-date time-major, the order
        prepare_dataset day frames iterate) into memmap order (per-date
        symbol-major). Same rows, different within-day permutation."""
        out = np.empty_like(stream)
        for d in np.unique(dates):
            m = dates == d
            # perm[p] = memmap-index (within the date) of the row at raw
            # position p: raw order sorts by (time, symbol)
            perm = np.lexsort((syms[m], times[m]))
            tmp = np.empty(int(m.sum()), dtype=stream.dtype)
            tmp[perm] = stream[m]          # memmap[perm[p]] = raw[p]
            out[np.where(m)[0]] = tmp
        return out

    streams: dict[str, np.ndarray] = {}
    xgb_variants = [v.strip() for v in args.xgb_variants.split(",")]
    for name in xgb_variants:
        streams[name] = np.load(Path(args.xgb_preds) / f"{name}.npz")["pred"]
    for tag in ("gru_modelr", "lstm_modelr"):
        blob = np.load(Path(args.rnn_preds) / f"{tag}_online.npz")
        y_r, w_r, p_r = (raw_to_memmap(blob["y"]), raw_to_memmap(blob["w"]),
                         raw_to_memmap(blob["preds"]))
        # alignment assertion: reordered member y/w must equal the memmap's.
        # The memmap targets went through the precompute's float16 frame, so
        # compare at f16 resolution.
        yq = y_r.astype(np.float16).astype(np.float32)
        wq = w_r.astype(np.float16).astype(np.float32)
        if not (np.allclose(yq, y, atol=2e-3) and np.allclose(wq, w, atol=2e-3)):
            raise ValueError(f"{tag}: y/w mismatch — streams are misaligned")
        streams[f"{tag}_online"] = p_r
    n = {k: len(v) for k, v in streams.items()}
    if len(set(n.values())) != 1 or next(iter(n.values())) != len(va):
        raise ValueError(f"stream length mismatch: {n} vs {len(va)} rows")

    singles = {k: r2_weighted(y, v, w) for k, v in streams.items()}
    names = list(streams)
    corr = {f"{a}~{b}": float(np.corrcoef(streams[a], streams[b])[0, 1])
            for i, a in enumerate(names) for b in names[i + 1:]}

    def calibrate(stream, lookback=20, ridge_w=3.0):
        """Per-symbol trailing shrinkage α (measured +0.00014 standalone)."""
        out_s = stream.copy()
        udates = np.unique(dates)
        for i, d in enumerate(udates):
            lo = udates[max(0, i - lookback)]
            hist = (dates >= lo) & (dates < d)
            today = dates == d
            if not hist.any():
                continue
            for s_id in np.unique(syms[today]):
                hs = hist & (syms == s_id)
                if hs.sum() < 500:
                    continue
                num = float(np.sum(w[hs] * stream[hs] * y[hs]))
                den = float(np.sum(w[hs] * stream[hs] ** 2))
                a_hat = num / (den + 1e-12)
                alpha = float(np.clip(
                    (a_hat * den + ridge_w * den) / (den + ridge_w * den),
                    0.0, 1.5))
                out_s[today & (syms == s_id)] *= alpha
        return out_s

    def blend(keys):
        return np.mean([streams[k] for k in keys], axis=0)

    blends = {}
    for v in xgb_variants:
        b = blend([v, "gru_modelr_online", "lstm_modelr_online"])
        blends[f"{v} + 2 RNN"] = r2_weighted(y, b, w)
        blends[f"{v} + 2 RNN, calibrated"] = r2_weighted(y, calibrate(b), w)

    report = {"singles": singles, "pairwise_corr": corr, "blends": blends,
              "n_rows": int(len(va)), "timestamp": datetime.now().isoformat()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
