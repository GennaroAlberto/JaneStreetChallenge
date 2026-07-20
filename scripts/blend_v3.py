"""Stack v3 assembly — the 280d checkpoint blend.

Streams (all trained on the 280d window, tail 1599–1698 online walks):
    xgb_all_chain_ranks   pool v3 XGB, 241 features         (memmap order)
    gru/lstm_modelr       full-size regen members           (raw-walk order)
    lstm64_s42/s1/s2      cheap [64,64] seed bag            (memmap order)
plus the v2 XGB stream as the incumbent reference.

Protocol (matches the interpretation-probe stack test): equal-weight
blends → per-symbol trailing calibration → vol-scaling pred·(σ̂/σ̄)^γ
with the |y| model fit on the train window and γ selected on tail half 1
(dates ≤ 1649); verdicts on the untouched half 2.

Usage
-----
    uv run python scripts/blend_v3.py --out artifacts/bench/pool_rebuild_280/blend_v3.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

SPLIT_DATE = 1649          # tail half 1 = dates <= SPLIT_DATE (selection)
GAMMAS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--v3-preds", default="artifacts/bench/pool_rebuild_280/preds")
    p.add_argument("--v2-preds", default="artifacts/bench/pool_rebuild_v2/preds")
    p.add_argument("--rnn-preds", default="artifacts/bench/regen_ensemble/preds")
    p.add_argument("--train-lo", type=int, default=1318)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--vol-row-step", type=int, default=4)
    p.add_argument("--out", default="artifacts/bench/pool_rebuild_280/blend_v3.json")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    ranges = manifest["date_row_ranges"]
    K = manifest["K"]

    def rows(lo, hi, step=1):
        r = np.concatenate([np.arange(*ranges[str(d)], dtype=np.int64)
                            for d in range(lo, hi + 1) if str(d) in ranges])
        return r[::step] if step > 1 else r

    va = rows(args.valid_lo, args.valid_hi)
    y_all = np.load(data / "y.f32.npy", mmap_mode="r")
    w_all = np.load(data / "w.f32.npy", mmap_mode="r")
    y = np.ascontiguousarray(y_all[va]).astype(np.float64)
    w = np.ascontiguousarray(w_all[va]).astype(np.float64)
    syms = np.load(data / "symbols.i16.npy", mmap_mode="r")[va]
    times = np.load(data / "times.i16.npy", mmap_mode="r")[va]
    dates = np.load(data / "dates.i16.npy", mmap_mode="r")[va]
    h1, h2 = dates <= SPLIT_DATE, dates > SPLIT_DATE

    def raw_to_memmap(stream):
        out = np.empty_like(stream)
        for d in np.unique(dates):
            m = dates == d
            perm = np.lexsort((syms[m], times[m]))
            tmp = np.empty(int(m.sum()), dtype=stream.dtype)
            tmp[perm] = stream[m]
            out[np.where(m)[0]] = tmp
        return out

    def check_yw(y_r, w_r, tag):
        yq = y_r.astype(np.float16).astype(np.float32)
        wq = w_r.astype(np.float16).astype(np.float32)
        if not (np.allclose(yq, y, atol=2e-3) and np.allclose(wq, w, atol=2e-3)):
            raise ValueError(f"{tag}: y/w mismatch — streams are misaligned")

    streams: dict[str, np.ndarray] = {}
    streams["xgb_v2"] = np.load(Path(args.v2_preds) / "xgb_all_chain.npz")["pred"]
    streams["xgb_v3"] = np.load(
        Path(args.v3_preds) / "xgb_all_chain_ranks.npz")["pred"]
    for s in (42, 1, 2):
        blob = np.load(Path(args.v3_preds) / f"lstm64_s{s}.npz")
        check_yw(blob["y"], blob["w"], f"lstm64_s{s}")   # already memmap order
        streams[f"lstm64_s{s}"] = blob["pred"]
    for tag in ("gru_modelr", "lstm_modelr"):
        blob = np.load(Path(args.rnn_preds) / f"{tag}_online.npz")
        y_r, w_r = raw_to_memmap(blob["y"]), raw_to_memmap(blob["w"])
        check_yw(y_r, w_r, tag)
        streams[tag] = raw_to_memmap(blob["preds"])
    streams["lstm64_bag"] = np.mean(
        [streams[f"lstm64_s{s}"] for s in (42, 1, 2)], axis=0)

    singles = {k: r2_weighted(y, v, w) for k, v in streams.items()}
    names = list(streams)
    corr = {f"{a}~{b}": float(np.corrcoef(streams[a], streams[b])[0, 1])
            for i, a in enumerate(names) for b in names[i + 1:]}

    # ---- vol model for the scale step (train window, base features) --------
    tr_s = rows(args.train_lo, args.train_hi, args.vol_row_step)
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    Xtr = np.ascontiguousarray(X_mm[tr_s]).astype(np.float32)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.5,
                         min_child_weight=10, reg_lambda=1.0,
                         tree_method="hist", max_bin=128, n_jobs=4,
                         random_state=0, objective="reg:squarederror")
    m.fit(Xtr, np.log1p(np.abs(y_all[tr_s])), sample_weight=w_all[tr_s],
          verbose=False)
    del Xtr
    Xva = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    vh = np.clip(np.expm1(m.predict(Xva).astype(np.float64)), 0.05, None)
    del Xva
    vbar = float(np.average(vh[h1], weights=w[h1]))

    def calibrate(stream, lookback=20, ridge_w=3.0):
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

    combos = {
        "incumbent(xgb_v2+gru+lstm)": ["xgb_v2", "gru_modelr", "lstm_modelr"],
        "v3_core(xgb_v3+gru+lstm)": ["xgb_v3", "gru_modelr", "lstm_modelr"],
        "v3_bag4(+lstm64_bag)": ["xgb_v3", "gru_modelr", "lstm_modelr",
                                 "lstm64_bag"],
        "v3_all6(members flat)": ["xgb_v3", "gru_modelr", "lstm_modelr",
                                  "lstm64_s42", "lstm64_s1", "lstm64_s2"],
    }
    report_blends = {}
    for label, keys in combos.items():
        b = np.mean([streams[k] for k in keys], axis=0)
        cal = calibrate(b)
        row = {
            "raw_full": r2_weighted(y, b, w),
            "cal_full": r2_weighted(y, cal, w),
            "cal_h2": r2_weighted(y[h2], cal[h2], w[h2]),
        }
        best_g, best_r2 = None, -9.0
        for g in GAMMAS:
            r2_h1 = r2_weighted(y[h1], (cal * (vh / vbar) ** g)[h1], w[h1])
            if r2_h1 > best_r2:
                best_g, best_r2 = g, r2_h1
        vs = cal * (vh / vbar) ** best_g
        row["gamma"] = best_g
        row["cal_vol_full"] = r2_weighted(y, vs, w)
        row["cal_vol_h2"] = r2_weighted(y[h2], vs[h2], w[h2])
        report_blends[label] = row
        print(f"{label:28s} raw={row['raw_full']:+.5f} "
              f"cal={row['cal_full']:+.5f} "
              f"cal+vol(g={best_g})={row['cal_vol_full']:+.5f} "
              f"| h2: cal={row['cal_h2']:+.5f} "
              f"cal+vol={row['cal_vol_h2']:+.5f}", flush=True)

    Path(args.out).write_text(json.dumps({
        "singles": singles, "pairwise_corr": corr, "blends": report_blends,
        "split_date": SPLIT_DATE, "n_rows": int(len(va)),
        "timestamp": datetime.now().isoformat()}, indent=2))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
