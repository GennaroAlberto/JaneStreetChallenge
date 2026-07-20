"""Innovation labels as aux targets — label engineering via deconvolution.

The construction says the target is an average of future innovations s.
Every aux target we (and the 8th place) have used is a *smoothed* forward
quantity; the deconvolution machinery can instead manufacture the raw
thing at training time: ŝ(t+1), the next innovation, ridge-deconvolved
from the day's responder_8 path. Labels are train-time-only, so this is
legal by construction.

A/B on ModelR where ONLY the two synthetic aux targets differ:

  A (baseline): aux = [r7, r8, r9synth, r10synth]   (Volkova forward SMAs)
  B (innovation): aux = [r7, r8, ŝ(t+1), mean ŝ(t+1..t+4)]

Identical architecture, seeds, windows, and walk protocol. Both members
are trained lighter than production (hidden [64,64]) — the comparison is
the point, not the absolute score. Walk-forward drives the nn.Module
directly with (symbols, T, K) day tensors from the memmap (symbol-major
blocks), sidestepping the time-major reshape helpers entirely.

Usage
-----
    uv run python scripts/aux_innovation_lab.py --out artifacts/bench/innov_aux_lab
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from profile_features import deconvolution_matrix  # noqa: E402
from responder_chain import load_raw_responders  # noqa: E402

from janestreet.models.base import FitData  # noqa: E402
from janestreet.models.recurrent import RecurrentModel  # noqa: E402

N_TIMES = 968
SMA_W = 4


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(ranges, lo, hi):
    return np.concatenate([np.arange(*ranges[str(d)])
                           for d in range(lo, hi + 1) if str(d) in ranges])


def innovation_labels(R8_rows: np.ndarray, m_inv: np.ndarray):
    """Per (date, symbol) block: ŝ(t+1) and mean ŝ(t+1..t+4).

    ``R8_rows`` is the responder_8 column in memmap row order; blocks of
    N_TIMES per (date, symbol). ŝ has length T + w - 1; index t+1 is
    defined for every t in [0, T)."""
    n = len(R8_rows)
    blocks = R8_rows.reshape(n // N_TIMES, N_TIMES)
    s_hat = blocks @ m_inv.T                     # (B, T + w - 1)
    # pad one column so the last row's 4-step window is defined (edge
    # effect: 1 of 4 terms zero on the final timestep of each day)
    s_hat = np.concatenate(
        [s_hat, np.zeros((len(s_hat), 1), s_hat.dtype)], axis=1)
    innov1 = s_hat[:, 1:N_TIMES + 1]
    cs = np.cumsum(np.concatenate(
        [np.zeros((len(s_hat), 1), np.float64), s_hat], axis=1), axis=1)
    innov4 = (cs[:, 1 + SMA_W:1 + SMA_W + N_TIMES] - cs[:, 1:1 + N_TIMES]) / SMA_W
    return (innov1.reshape(n).astype(np.float32),
            innov4.reshape(n).astype(np.float32))


def synth_forward(Rcol: np.ndarray, shifts: list[int]):
    """Volkova synthetic: sum of the column and its forward shifts, per block."""
    n = len(Rcol)
    b = Rcol.reshape(n // N_TIMES, N_TIMES).astype(np.float64)
    out = b.copy()
    for k in shifts:
        sh = np.zeros_like(b)
        sh[:, :-k] = b[:, k:]
        out += sh
    return out.reshape(n).astype(np.float32)


def walk_forward_direct(model, X_mm, ranges, y_all, w_all, K, valid_lo,
                        valid_hi, device, log):
    """Day-by-day online walk driving the nn.Module directly with
    symbol-major (D, T, K) day tensors from the memmap."""
    net = model.model
    preds, ys, ws = [], [], []
    days = [d for d in range(valid_lo, valid_hi + 1) if str(d) in ranges]
    for i, d in enumerate(days):
        if i > 0 and model.lr_refit > 0:
            pd = days[i - 1] if i > 0 else None
            s, e = ranges[str(pd)]
            Xd = np.nan_to_num(np.ascontiguousarray(X_mm[s:e]).astype(np.float32))
            xt = torch.from_numpy(Xd).reshape((e - s)//N_TIMES, N_TIMES, K).to(device)
            yt = torch.from_numpy(y_all[s:e].astype(np.float32)).reshape(-1, N_TIMES).to(device)
            wt = torch.from_numpy(w_all[s:e].astype(np.float32)).reshape(-1, N_TIMES).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=model.lr_refit,
                                    weight_decay=model.weight_decay)
            net.train(); opt.zero_grad()
            out = net(xt, None)
            loss = model.criterion(out[0].flatten(), yt.flatten(), wt.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), model.grad_clip)
            opt.step()
        s, e = ranges[str(d)]
        Xd = np.nan_to_num(np.ascontiguousarray(X_mm[s:e]).astype(np.float32))
        xt = torch.from_numpy(Xd).reshape((e - s)//N_TIMES, N_TIMES, K).to(device)
        net.eval()
        with torch.no_grad():
            out = net(xt, None)
        preds.append(np.clip(out[0].cpu().numpy().reshape(-1), -5, 5))
        ys.append(y_all[s:e].astype(np.float64))
        ws.append(w_all[s:e].astype(np.float64))
        if (i + 1) % 25 == 0:
            log(f"    walked {i + 1}/{len(days)}")
    return map(np.concatenate, (preds, ys, ws))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1398)
    p.add_argument("--train-hi", type=int, default=1597)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="mps")
    p.add_argument("--ridge-lambda", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/innov_aux_lab")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg):
        line = str(msg) if str(msg).endswith("\n") else str(msg) + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    ranges = manifest["date_row_ranges"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    y_all = np.load(data / "y.f32.npy", mmap_mode="r")
    w_all = np.load(data / "w.f32.npy", mmap_mode="r")

    tr = rows_for_dates(ranges, args.train_lo, args.train_hi)
    va = rows_for_dates(ranges, args.valid_lo, args.valid_hi)
    log(f"train rows {len(tr):,}  valid rows {len(va):,}")

    log("loading raw responders + building labels…")
    Rtr = load_raw_responders(args.train_lo, args.train_hi)
    Rva = load_raw_responders(args.valid_lo, args.valid_hi)
    if not np.allclose(Rtr[:, 6].astype(np.float16).astype(np.float32),
                       y_all[tr], atol=2e-3):
        raise ValueError("raw/memmap alignment failed")
    m_inv = deconvolution_matrix(N_TIMES, SMA_W, args.ridge_lambda)

    def resp_variants(R):
        r7, r8, r6 = R[:, 7], R[:, 8], R[:, 6]
        r9s = synth_forward(r8, [4])
        r10s = synth_forward(r6, [20, 40])
        i1, i4 = innovation_labels(r8.astype(np.float64), m_inv)
        A = np.column_stack([r7, r8, r9s, r10s]).astype(np.float32)
        B = np.column_stack([r7, r8, i1, i4]).astype(np.float32)
        return A, B

    respA_tr, respB_tr = resp_variants(Rtr)
    respA_va, respB_va = resp_variants(Rva)
    del Rtr, Rva

    Xtr = np.nan_to_num(np.ascontiguousarray(X_mm[tr]).astype(np.float32))
    Xva = np.nan_to_num(np.ascontiguousarray(X_mm[va]).astype(np.float32))
    sym = np.load(data / "symbols.i16.npy", mmap_mode="r")
    dat = np.load(data / "dates.i16.npy", mmap_mode="r")
    tim = np.load(data / "times.i16.npy", mmap_mode="r")

    def fitdata(rows_, X, resp):
        return FitData(X=X, resp=resp,
                       y=y_all[rows_].astype(np.float32),
                       w=w_all[rows_].astype(np.float32),
                       symbols=sym[rows_].astype(np.int64),
                       dates=dat[rows_].astype(np.int64),
                       times=tim[rows_].astype(np.int64))

    results = {}
    for tag, rtr_, rva_ in (("A_sma_synth", respA_tr, respA_va),
                            ("B_innovation", respB_tr, respB_va)):
        log(f"== variant {tag} ==")
        model = RecurrentModel(
            model_type="lstm", aux_branches=True, num_aux=4,
            hidden_sizes=[64, 64], dropout_rates=[0.1, 0.1],
            hidden_sizes_linear=[], dropout_rates_linear=[],
            lr=1e-3, weight_decay=1e-2, batch_size=1, epochs=args.epochs,
            early_stopping_patience=3, grad_clip=1.0, lr_refit=1e-3,
            seed=args.seed, device=args.device)
        t0 = time.time()
        model.fit(fitdata(tr, Xtr, rtr_), fitdata(va, Xva, rva_), verbose=True)
        log(f"  fit {((time.time() - t0)/60):.1f} min")
        p_, y_, w_ = walk_forward_direct(
            model, X_mm, ranges, y_all, w_all, K,
            args.valid_lo, args.valid_hi, model.device, log)
        r2 = r2_weighted(y_, p_, w_)
        results[tag] = r2
        log(f"  {tag} online R² = {r2:+.5f}")
        np.savez_compressed(out / f"{tag}_online.npz",
                            preds=p_.astype(np.float32),
                            y=y_.astype(np.float32), w=w_.astype(np.float32))

    (out / "result.json").write_text(json.dumps({
        "results": results,
        "delta_B_vs_A": results["B_innovation"] - results["A_sma_synth"],
        "config": {"train": [args.train_lo, args.train_hi],
                   "hidden": [64, 64], "epochs": args.epochs,
                   "ridge_lambda": args.ridge_lambda},
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log(f"delta B vs A: {results['B_innovation'] - results['A_sma_synth']:+.5f}")


if __name__ == "__main__":
    main()
