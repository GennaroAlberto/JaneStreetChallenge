"""How much of r6 lives in the other responders — nonlinearity and story.

Extends the linear same-time oracle (R² 0.84 from all 8 others) along two
axes the user asked for: **nonlinearity** (MLP / GRU instead of ridge) and
**story** (up to 5 previous days of all nine responders, via flat lags, a
recurrent net over the day-lag sequence, or a depth-2 log-signature of the
multi-day responder path).

Two families of variants, kept clearly apart:

  CEILING (uses TRUE same-time others — not available at inference;
  measures structure, not deployable score):
    lin_same        ridge on r0..r5,r7,r8 at time t                (8)
    mlp_same        MLP on the same 8
    mlp_same_hist   MLP on same 8 + 5-day lags of all nine        (53)
    gru_days        GRU over the 5-day lag sequence + same-time 8
    mlp_same_sig    MLP on same 8 + log-sig L2 of the last-5-day
                    responder path (day-level context, 55 dims)

  DEPLOYABLE (history only — everything is yesterday-or-older):
    lin_hist        ridge on the 45 lag columns
    mlp_hist        MLP on the 45 lag columns

All fit on ``--train-days`` (default 250) ending at 1597 (embargo), tested
statically on the 1599–1698 tail with the competition metric.

Usage
-----
    uv run python scripts/resp_story_lab.py --out artifacts/bench/story_lab
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from janestreet.config import Cfg
from janestreet.data.ingest import scan_train_dates
from janestreet.theory.signatures import compute_log_signature
from janestreet.theory.torch_utils import auto_device

RESP = [f"responder_{i}" for i in range(9)]
OTHERS = [f"responder_{i}" for i in (0, 1, 2, 3, 4, 5, 7, 8)]
N_LAG_DAYS = 5


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# data assembly (responders only — tiny)
# ---------------------------------------------------------------------------

def build_frame(lo, hi):
    """Rows for [lo, hi] with same-time responders + 5 previous-day lags."""
    cfg = Cfg()
    df = (scan_train_dates(cfg, lo - N_LAG_DAYS, hi)
          .select(["date_id", "symbol_id", "time_id", "weight", *RESP])
          .collect())
    out = df
    for k in range(1, N_LAG_DAYS + 1):
        lagged = (df.select(["date_id", "symbol_id", "time_id", *RESP])
                  .with_columns((pl.col("date_id") + k).alias("date_id"))
                  .rename({c: f"{c}_lag{k}" for c in RESP}))
        out = out.join(lagged, on=["date_id", "symbol_id", "time_id"], how="left")
    out = out.filter(pl.col("date_id") >= lo).fill_null(0.0)
    return out.sort(["date_id", "symbol_id", "time_id"])


def day_sigs(lo, hi, downsample=4):
    """Depth-2 log-signature of the last-5-day responder path, one per
    (symbol, date) — day-level 'story' context (55 dims incl. time channel)."""
    cfg = Cfg()
    df = (scan_train_dates(cfg, lo - N_LAG_DAYS, hi)
          .select(["date_id", "symbol_id", "time_id", *RESP])
          .collect()
          .sort(["symbol_id", "date_id", "time_id"]))
    sym = df["symbol_id"].to_numpy()
    dat = df["date_id"].to_numpy()
    R = df.select(RESP).to_numpy().astype(np.float32)
    np.nan_to_num(R, copy=False)

    sigs: dict[tuple[int, int], np.ndarray] = {}
    t_per = Cfg().n_times_per_date
    for s in np.unique(sym):
        m = sym == s
        Rs, ds = R[m], dat[m]
        days = np.unique(ds)
        day_ix = {int(d): np.where(ds == d)[0] for d in days}
        for d in days:
            if int(d) < lo:
                continue
            hist = [day_ix.get(int(d) - k) for k in range(N_LAG_DAYS, 0, -1)]
            if any(h is None or len(h) != t_per for h in hist):
                continue
            path = np.concatenate([Rs[h] for h in hist])[::downsample]  # (T', 9)
            tt = np.linspace(0.0, 1.0, len(path), dtype=np.float32)[:, None]
            seg = torch.from_numpy(np.concatenate([tt, path], axis=1)[None])
            with torch.no_grad():
                sig = compute_log_signature(seg, depth=2, hurst=None)
            sigs[(int(s), int(d))] = sig.numpy()[0].astype(np.float32)
    return sigs


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, k, hidden=(128, 64), dropout=0.1):
        super().__init__()
        dims = [k, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRUDays(nn.Module):
    """GRU over the (5-day, 9-responder) lag sequence + same-time others."""

    def __init__(self, n_same=8, hidden=32):
        super().__init__()
        self.gru = nn.GRU(9, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_same, 64), nn.GELU(),
                                  nn.Linear(64, 1))

    def forward(self, x):                      # x = [same(8) | lags(45)]
        same, lags = x[:, :8], x[:, 8:].reshape(-1, N_LAG_DAYS, 9)
        _, h = self.gru(lags)
        return self.head(torch.cat([h[-1], same], dim=-1)).squeeze(-1)


def train_nn(model, Xtr, ytr, wtr, Xva, yva, wva, device, epochs=15,
             bs=16384, lr=1e-3, log=print):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr); wt = torch.from_numpy(wtr)
    Xv = torch.from_numpy(Xva).to(device)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            sl = perm[i:i + bs]
            xb, yb, wb = (Xt[sl].to(device), yt[sl].to(device), wt[sl].to(device))
            opt.zero_grad()
            p = model(xb)
            loss = (wb * (p - yb) ** 2).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(Xv[i:i + 65536]).cpu().numpy()
                                 for i in range(0, len(Xv), 65536)])
        r2 = r2_weighted(yva, pv, wva)
        log(f"    epoch {ep + 1:>2}  tail R²={r2:+.5f}")
        if r2 > best:
            best, bad, best_state = r2, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= 4:
            break
    model.load_state_dict(best_state)
    return best


def ridge(Xtr, ytr, wtr, Xva, yva, wva, lam=1e2):
    A = (Xtr * wtr[:, None]).T @ Xtr + lam * np.eye(Xtr.shape[1])
    b = (Xtr * wtr[:, None]).T @ ytr
    coef = np.linalg.solve(A, b)
    return r2_weighted(yva, Xva @ coef, wva)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-days", type=int, default=250)
    p.add_argument("--train-hi", type=int, default=1597)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="artifacts/bench/story_lab")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg):
        line = msg if str(msg).endswith("\n") else str(msg) + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    device = auto_device(args.device)
    lo = args.train_hi - args.train_days + 1
    log(f"train {lo}-{args.train_hi} ({args.train_days}d)  "
        f"tail {args.valid_lo}-{args.valid_hi}  device={device}")

    t0 = time.time()
    tr = build_frame(lo, args.train_hi)
    va = build_frame(args.valid_lo, args.valid_hi)
    lag_cols = [f"{c}_lag{k}" for k in range(1, N_LAG_DAYS + 1) for c in RESP]

    def mats(df):
        same = df.select(OTHERS).to_numpy().astype(np.float32)
        hist = df.select(lag_cols).to_numpy().astype(np.float32)
        y = df["responder_6"].to_numpy().astype(np.float32)
        w = df["weight"].to_numpy().astype(np.float32)
        return same, hist, y, w

    s_tr, h_tr, y_tr, w_tr = mats(tr)
    s_va, h_va, y_va, w_va = mats(va)
    log(f"frames: train {len(y_tr):,} rows, valid {len(y_va):,} "
        f"({time.time() - t0:.0f}s)")

    results = {}

    log("== CEILING family (true same-time others: structure, not deployable) ==")
    results["lin_same"] = ridge(s_tr, y_tr, w_tr, s_va, y_va, w_va)
    log(f"  lin_same(8): {results['lin_same']:+.5f}")
    log("  mlp_same(8):")
    results["mlp_same"] = train_nn(MLP(8), s_tr, y_tr, w_tr, s_va, y_va, w_va,
                                   device, log=log)
    X_tr53 = np.hstack([s_tr, h_tr]); X_va53 = np.hstack([s_va, h_va])
    log("  mlp_same_hist(53):")
    results["mlp_same_hist"] = train_nn(MLP(53), X_tr53, y_tr, w_tr,
                                        X_va53, y_va, w_va, device, log=log)
    log("  gru_days(5x9 + 8):")
    results["gru_days"] = train_nn(GRUDays(), X_tr53, y_tr, w_tr,
                                   X_va53, y_va, w_va, device, log=log)

    log("  computing day-level log-signatures (L2, 5-day paths)…")
    t0 = time.time()
    sig_tr = day_sigs(lo, args.train_hi)
    sig_va = day_sigs(args.valid_lo, args.valid_hi)
    sig_dim = next(iter(sig_tr.values())).shape[0]
    log(f"  {len(sig_tr):,}+{len(sig_va):,} day-sigs of dim {sig_dim} "
        f"({time.time() - t0:.0f}s)")

    def sig_matrix(df, sigs):
        keys = df.select(["symbol_id", "date_id"]).to_numpy()
        M = np.zeros((len(keys), sig_dim), np.float32)
        for i, (s, d) in enumerate(keys):
            v = sigs.get((int(s), int(d)))
            if v is not None:
                M[i] = v
        # standardize the signature block (scales vary wildly across terms)
        mu, sd = M.mean(0, keepdims=True), M.std(0, keepdims=True) + 1e-8
        return (M - mu) / sd

    Xs_tr = np.hstack([s_tr, sig_matrix(tr, sig_tr)])
    Xs_va = np.hstack([s_va, sig_matrix(va, sig_va)])
    log(f"  mlp_same_sig({Xs_tr.shape[1]}):")
    results["mlp_same_sig"] = train_nn(MLP(Xs_tr.shape[1]), Xs_tr, y_tr, w_tr,
                                       Xs_va, y_va, w_va, device, log=log)

    log("== DEPLOYABLE family (history only) ==")
    results["lin_hist"] = ridge(h_tr, y_tr, w_tr, h_va, y_va, w_va)
    log(f"  lin_hist(45): {results['lin_hist']:+.5f}")
    log("  mlp_hist(45):")
    results["mlp_hist"] = train_nn(MLP(45), h_tr, y_tr, w_tr,
                                   h_va, y_va, w_va, device, log=log)

    (out / "story.json").write_text(json.dumps({
        "results": results,
        "train": [lo, args.train_hi], "valid": [args.valid_lo, args.valid_hi],
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log("== summary ==")
    for k, v in results.items():
        log(f"  {k:15s} {v:+.5f}")


if __name__ == "__main__":
    main()
