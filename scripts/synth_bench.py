"""Synthetic world for architecture triage (TimeXer / iTransformer / RNN).

Why synthetic: #555562 hands us the real data-generating process — an
underlying near-white per-minute signal u, responders = forward SMAs
(4/20/120) of u on two exchanges + noise, features = mostly noisy
descriptions of u's realized path with a small lead component. We generate
that world with *known planted mechanisms* and a closed-form reference
predictor, so each architecture's score reads as a **fraction of the
knowable ceiling** at the real problem's SNR (weighted R² of a few 0.01).

Planted mechanisms (each toggleable; each matched to one inductive bias):

  rev   intraday mean-reversion: E[u(t)] ∝ −SMA20(u)(t−1).
        Any causal temporal model (GRU, transformer) should get this.
  xd    cross-day: E[u(t)] ∝ L·φ(t) where L = yesterday's last-120 trend and
        φ(t) gates by time of day (+1 early, −1 late). Recoverable only by
        combining yesterday's responder path with today's clock —
        **TimeXer's endogenous-patch mechanism.**
  gate  cross-variate interaction: E[u(t)] ∝ min(f_a, f_b) of two persistent
        observable factors (one idio, one market-wide) —
        **iTransformer's inverted variate-attention mechanism** (and the
        min/max structure the DRW lab found on real data).
  lead  two features observe the next-4 innovations noisily (the
        feature_37/38 analog). Linear-Gaussian, any model can read it.

Feature layout mimics the real atlas taxonomy at the *same indices*:
37/38 = lead, 45/46/56/57(/58/60) = reversal-nowcast block (negated, noisy),
12/67/70 = SMA-160-smoothed slow copies, 06/04/07 = market factor + noisy
copies, 36/59 = idio factor + noisy copy; everything else is AR(1)
distractor noise. Responders follow the #555562 mapping: A = {6,7,8} =
SMA{20,120,4}(u), B = {3,4,5} on a noisy copy of u, {0,1,2} = B−A.

The reference predictor (per-mechanism components stored in oracle.parquet)
is Bayes-optimal for rev/xd/lead and a ρ^k-decay approximation for the gate
(Jensen gap ignored) — an *upper reference*, not a strict bound.

Usage
-----
    # generate a world (canonical Kaggle layout → any pipeline tool works)
    uv run python scripts/synth_bench.py gen --out artifacts/synth/world_full \\
        --symbols 8 --days 160

    # run the model zoo on it under the exact online-refit CV protocol
    uv run python scripts/synth_bench.py run --data artifacts/synth/world_full \\
        --models xgb,gru,gru_modelr,transformer,patchtst,itransformer,timexer \\
        --test-size 40
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import Cfg
from janestreet.data.features import FeatureBuilder
from janestreet.pipeline import prepare_dataset, run_cv
from janestreet.training.metrics import r2_weighted

TAIL = 130          # extra per-day steps so forward SMA-120 is defined at t=967
E_MIN2 = -0.5642    # E[min(X,Y)], X,Y iid std normal = -1/sqrt(pi)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _ar1(rng, shape, rho):
    """AR(1) with unit stationary variance along the last axis."""
    x = np.empty(shape, dtype=np.float64)
    innov = rng.normal(0.0, np.sqrt(1 - rho * rho), shape)
    x[..., 0] = rng.normal(0.0, 1.0, shape[:-1])
    for t in range(1, shape[-1]):
        x[..., t] = rho * x[..., t - 1] + innov[..., t]
    return x


def _fwd_sma(u: np.ndarray, w: int, t_out: int) -> np.ndarray:
    """r(t) = mean(u[t+1 .. t+w]) for t in [0, t_out) — #555562 convention."""
    cs = np.cumsum(u, axis=-1)
    cs = np.concatenate([np.zeros((*u.shape[:-1], 1)), cs], axis=-1)
    return (cs[..., 1 + w : 1 + w + t_out] - cs[..., 1:1 + t_out]) / w


def generate(args) -> None:
    rng = np.random.default_rng(args.seed)
    S, D, T = args.symbols, args.days, 968
    Tx = T + TAIL
    d0 = 700
    b_rev, b_xd, b_gate = args.beta_rev, args.beta_xd, args.beta_gate

    # time-of-day gate for the cross-day mechanism
    phi = np.zeros(Tx)
    phi[: T // 3] = 1.0
    phi[2 * T // 3:] = -1.0

    # persistent observable factors (gate drivers), extended past day end
    f_a = _ar1(rng, (S, D, Tx), args.rho_factor)          # idio
    f_b = _ar1(rng, (D, Tx), args.rho_factor)[None].repeat(S, axis=0)  # market

    # innovations: common market component + idio
    g = rng.normal(0.0, 1.0, (D, Tx))
    nu = np.sqrt(args.rho_mkt) * g[None] + np.sqrt(1 - args.rho_mkt) * rng.normal(0.0, 1.0, (S, D, Tx))

    gate = b_gate * (np.minimum(f_a, f_b) - E_MIN2)

    # sequential u: day loop (for yesterday's L), t loop (for the rev window)
    u = np.zeros((S, D, Tx))
    L = np.zeros((S, D))                                   # yesterday's last-120 trend
    for d in range(D):
        if d > 0:
            L[:, d] = u[:, d - 1, T - 120:T].mean(axis=-1)
        xd_d = b_xd * L[:, d, None] * phi[None, :]         # (S, Tx)
        win = np.zeros(S)                                  # rolling sum of last 20 u
        for t in range(Tx):
            mu = -b_rev * win / 20.0 + xd_d[:, t] + gate[:, d, t]
            ut = mu + nu[:, d, t]
            u[:, d, t] = ut
            win += ut
            if t >= 20:
                win -= u[:, d, t - 20]

    # ---- responders (A = u, B = u + venue noise) ---------------------------
    uB = u + rng.normal(0.0, args.venue_noise, u.shape)
    resp = {}
    for r, (w, base) in {6: (20, u), 7: (120, u), 8: (4, u),
                         3: (20, uB), 4: (120, uB), 5: (4, uB)}.items():
        resp[r] = _fwd_sma(base, w, T) + rng.normal(0.0, args.resp_noise, (S, D, T))
    for r, (a, b) in {0: (3, 6), 1: (4, 7), 2: (5, 8)}.items():
        resp[r] = resp[a] - resp[b]

    # ---- features (79 slots, real-atlas indices for the planted ones) ------
    ut = u[..., :T]
    # realized SMAs (shift the forward operator back by w)
    def realized_sma(w):
        r = np.zeros((S, D, T))
        r[..., w:] = _fwd_sma(u, w, T - w)
        return r
    r4, r20, r160 = realized_sma(4), realized_sma(20), realized_sma(160)

    lead_sig = _fwd_sma(nu, 4, T)                           # mean of next-4 innovations

    feats = {}
    nz = lambda s: rng.normal(0.0, s, (S, D, T))            # noqa: E731
    feats[37] = lead_sig + nz(args.lead_noise1)
    feats[38] = lead_sig + nz(args.lead_noise2)
    feats[45] = -r20 * 3.0 + nz(0.15)
    feats[46] = -r4 * 2.0 + nz(0.20)
    feats[56] = -r20 * 3.0 + nz(0.25)
    feats[57] = -r4 * 2.0 + nz(0.15)
    feats[58] = -r4 * 2.0 + nz(0.10)
    feats[60] = -r4 * 2.5 + nz(0.05)
    feats[12] = r160 * 8.0 + nz(0.30)
    feats[67] = r160 * 8.0 + nz(0.50)
    feats[70] = r160 * 8.0 + nz(0.40)
    feats[36] = f_a[..., :T].copy()
    feats[59] = f_a[..., :T] + nz(0.50)
    feats[6] = f_b[..., :T].copy()
    feats[4] = f_b[..., :T] + nz(0.30)
    feats[7] = f_b[..., :T] + nz(0.50)
    planted = sorted(feats)
    rho_pool = [0.0, 0.9, 0.99]
    for i in range(79):
        if i in feats:
            continue
        feats[i] = _ar1(rng, (S, D, T), rho_pool[i % 3])
    for i in (9, 10, 11):                                  # categorical slots (dropped)
        feats[i] = rng.integers(0, 5, (S, D, T)).astype(np.float64)

    # ---- reference predictor components ------------------------------------
    # rev: ŷ_rev(t) = −β/400 · (20·cs(t) − Σ_{j=t-20}^{t-1} cs(j)), cs = cumsum(u)
    cs = np.cumsum(ut, axis=-1)
    cs_pad = np.concatenate([np.zeros((S, D, 20)), cs], axis=-1)
    win_cs = np.stack([cs_pad[..., 20 + t - 20: 20 + t].sum(axis=-1) for t in range(T)], axis=-1)
    o_rev = -b_rev / 400.0 * (20.0 * cs - win_cs)
    # xd: ŷ_xd(t) = β·L·mean(φ(t+1..t+20))
    phi_fwd = np.convolve(phi, np.ones(20) / 20.0)[20 - 1:][:T]  # mean φ over (t, t+20]
    phi_fwd = np.concatenate([phi_fwd[1:], phi_fwd[-1:]])        # shift to (t+1..t+20)
    o_xd = b_xd * L[..., None] * phi_fwd[None, None, :]
    # gate: ρ^k decay approximation
    rho_bar = np.mean([args.rho_factor ** k for k in range(1, 21)])
    o_gate = b_gate * rho_bar * (np.minimum(f_a[..., :T], f_b[..., :T]) - E_MIN2)
    # lead: posterior mean of next-4 innovation mean from the two lead features
    prior_prec = 4.0
    p1, p2 = 1.0 / args.lead_noise1 ** 2, 1.0 / args.lead_noise2 ** 2
    post = (feats[37] * p1 + feats[38] * p2) / (prior_prec + p1 + p2)
    o_lead = (4.0 / 20.0) * post
    o_full = o_rev + o_xd + o_gate + o_lead

    # ---- assemble frames ----------------------------------------------------
    # Row order MUST be (date, time, symbol) — time-major within a day, the
    # canonical Kaggle layout. reshape_flat_to_sequence / update() / predict()
    # all assume it ("we sort externally"); writing symbol-major here
    # scrambles the sequence axis and manufactures within-day future leakage
    # (found the hard way: a GRU scored 3.6x the oracle ceiling).
    def flat(a: np.ndarray) -> np.ndarray:
        """(S, D, T) -> flat rows ordered by (date, time, symbol)."""
        return np.ascontiguousarray(a.transpose(1, 2, 0)).reshape(-1)

    sym = np.tile(np.arange(S, dtype=np.int16), D * T)
    dat = np.repeat(np.arange(d0, d0 + D, dtype=np.int16), T * S)
    tim = np.tile(np.repeat(np.arange(T, dtype=np.int16), S), D)
    weight = np.exp(rng.normal(0.0, 0.3, S * D * T)).astype(np.float32)

    cols = {
        "symbol_id": sym, "date_id": dat, "time_id": tim, "weight": weight,
        **{f"feature_{i:02d}": flat(feats[i]).astype(np.float32) for i in range(79)},
        **{f"responder_{r}": flat(resp[r]).astype(np.float32) for r in range(9)},
    }
    df = pl.DataFrame(cols)

    out = Path(args.out)
    (out / "train.parquet" / "partition_id=0").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "train.parquet" / "partition_id=0" / "part-0.parquet")

    pl.DataFrame({
        "symbol_id": sym, "date_id": dat, "time_id": tim,
        **{f"oracle_{k}": flat(v).astype(np.float32)
           for k, v in {"full": o_full, "rev": o_rev, "xd": o_xd,
                        "gate": o_gate, "lead": o_lead}.items()},
    }).write_parquet(out / "oracle.parquet")

    # in-sample reference R² (whole world) for calibration
    y = flat(resp[6])
    r2s = {k: float(r2_weighted(y, flat(v), weight))
           for k, v in {"full": o_full, "rev": o_rev, "xd": o_xd,
                        "gate": o_gate, "lead": o_lead}.items()}
    manifest = {
        "symbols": S, "days": D, "d0": d0, "seed": args.seed,
        "betas": {"rev": b_rev, "xd": b_xd, "gate": b_gate},
        "noises": {"resp": args.resp_noise, "venue": args.venue_noise,
                   "lead1": args.lead_noise1, "lead2": args.lead_noise2},
        "rho": {"factor": args.rho_factor, "mkt": args.rho_mkt},
        "planted_features": [f"feature_{i:02d}" for i in planted],
        "oracle_r2_insample": r2s,
        "timestamp": datetime.now().isoformat(),
    }
    (out / "synth_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"world → {out}  ({S}sym × {D}d × {T}t = {S*D*T:,} rows)")
    print("reference R² (in-sample, weighted):",
          " ".join(f"{k}={v:+.5f}" for k, v in r2s.items()))


# ---------------------------------------------------------------------------
# benchmark runner
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict] = {
    "xgb": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "n_jobs": 1},
    "gru": {"hidden_sizes": [48, 48], "dropout_rates": [0.1, 0.1],
            "hidden_sizes_linear": [], "dropout_rates_linear": [],
            "lr_refit": 1e-3},
    "gru_modelr": {"hidden_sizes": [32, 32], "dropout_rates": [0.1, 0.1],
                   "hidden_sizes_linear": [], "dropout_rates_linear": [],
                   "num_aux": 4, "lr_refit": 1e-3},
    "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 2, "ff_mult": 2,
                    "lr_refit": 3e-5},
    "patchtst": {"d_model": 64, "n_heads": 4, "n_layers": 2, "ff_mult": 2,
                 "patch_len": 16, "patch_stride": 8, "lr_refit": 3e-5},
    "itransformer": {"d_model": 64, "n_heads": 4, "n_inv_layers": 1,
                     "n_temporal_layers": 2, "ff_mult": 2, "lr_refit": 3e-5},
    "timexer": {"d_model": 64, "n_heads": 4, "n_endo_layers": 1,
                "n_exo_layers": 2, "ff_mult": 2, "patch_len": 44,
                "lr_refit": 3e-5},
}


def run_bench(args) -> None:
    data = Path(args.data)
    manifest = json.loads((data / "synth_manifest.json").read_text())
    d0, D = manifest["d0"], manifest["days"]

    base = Cfg()
    base.data_root = data
    base.min_date_id, base.max_date_id = d0, d0 + D - 1
    base.n_splits, base.test_size_dates = 1, args.test_size
    base.lagged_responders = [6, 7, 8]      # same inputs for every model
    base.epochs = args.epochs

    df = prepare_dataset(base)
    fb = FeatureBuilder(base)
    fcols = fb.feature_columns()
    lag_idx = [fcols.index(c) for c in fb.lagged_responder_columns()]
    planted_idx = [fcols.index(c) for c in manifest["planted_features"] if c in fcols]

    # ceiling on the validation tail
    valid_dates = list(range(d0 + D - args.test_size, d0 + D))
    ora = pl.read_parquet(data / "oracle.parquet").filter(pl.col("date_id").is_in(valid_dates))
    raw = pl.scan_parquet(str(data / "train.parquet" / "**" / "*.parquet")).filter(
        pl.col("date_id").is_in(valid_dates)
    ).select(["symbol_id", "date_id", "time_id", "weight", "responder_6"]).collect()
    joined = raw.join(ora, on=["symbol_id", "date_id", "time_id"])
    y = joined["responder_6"].to_numpy()
    w = joined["weight"].to_numpy()
    ceiling = {k: float(r2_weighted(y, joined[f"oracle_{k}"].to_numpy(), w))
               for k in ("full", "rev", "xd", "gate", "lead")}
    print("reference R² on the validation tail:",
          " ".join(f"{k}={v:+.5f}" for k, v in ceiling.items()), flush=True)

    results = {}
    for name in args.models.split(","):
        spec = dict(MODEL_SPECS[name])
        if name in ("transformer", "patchtst", "itransformer", "timexer", "gru", "gru_modelr"):
            spec.update(epochs=args.epochs, device=args.device)
        if name == "itransformer":
            spec["variate_channels"] = planted_idx + lag_idx
        if name == "timexer":
            spec["endo_channels"] = lag_idx
        cfg = Cfg()
        for k in ("data_root", "min_date_id", "max_date_id", "n_splits",
                  "test_size_dates", "lagged_responders", "epochs"):
            setattr(cfg, k, getattr(base, k))
        cfg.model_name, cfg.model_kwargs = name, spec
        cfg.verbose = True
        t0 = time.time()
        try:
            scores = run_cv(cfg, df=df)
            r2 = float(np.mean(scores))
        except Exception as e:  # noqa: BLE001 — keep the bench going
            print(f"[{name}] FAILED: {e}", flush=True)
            results[name] = {"error": str(e)}
            continue
        frac = r2 / ceiling["full"] if ceiling["full"] > 0 else float("nan")
        results[name] = {"r2_online": r2, "fraction_of_reference": frac,
                         "seconds": round(time.time() - t0)}
        print(f"[{name}] online R² = {r2:+.5f}  ({frac:.0%} of reference, "
              f"{results[name]['seconds']}s)", flush=True)

    out = data / f"bench_{datetime.now().strftime('%m%d_%H%M')}.json"
    out.write_text(json.dumps({"ceiling": ceiling, "models": results,
                               "epochs": args.epochs, "test_size": args.test_size,
                               "device": args.device}, indent=2))
    print(f"→ {out}", flush=True)


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a synthetic world")
    g.add_argument("--out", required=True)
    g.add_argument("--symbols", type=int, default=8)
    g.add_argument("--days", type=int, default=160)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--beta-rev", type=float, default=0.10)
    g.add_argument("--beta-xd", type=float, default=0.25)
    g.add_argument("--beta-gate", type=float, default=0.03)
    g.add_argument("--rho-factor", type=float, default=0.995)
    g.add_argument("--rho-mkt", type=float, default=0.3)
    g.add_argument("--resp-noise", type=float, default=0.05)
    g.add_argument("--venue-noise", type=float, default=0.3)
    g.add_argument("--lead-noise1", type=float, default=1.0)
    g.add_argument("--lead-noise2", type=float, default=1.5)
    g.set_defaults(func=generate)

    r = sub.add_parser("run", help="run the model zoo on a world")
    r.add_argument("--data", required=True)
    r.add_argument("--models",
                   default="xgb,gru,gru_modelr,transformer,patchtst,itransformer,timexer")
    r.add_argument("--test-size", type=int, default=40)
    r.add_argument("--epochs", type=int, default=8)
    r.add_argument("--device", default="auto")
    r.set_defaults(func=run_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
