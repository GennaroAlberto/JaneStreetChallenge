"""Autoencoder bottleneck features (the DRW winner's biggest boost).

Train a small denoising autoencoder on the standardized feature pool (train
rows only — the AE is preprocessing and obeys the leakage boundary), then:

  1. append the 8-dim bottleneck as features and ablate on the standard XGB
     harness (base 134 vs base+AE8);
  2. persist the encoder weights and the per-row latents for the harness
     range, so downstream experiments (log-signature of the latent *path*,
     latent-space clustering) can reuse them without retraining.

The AE is a nonlinear factor model of the feature set: the bottleneck
coordinates expose "market state" directions that axis-aligned tree splits
cannot synthesize. DRW 1st place reported this as his single largest gain.

Usage
-----
    uv run python scripts/ae_features.py \\
        --data artifacts/precomputed/pool700_lags \\
        --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \\
        --out artifacts/bench/ae_lab
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from torch import nn


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi, step=1):
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    rows = np.concatenate(parts)
    return rows[::step] if step > 1 else rows


class AE(nn.Module):
    """Denoising AE with an optional target head on the bottleneck.

    A pure reconstruction objective preserves high-VARIANCE directions; at
    alpha ≈ 1% of variance it has no reason to keep the predictive ones
    (measured: plain-AE latents were flat on the XGB harness). The target
    head tilts the latent space toward alpha without giving up the manifold
    structure the reconstruction term maintains.
    """

    def __init__(self, k: int, hidden: int = 64, latent: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, k),
        )
        self.head = nn.Linear(latent, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(X, y, w, latent, hidden, epochs, noise_std, sup_weight, seed, log):
    torch.manual_seed(seed)
    torch.set_num_threads(2)          # be a good neighbour to running jobs
    model = AE(X.shape[1], hidden, latent)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.from_numpy(X)
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    n = len(ds)
    bs = 8192
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot_r = tot_s = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = ds[idx]
            xin = xb + noise_std * torch.randn_like(xb) if noise_std > 0 else xb
            opt.zero_grad()
            rec, z = model(xin)
            loss_r = nn.functional.mse_loss(rec, xb)
            loss = loss_r
            loss_s = torch.tensor(0.0)
            if sup_weight > 0:
                pred = model.head(z).squeeze(-1)
                wb = wt[idx]
                loss_s = ((pred - yt[idx]) ** 2 * wb).sum() / wb.sum()
                loss = loss_r + sup_weight * loss_s
            loss.backward()
            opt.step()
            tot_r += loss_r.item() * len(xb)
            tot_s += float(loss_s) * len(xb)
        log(f"  AE epoch {ep + 1}/{epochs}  recon MSE {tot_r / n:.4f}"
            + (f"  target MSE {tot_s / n:.4f}" if sup_weight > 0 else ""))
    model.eval()
    return model


@torch.no_grad()
def encode(model, X, bs=65536):
    outs = []
    for i in range(0, len(X), bs):
        _, z = model(torch.from_numpy(X[i:i + bs]))
        outs.append(z.numpy())
    return np.concatenate(outs).astype(np.float32)


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
    return r2_weighted(yva, pred, wva), best_it, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--supervised-weight", type=float, default=0.0,
                   help="weight of the target-prediction loss on the "
                        "bottleneck (0 = plain reconstruction AE)")
    p.add_argument("--input-cols", type=str, default=None,
                   help="comma-separated feature names to feed the AE "
                        "(default: all pool columns). Training on the "
                        "curated signal-dense core mirrors DRW's placement "
                        "of the AE *after* selection: a broad pool's "
                        "reconstruction loss over-weights redundant "
                        "clusters and ignores low-variance alpha columns")
    p.add_argument("--ae-row-step", type=int, default=4,
                   help="train the AE on every Nth train row")
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/ae_lab")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "ae_log.txt"

    def log(msg):
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["feature_cols"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    if args.input_cols:
        in_idx = [names.index(c) for c in args.input_cols.split(",") if c in names]
    else:
        in_idx = list(range(K))

    # ---- train the AE on (subsampled) train rows only ----------------------
    ae_rows = rows_for_dates(manifest, args.train_lo, args.train_hi, args.ae_row_step)
    X_ae = np.nan_to_num(
        np.ascontiguousarray(X_mm[ae_rows][:, in_idx]).astype(np.float32))
    y_ae, w_ae = y[ae_rows], w[ae_rows]
    log(f"AE training slice: {X_ae.shape[0]:,} rows × {len(in_idx)} inputs "
        f"(step {args.ae_row_step}); latent={args.latent}  "
        f"supervised_weight={args.supervised_weight}")
    model = train_ae(X_ae, y_ae, w_ae, args.latent, args.hidden, args.epochs,
                     args.noise_std, args.supervised_weight, args.seed, log)
    del X_ae
    gc.collect()
    torch.save(model.state_dict(), out / "ae_encoder.pt")

    # ---- encode harness rows, ablate ---------------------------------------
    tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    Xtr = np.nan_to_num(np.ascontiguousarray(X_mm[tr]).astype(np.float32))
    Xva = np.nan_to_num(np.ascontiguousarray(X_mm[va]).astype(np.float32))
    ytr, wtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr])
    yva, wva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va])

    Ztr = encode(model, np.ascontiguousarray(Xtr[:, in_idx]))
    Zva = encode(model, np.ascontiguousarray(Xva[:, in_idx]))
    np.savez_compressed(out / "latents.npz", rows_train=tr, rows_valid=va,
                        z_train=Ztr, z_valid=Zva)
    log(f"latents saved: train {Ztr.shape}, valid {Zva.shape}")

    results = []
    for name, xt, xv in (
        ("base", Xtr, Xva),
        ("base+ae", np.hstack([Xtr, Ztr]), np.hstack([Xva, Zva])),
    ):
        r2, best_it, m = fit_eval(xt, ytr, wtr, xv, yva, wva,
                                  args.n_estimators, args.seed)
        log(f"  {name:8s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
            f"R²={r2:+.5f}")
        row = {"variant": name, "n_features": int(xt.shape[1]),
               "best_iter": best_it, "r2_weighted": r2}
        if name == "base+ae":
            imp = m.feature_importances_
            ae_imp = imp[-args.latent:]
            row["ae_importance_rank"] = int(
                np.sum(imp[:, None] > ae_imp[None, :]) / args.latent)
            row["ae_importances"] = [round(float(v), 4) for v in ae_imp]
        results.append(row)
        del xt, xv
        gc.collect()

    results[1]["delta_vs_base"] = (results[1]["r2_weighted"]
                                   - results[0]["r2_weighted"])
    (out / "ablation.json").write_text(json.dumps({
        "data": str(data),
        "config": {"latent": args.latent, "hidden": args.hidden,
                   "epochs": args.epochs, "noise_std": args.noise_std,
                   "supervised_weight": args.supervised_weight,
                   "input_cols": ([names[i] for i in in_idx]
                                  if args.input_cols else "all")},
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log(f"delta vs base: {results[1]['delta_vs_base']:+.5f} → {out}/ablation.json")


if __name__ == "__main__":
    main()
