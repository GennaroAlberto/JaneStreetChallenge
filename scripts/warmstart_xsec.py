"""Warm-start xsec transplant — attach cross-sectional attention to a
TRAINED plain twin instead of training jointly from scratch.

Mechanics: load the plain ModelR checkpoint, copy its weights into an
xsec-enabled twin (`strict=False` leaves only the attention params fresh),
zero-init each branch's attention `out_proj` so the network starts as an
exact functional copy of the trained twin, then fine-tune with the
attention at its own learning rate (`lr_groups`). The model can only
depart from the twin where attention across symbols genuinely helps —
"fitted properly" for an attached module.

Usage
-----
    uv run python scripts/warmstart_xsec.py --family lstm \\
        --out artifacts/bench/arch_wave2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from train_from_memmap import build_fitdata_for_dates, walk_forward  # noqa: E402

from janestreet.config import COL_DATE, Cfg  # noqa: E402
from janestreet.models.recurrent import RecurrentModel  # noqa: E402
from janestreet.pipeline import FullPipeline, prepare_dataset  # noqa: E402

PLAIN_CKPT = {
    "gru": "artifacts/bench/regen_ensemble/checkpoints/gru_modelr_gru_modelr.pkl",
    "lstm": "artifacts/bench/regen_ensemble/checkpoints/lstm_modelr_lstm_modelr.pkl",
}


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["gru", "lstm"], required=True)
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1318)
    p.add_argument("--train-hi", type=int, default=1597)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="backbone fine-tune lr (attention runs at --attn-lr)")
    p.add_argument("--attn-lr", type=float, default=1e-3)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/arch_wave2")
    args = p.parse_args()

    out = Path(args.out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    tag = f"xsec_{args.family}_warm"

    plain_pipe = FullPipeline.load(Path(PLAIN_CKPT[args.family]))
    plain_net = plain_pipe.model.model

    model = RecurrentModel(
        model_type=args.family, aux_branches=True, num_aux=4,
        hidden_sizes=[96, 96, 96], dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=args.lr, weight_decay=1e-2, batch_size=1, epochs=args.epochs,
        early_stopping_patience=3, grad_clip=1.0, lr_refit=1e-3,
        seed=args.seed, device=args.device, xsec_heads=4,
        lr_groups=[("xsec", args.attn_lr)])
    net = model._build(plain_net.branches[0].rnns[0].input_size)
    missing = net.load_state_dict(plain_net.state_dict(), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    assert all("xsec" in k for k in missing.missing_keys), missing.missing_keys
    for br in net.branches:
        br.xsec.out_proj.weight.data.zero_()
        br.xsec.out_proj.bias.data.zero_()
    model.model = net
    print(f"[{tag}] transplanted {args.family} twin; fresh attention params: "
          f"{len(missing.missing_keys)} tensors (out_proj zero-init)", flush=True)

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_fd = build_fitdata_for_dates(
        data_dir, manifest, list(range(args.train_lo, args.train_hi + 1)), None, None)
    valid_fd = build_fitdata_for_dates(
        data_dir, manifest, list(range(args.valid_lo, args.valid_hi + 1)), None, None)

    t0 = time.time()
    model.fit(train_fd, valid_fd, verbose=True, warm_start=True)
    fit_s = time.time() - t0
    print(f"[{tag}] fine-tune {fit_s / 60:.1f} min", flush=True)
    del train_fd, valid_fd

    pipe = plain_pipe
    pipe.model = model
    pipe.save(out / "checkpoints" / f"{tag}.pkl")

    cfg_eval = Cfg()
    cfg_eval.lagged_responders = list(manifest.get("lagged_responders", []))
    lookback = (cfg_eval.rolling_window + 967) // 968 + 1
    cfg_eval.min_date_id = args.valid_lo - lookback
    cfg_eval.max_date_id = args.valid_hi
    df_eval = prepare_dataset(cfg_eval, storage_precision="float32")
    valid_dates = np.arange(args.valid_lo, args.valid_hi + 1)
    valid_dates = valid_dates[np.isin(
        valid_dates,
        df_eval.select(pl.col(COL_DATE).unique()).to_series().to_numpy())]

    p_o, y_o, w_o = walk_forward(pipe, df_eval, valid_dates, pipe.target_col)
    r2 = r2_weighted(y_o, p_o, w_o)
    print(f"[{tag}] online R² = {r2:+.5f}", flush=True)
    np.savez_compressed(out / "preds" / f"{tag}_online.npz",
                        preds=p_o.astype(np.float32), y=y_o.astype(np.float32),
                        w=w_o.astype(np.float32),
                        valid_dates=valid_dates.astype(np.int32))
    (out / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "family": args.family, "lr": args.lr,
        "attn_lr": args.attn_lr, "epochs": args.epochs,
        "train": [args.train_lo, args.train_hi], "fit_s": fit_s,
        "r2_online": r2, "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
