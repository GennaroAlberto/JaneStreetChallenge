"""Walk-only rerun of a saved pipeline checkpoint with a different online
protocol — the refit-lr sweep without retraining.

Loads a `pipe.save()` checkpoint, overrides `lr_refit` (and optionally
freezes the xsec attention during refits via a zero-lr param group), then
runs the standard online walk on the tail. ~40 min on MPS vs hours for a
retrain — the standing rule's price of admission for architecture verdicts.

Usage
-----
    uv run python scripts/rewalk_checkpoint.py \\
        --ckpt artifacts/bench/arch_wave2/checkpoints/gru_modelr_xsec_xsec_gru_s42.pkl \\
        --lr-refit 3e-4 --tag xsec_gru_r3em4 --out artifacts/bench/arch_wave2/rewalks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from train_from_memmap import walk_forward  # noqa: E402

from janestreet.config import COL_DATE, Cfg  # noqa: E402
from janestreet.pipeline import FullPipeline, prepare_dataset  # noqa: E402


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--lr-refit", type=float, default=None)
    p.add_argument("--freeze-attn-refit", action="store_true",
                   help="refit with the xsec attention frozen (lr-0 group)")
    p.add_argument("--device", default="mps")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = FullPipeline.load(Path(args.ckpt))
    model = pipe.model
    model.device = args.device
    model.model = model.model.to(args.device)
    if args.lr_refit is not None:
        model.lr_refit = args.lr_refit
    if args.freeze_attn_refit:
        model.refit_lr_groups = [("xsec", 0.0)]
    print(f"[{args.tag}] lr_refit={model.lr_refit}  "
          f"refit_groups={getattr(model, 'refit_lr_groups', None)}", flush=True)

    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
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
    print(f"[{args.tag}] online R² = {r2:+.5f}", flush=True)
    np.savez_compressed(out / f"{args.tag}_online.npz",
                        preds=p_o.astype(np.float32), y=y_o.astype(np.float32),
                        w=w_o.astype(np.float32),
                        valid_dates=valid_dates.astype(np.int32))
    (out / f"{args.tag}.json").write_text(json.dumps({
        "ckpt": args.ckpt, "lr_refit": model.lr_refit,
        "freeze_attn_refit": bool(args.freeze_attn_refit),
        "r2_online": r2, "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
