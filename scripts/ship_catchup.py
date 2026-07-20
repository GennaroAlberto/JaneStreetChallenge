"""Catch-up replay for submission shipping: online-refit a trained
checkpoint through the held-out tail and save the CAUGHT-UP weights.

The shipped model should arrive at the private test with no gap between
its last gradient step and the test's first day. This replays the exact
walk protocol (update on d-1, predict d) over 1598–1698, adds the final
update on 1698 that the scoring walk never uses, and saves the walked
pipeline. Provenance rule (graveyard §18): post-walk weights are
CORRECT for shipping forward and only ever wrong for re-scoring the
past — these checkpoints are stamped accordingly.

Usage
-----
    uv run python scripts/ship_catchup.py \\
        --ckpt artifacts/out/gru_wide/checkpoints/gru_modelr_gru_wide.pkl \\
        --expect-r2 0.00999 --out artifacts/ship
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))
from janestreet.config import COL_DATE, Cfg  # noqa: E402
from janestreet.pipeline import FullPipeline, prepare_dataset  # noqa: E402


class _CPUUnpickler(pickle.Unpickler):
    """Deserialize checkpoints pickled with CUDA tensors on a CUDA-less box."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pipe_cpu(path: Path) -> FullPipeline:
    with path.open("rb") as f:
        blob = _CPUUnpickler(f).load()
    return FullPipeline(cfg=blob["cfg"], model=blob["model"],
                        feature_cols=blob["feature_cols"],
                        aux_cols=blob["aux_cols"],
                        target_col=blob["target_col"],
                        preprocessor=blob["preprocessor"])


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--lo", type=int, default=1598,
                   help="first replay date (update source for lo+1)")
    p.add_argument("--hi", type=int, default=1698)
    p.add_argument("--device", default="mps")
    p.add_argument("--expect-r2", type=float, default=None,
                   help="known online R² for 1599-1698 from the original "
                        "walk; sanity-asserted within ±0.0015 when given")
    p.add_argument("--out", default="artifacts/ship")
    args = p.parse_args()

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    name = Path(args.ckpt).stem

    pipe = load_pipe_cpu(Path(args.ckpt))
    pipe.model.device = args.device
    pipe.model.model = pipe.model.model.to(args.device)
    pipe = copy.deepcopy(pipe)   # never mutate the loaded original path

    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    cfg_eval = Cfg()
    cfg_eval.lagged_responders = list(manifest.get("lagged_responders", []))
    lookback = (cfg_eval.rolling_window + 967) // 968 + 1
    cfg_eval.min_date_id = args.lo - lookback
    cfg_eval.max_date_id = args.hi
    df = prepare_dataset(cfg_eval, storage_precision="float32")
    dates = np.arange(args.lo, args.hi + 1)
    dates = dates[np.isin(
        dates, df.select(pl.col(COL_DATE).unique()).to_series().to_numpy())]

    preds, ys, ws = [], [], []
    for i, dt in enumerate(dates):
        day = df.filter(pl.col(COL_DATE) == dt)
        if i > 0:
            prev = df.filter(pl.col(COL_DATE) == int(dt) - 1)
            if prev.height > 0:
                pipe.update(prev)
        if dt >= args.lo + 1:                 # score 1599+ (matches the walk)
            preds.append(pipe.predict(day))
            ys.append(day.select(pipe.target_col).to_series().to_numpy())
            ws.append(day.select("weight").to_series().to_numpy())
    # the final update the scoring walk never applies: learn from the last day
    last = df.filter(pl.col(COL_DATE) == int(dates[-1]))
    pipe.update(last)

    r2 = r2_weighted(np.concatenate(ys), np.concatenate(preds), np.concatenate(ws))
    print(f"[{name}] replay online R² ({args.lo + 1}-{args.hi}) = {r2:+.5f}",
          flush=True)
    if args.expect_r2 is not None:
        assert abs(r2 - args.expect_r2) < 1.5e-3, (
            f"replay {r2:+.5f} vs expected {args.expect_r2:+.5f} — protocol "
            "drift, do not ship")

    pipe.model.model = pipe.model.model.cpu()
    pipe.model.device = "cpu"
    ck = out / "checkpoints" / f"{name}_caughtup.pkl"
    pipe.save(ck)
    (out / f"{name}_caughtup.json").write_text(json.dumps({
        "source_ckpt": args.ckpt, "replayed": [int(dates[0]), int(dates[-1])],
        "final_update_date": int(dates[-1]), "replay_r2": r2,
        "provenance": "post-walk weights: ship-forward only (graveyard §18)",
        "timestamp": datetime.now().isoformat()}, indent=2))
    print(f"[{name}] caught-up checkpoint → {ck}", flush=True)


if __name__ == "__main__":
    main()
