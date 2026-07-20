"""Walk-only rerun of a saved lstm64 cheap member at a different refit lr.

Loads the ``.pt`` state dict saved by ``train_cheap_member.py`` and re-runs
the online walk via ``walk_forward_direct`` — the 3e-4-vs-1e-3 question for
the bag members, at ~5 min per walk instead of a retrain.

Usage
-----
    uv run python scripts/rewalk_cheap_member.py --seed 1 --lr-refit 3e-4 \\
        --weights artifacts/bench/pool_rebuild_500/preds400/lstm64_s1.pt \\
        --out artifacts/bench/pool_rebuild_500/preds400
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from aux_innovation_lab import r2_weighted, rows_for_dates, walk_forward_direct  # noqa: E402

from janestreet.models.recurrent import RecurrentModel  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--weights", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr-refit", type=float, default=3e-4)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    tag = f"lstm64_s{args.seed}_r{args.lr_refit:g}"

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    ranges = manifest["date_row_ranges"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"], dtype=np.float16, mode="r",
                     shape=(manifest["N"], K))
    y_all = np.load(data / "y.f32.npy", mmap_mode="r")
    w_all = np.load(data / "w.f32.npy", mmap_mode="r")

    model = RecurrentModel(
        model_type="lstm", aux_branches=True, num_aux=4,
        hidden_sizes=[64, 64], dropout_rates=[0.1, 0.1],
        hidden_sizes_linear=[], dropout_rates_linear=[],
        lr=1e-3, weight_decay=1e-2, batch_size=1, epochs=1,
        early_stopping_patience=1, grad_clip=1.0, lr_refit=args.lr_refit,
        seed=args.seed, device=args.device)
    model.model = model._build(K)
    model.model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.model = model.model.to(args.device)

    def log(msg):
        print(msg, flush=True)

    p_, y_, w_ = walk_forward_direct(
        model, X_mm, ranges, y_all, w_all, K,
        args.valid_lo, args.valid_hi, model.device, log)
    r2 = r2_weighted(y_, p_, w_)
    print(f"{tag} online R² = {r2:+.5f}", flush=True)
    np.savez_compressed(out / f"{tag}.npz", pred=p_.astype(np.float32),
                        y=y_.astype(np.float32), w=w_.astype(np.float32))
    (out / f"{tag}.json").write_text(json.dumps({
        "tag": tag, "seed": args.seed, "lr_refit": args.lr_refit,
        "r2_online": r2, "timestamp": datetime.now().isoformat()}, indent=2))


if __name__ == "__main__":
    main()
