"""Re-encode AE latents for a different harness window.

The supervised AE (`ae_features.py`) trains once and saves its weights;
the latents.npz it writes covers only the window it was ablated on. The
pool rebuild asserts row-exact latents, so a wider window (e.g. the 280d
production window) needs a re-encode — the encoder is a fixed function of
the already-standardized pool columns, so this is pure inference: no
retraining, no leakage, valid-side rows encoded by the same frozen net.

Usage
-----
    uv run python scripts/encode_ae_latents.py \\
        --encoder artifacts/bench/ae_lab_sup/ae_encoder.pt \\
        --train-lo 1318 --train-hi 1598 \\
        --out artifacts/bench/ae_lab_sup/latents_280.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from ae_features import AE, rows_for_dates  # noqa: E402


@torch.no_grad()
def encode_rows(model, X_mm, rows, bs=65536):
    outs = []
    for i in range(0, len(rows), bs):
        chunk = np.nan_to_num(
            np.ascontiguousarray(X_mm[rows[i:i + bs]]).astype(np.float32))
        _, z = model(torch.from_numpy(chunk))
        outs.append(z.numpy())
    return np.concatenate(outs).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--encoder", default="artifacts/bench/ae_lab_sup/ae_encoder.pt")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--train-lo", type=int, default=1318)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--out", default="artifacts/bench/ae_lab_sup/latents_280.npz")
    args = p.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))

    model = AE(K, args.hidden, args.latent)
    model.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    model.eval()

    tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    print(f"encoding {len(tr):,} train + {len(va):,} valid rows", flush=True)
    z_tr = encode_rows(model, X_mm, tr)
    z_va = encode_rows(model, X_mm, va)
    np.savez_compressed(args.out, rows_train=tr, rows_valid=va,
                        z_train=z_tr, z_valid=z_va)
    print(f"saved {args.out}: train {z_tr.shape}, valid {z_va.shape}")


if __name__ == "__main__":
    main()
