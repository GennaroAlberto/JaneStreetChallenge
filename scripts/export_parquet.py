"""Export a memmap precompute to a partitioned Parquet dataset.

Turns the standardized-feature memmap (from ``precompute_dataset.py``) into a
Parquet dataset partitioned by ``date_id`` — portable, self-describing, and
zstd-compressed (typically ~3x smaller than the raw Float16 memmap, so it
uploads to Google Drive quickly). The trainer reads either format
transparently; this is the format to put on Drive for the "mount + train on
Colab GPU" workflow, so preprocessing is done once here and never repeated.

Layout written to ``--out``::

    data/date_id=700/part.parquet   (one file per date)
    data/date_id=701/part.parquet
    ...
    preprocessor.pkl                (copied verbatim — for the eval path)
    manifest.json                   (format="parquet", + feature/lag/aux schema)

Each row carries: symbol_id, date_id, time_id, weight, responder_6 (target),
the aux responders, and the standardized feature columns — everything the
trainer needs to build a FitData for a date window.

Usage
-----

    uv run python scripts/export_parquet.py \\
        --memmap artifacts/precomputed/pool700_lags \\
        --out artifacts/precomputed/pool700_lags_parquet
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", type=str, required=True, help="an existing memmap precompute dir")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--compression", type=str, default="zstd")
    args = p.parse_args()

    src = Path(args.memmap)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    K = manifest["K"]
    feature_cols = manifest["feature_cols"]
    aux_cols = manifest["aux_cols"]

    X = np.memmap(src / manifest["X_file"],
                  dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                  mode="r", shape=(manifest["N"], K))
    resp = np.load(src / "resp.f16.npy", mmap_mode="r")
    y = np.load(src / "y.f32.npy", mmap_mode="r")
    w = np.load(src / "w.f32.npy", mmap_mode="r")
    symbols = np.load(src / "symbols.i16.npy", mmap_mode="r")
    dates = np.load(src / "dates.i16.npy", mmap_mode="r")
    times = np.load(src / "times.i16.npy", mmap_mode="r")

    ranges = manifest["date_row_ranges"]
    t0 = time.time()
    date_ids = sorted(int(d) for d in ranges)
    for i, d in enumerate(date_ids):
        s, e = ranges[str(d)]
        cols = {
            "symbol_id": np.asarray(symbols[s:e], dtype=np.int16),
            "date_id": np.asarray(dates[s:e], dtype=np.int16),
            "time_id": np.asarray(times[s:e], dtype=np.int16),
            "weight": np.asarray(w[s:e], dtype=np.float32),
            manifest["target"]: np.asarray(y[s:e], dtype=np.float32),
        }
        for a_i, a in enumerate(aux_cols):
            cols[a] = np.asarray(resp[s:e, a_i], dtype=np.float32)
        Xblk = np.asarray(X[s:e], dtype=np.float32)
        for f_i, f in enumerate(feature_cols):
            cols[f] = Xblk[:, f_i]
        df = pl.DataFrame(cols)
        pdir = out / "data" / f"date_id={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(pdir / "part.parquet", compression=args.compression)
        if (i + 1) % 100 == 0:
            print(f"  wrote {i + 1}/{len(date_ids)} dates ({time.time()-t0:.0f}s)", flush=True)

    # Copy the preprocessor and write a parquet-flavoured manifest.
    shutil.copy(src / "preprocessor.pkl", out / "preprocessor.pkl")
    pq_manifest = dict(manifest)
    pq_manifest["format"] = "parquet"
    pq_manifest.pop("X_file", None)
    (out / "manifest.json").write_text(json.dumps(pq_manifest, indent=2))

    disk = sum(f.stat().st_size for f in (out / "data").rglob("*.parquet")) / 1e9
    src_gb = (src / manifest["X_file"]).stat().st_size / 1e9
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"  parquet: {disk:.2f} GB (memmap was {src_gb:.2f} GB)  → {len(date_ids)} date partitions")
    print(f"  put {out} on Drive and point the trainer's --data at it")


if __name__ == "__main__":
    main()
