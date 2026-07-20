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

from janestreet.config import Cfg
from janestreet.data.ingest import scan_train_dates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", type=str, required=True, help="an existing memmap precompute dir")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--compression", type=str, default="zstd")
    p.add_argument("--min-date", type=int, default=None,
                   help="export only dates >= this (smaller Drive upload)")
    p.add_argument("--max-date", type=int, default=None)
    p.add_argument("--include-responders", action="store_true",
                   help="join all nine raw responders into each partition "
                        "(needed for --aux-targets beyond the memmap's frozen "
                        "aux set, e.g. the spread-aux experiments). Requires "
                        "the raw competition data locally.")
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
    if args.min_date is not None:
        date_ids = [d for d in date_ids if d >= args.min_date]
    if args.max_date is not None:
        date_ids = [d for d in date_ids if d <= args.max_date]

    # Optionally join all nine raw responders, aligned to memmap row order
    # per date (both sides sorted by (date, symbol, time)).
    resp_all = None
    resp_names = [f"responder_{i}" for i in range(9)]
    if args.include_responders:
        raw = (scan_train_dates(Cfg(), date_ids[0], date_ids[-1])
               .select(["date_id", "symbol_id", "time_id", *resp_names])
               .collect()
               .sort(["date_id", "symbol_id", "time_id"]))
        wanted = set(date_ids)
        resp_all = {}
        for key, g in raw.group_by("date_id", maintain_order=True):
            d_key = int(key[0] if isinstance(key, tuple) else key)
            if d_key in wanted:
                resp_all[d_key] = g.select(resp_names).to_numpy().astype(np.float32)
        print(f"  joined raw responders for {len(resp_all)} dates", flush=True)
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
        if resp_all is not None:
            R9 = resp_all[d]
            if len(R9) != e - s:
                raise ValueError(f"date {d}: raw/memmap row-count mismatch")
            # alignment assertion: raw responder_6 vs memmap y at f16 tol
            if not np.allclose(np.float16(R9[:, 6]).astype(np.float32),
                               cols[manifest["target"]], atol=2e-3):
                raise ValueError(f"date {d}: raw/memmap alignment failed")
            for r_i, rn in enumerate(resp_names):
                if rn not in cols:
                    cols[rn] = R9[:, r_i]
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
