"""Materialize the 241-column enriched pool as a NEW memmap dataset for the RNN trainers.

WHY: the pool-rebuild ablation (scripts/rebuild_pool_ablation.py) confirmed that the engineered
blocks — combos, reversal decomp, rsig lag-diffs, SMA innovations, cluster one-hots, AE latents,
stage-1 chain predictions, cross-sectional ranks — are jointly worth ~+0.002 weighted R², but so
far only XGB ever saw them: ``assemble()`` builds the matrix in RAM per run and throws it away.
``train_from_memmap`` / ``train_cheap_member`` consume a precomputed memmap directory, so the RNN
streams can only use the enriched pool if it exists ON DISK in the same schema as
``pool700_lags``. This script writes that dataset once, per-date, in float16.

Column layout (identical order to ``assemble()`` in rebuild_pool_ablation, step=1)::

    [ base 134 | combos 50 | decomp 14 | rsig 5 | innov 9 | cluster 6 | ae 8 | chain 9 | ranks 6 ]
      = 241

The block functions are IMPORTED from the ablation scripts (combo_block, decomp_features,
signal_features, innovation_features, rank_block, date_spans_subsampled) — never reimplemented —
so the materialized matrix is definitionally the matrix XGB was scored on. Per-date slab writing
is exact (not approximate): decomp/innov/rank all loop per date internally, and combos/rsig/
cluster are row-wise, so ``assemble([d])`` row-equals the d-slice of ``assemble(all_dates)``.

Alignment (CRITICAL): the AE latents npz and the chain cache were produced on the source-pool row
convention ``rows_for_dates(lo, train_until)`` (train, OOF stage-1) + ``rows_for_dates(
train_until+1, hi)`` (tail, full-train refit stage-1). We re-derive those row arrays from the
SOURCE manifest and hard-assert against ``rows_train``/``rows_valid`` recorded in the latents npz
(the chain cache stores no rows — length-asserted, as rebuild does). Output rows are the same
per-date concatenation, so position p in the output maps to ``z_train[p]`` for p < n_train_rows
and ``z_valid[p - n_train_rows]`` after. ``train_until`` is kept at the source value: chain/AE
columns past it come from full-train fits (deploy-style tail), so those dates are eval-only —
the manifest cap is what stops ``train_from_memmap`` from training on them.

Memory (16 GB Mac, budget < ~6 GB peak): the 241-col f16 output for 601 dates is ~10.7 GB ON
DISK (22,217,536 x 241 x 2 B) but is written one date at a time — a slab is at most
40 sym x 968 t x 241 x 4 B ~= 37 MB f32 (+ ~19 MB f16 cast). Resident arrays: z_train
18.5M x 8 f32 = 592 MB, z_valid 114 MB, chain_tr 18.5M x 9 f32 = 635 MB, chain_va 128 MB,
transient npz-decompress + row-assert buffers ~740 MB (freed), aux copies <= 178 MB each.
Peak ~= 2.3 GB — comfortably inside budget; the full pool never exists in f32 RAM.

Usage
-----
    uv run python scripts/enrich_pool.py \\
        --data artifacts/precomputed/pool700_lags --lo 1098 --hi 1698 \\
        --latents artifacts/bench/ae_lab_sup/latents_500.npz \\
        --chain-cache artifacts/bench/pool_rebuild_500/chain_cache.npz \\
        --out artifacts/precomputed/pool241

    # afterwards: sample 3 dates, re-derive all blocks multi-date (assemble()'s exact call
    # shape) and assert f16 equality against the memmap
    uv run python scripts/enrich_pool.py --out artifacts/precomputed/pool241 --verify

    # smoke (writes only the first 2 dates; still loads the full latents/chain for the
    # alignment asserts, ~1.6 GB resident — do NOT run while a training job owns the machine):
    uv run python scripts/enrich_pool.py --out /tmp/pool241_smoke --limit-dates 2
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from ablate_innovations import DELTAS as INNOV_DELTAS  # noqa: E402
from ablate_innovations import TRIO, innovation_features  # noqa: E402
from ablate_responder_signal import signal_features  # noqa: E402
from ablate_reversal_decomp import BLOCK, decomp_features  # noqa: E402
from rebuild_pool_ablation import (  # noqa: E402
    RANK_COLS, combo_block, date_spans_subsampled, rank_block, rows_for_dates,
)

RSIG_NAMES = ["rsig_momA_short", "rsig_momA_med", "rsig_momB_short", "rsig_momB_med",
              "rsig_spread_fine"]
N_CHAIN = 9


@dataclass
class Source:
    """The source memmap pool: manifest + open read-only arrays."""

    dir: Path
    manifest: dict
    names: list[str]
    K: int
    X_mm: np.memmap
    syms: np.ndarray


@dataclass
class Ctx:
    """Everything assemble_dates() needs beyond the source pool.

    ``z_tr``/``ch_tr`` are indexed POSITIONALLY by the output row space (train prefix),
    ``z_va``/``ch_va`` by output position minus ``n_train_rows`` — valid because output rows
    are the same per-date concatenation the npz/caches were built on (hard-asserted at load).
    """

    accepted: list[dict]
    cluster_map: dict[int, int]
    n_clusters: int
    block_idx: list[int]
    trio_idx: list[int]
    z_tr: np.ndarray
    z_va: np.ndarray
    ch_tr: np.ndarray
    ch_va: np.ndarray
    new_ranges: dict[str, tuple[int, int]]
    n_train_rows: int


def load_source(data_dir: Path) -> Source:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    K = manifest["K"]
    X_mm = np.memmap(data_dir / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    syms = np.load(data_dir / "symbols.i16.npy", mmap_mode="r")
    return Source(data_dir, manifest, manifest["feature_cols"], K, X_mm, syms)


def build_feature_names(src: Source, ctx: Ctx) -> tuple[list[str], list[tuple[str, int, int]]]:
    """241 column names + [(block, lo, hi)] slices, derived from the SAME artifacts that
    compute the blocks — widths can never drift from the code."""
    groups = [
        ("base", list(src.names)),
        ("combos", [f"combo_{i:02d}" for i in range(len(ctx.accepted))]),
        ("decomp", [f"dec_mkt_{c}" for c in BLOCK] + [f"dec_idio_{c}" for c in BLOCK]),
        ("rsig", list(RSIG_NAMES)),
        ("innov", [f"innov_{c}_d{d}" for d in INNOV_DELTAS for c in TRIO]),
        ("cluster", [f"clus_{j}" for j in range(1, ctx.n_clusters + 1)]),
        ("ae", [f"ae_{i}" for i in range(ctx.z_tr.shape[1])]),
        ("chain", [f"chain_r{i}" for i in range(N_CHAIN)]),
        ("ranks", [f"rank_{c}" for c in RANK_COLS]),
    ]
    names, slices, c = [], [], 0
    for block, cols in groups:
        slices.append((block, c, c + len(cols)))
        names += cols
        c += len(cols)
    return names, slices


def load_ctx(src: Source, args: argparse.Namespace,
             new_ranges: dict[str, tuple[int, int]]) -> Ctx:
    """Load combos/clusters/latents/chain and hard-assert row alignment.

    The latents npz records the exact source-pool row indices it was encoded on; equality with
    our re-derived selection is THE correctness guarantee for the positional ae/chain mapping.
    Transient cost: rows_train (148 MB i64) + our concatenation (148 MB) — freed immediately.
    """
    accepted = json.loads(Path(args.combos).read_text())["accepted"]
    cluster_map = {int(k): int(v) for k, v in
                   json.loads(Path(args.clusters).read_text())["cluster_map"].items()}
    split = src.manifest["train_until"]
    _, tr_rows = rows_for_dates(src.manifest, args.lo, split)
    _, va_rows = rows_for_dates(src.manifest, split + 1, args.hi)

    blob = np.load(args.latents)
    if not (np.array_equal(blob["rows_train"], tr_rows)
            and np.array_equal(blob["rows_valid"], va_rows)):
        raise ValueError(f"{args.latents}: rows_train/rows_valid do not match "
                         f"rows_for_dates({args.lo},{split})+({split + 1},{args.hi}) "
                         "of the source pool — wrong latents file for this window")
    z_tr, z_va = blob["z_train"], blob["z_valid"]

    cache = np.load(args.chain_cache)
    ch_tr, ch_va = cache["chain_tr"], cache["chain_va"]
    if len(ch_tr) != len(tr_rows) or len(ch_va) != len(va_rows):
        raise ValueError(f"{args.chain_cache}: chain_tr/chain_va lengths "
                         f"({len(ch_tr)}, {len(ch_va)}) do not match the harness rows "
                         f"({len(tr_rows)}, {len(va_rows)})")
    n_train_rows = len(tr_rows)
    del tr_rows, va_rows, blob, cache
    gc.collect()

    return Ctx(accepted=accepted, cluster_map=cluster_map, n_clusters=max(cluster_map.values()),
               block_idx=[src.names.index(b) for b in BLOCK],
               trio_idx=[src.names.index(t) for t in TRIO],
               z_tr=z_tr, z_va=z_va, ch_tr=ch_tr, ch_va=ch_va,
               new_ranges=new_ranges, n_train_rows=n_train_rows)


def _tail_slice(ctx: Ctx, d: int, train_arr: np.ndarray, valid_arr: np.ndarray) -> np.ndarray:
    """Positional ae/chain rows for one date in the output row space (see Ctx docstring)."""
    p0, p1 = ctx.new_ranges[str(d)]
    if p1 <= ctx.n_train_rows:
        return train_arr[p0:p1]
    if p0 < ctx.n_train_rows:  # a date cannot straddle the train/tail boundary
        raise AssertionError(f"date {d} straddles the train/tail row boundary")
    return valid_arr[p0 - ctx.n_train_rows:p1 - ctx.n_train_rows]


def assemble_dates(dates_sel: list[int], src: Source, ctx: Ctx, total: int) -> np.ndarray:
    """Float32 (n, 241) matrix for ``dates_sel`` — assemble() from rebuild_pool_ablation,
    minus the variant-prefix machinery, step=1.

    Reuses the ablation block functions verbatim so writer and verifier share one definition.
    Called with ONE date by the writer (slab ~37 MB) and with the 3 sampled dates by
    ``--verify`` (the multi-date call shape assemble() itself used, ~110 MB).
    """
    ranges = src.manifest["date_row_ranges"]
    spans = date_spans_subsampled(src.manifest, dates_sel, 1)
    n = spans[-1][1]
    K = src.K
    n_lag = len(src.manifest["lag_cols"])
    M = np.empty((n, total), dtype=np.float32)
    for d, (a, b) in zip(dates_sel, spans):
        r0, r1 = ranges[str(d)]
        M[a:b, :K] = src.X_mm[r0:r1]
    c = K
    M[:, c:c + len(ctx.accepted)] = combo_block(M[:, :K], src.names, ctx.accepted)
    c += len(ctx.accepted)
    M[:, c:c + 2 * len(BLOCK)] = decomp_features(src.X_mm, src.manifest, dates_sel, ctx.block_idx)
    c += 2 * len(BLOCK)
    M[:, c:c + len(RSIG_NAMES)] = signal_features(M[:, K - n_lag:K])
    c += len(RSIG_NAMES)
    n_innov = len(TRIO) * len(INNOV_DELTAS)
    M[:, c:c + n_innov] = innovation_features(src.X_mm, src.manifest, dates_sel, ctx.trio_idx)
    c += n_innov
    sym_parts = [src.syms[slice(*ranges[str(d)])] for d in dates_sel]
    cl = np.array([ctx.cluster_map.get(int(s), 0) for s in np.concatenate(sym_parts)])
    for j in range(ctx.n_clusters):
        M[:, c + j] = (cl == j + 1).astype(np.float32)
    c += ctx.n_clusters
    n_ae = ctx.z_tr.shape[1]
    for d, (a, b) in zip(dates_sel, spans):
        M[a:b, c:c + n_ae] = _tail_slice(ctx, d, ctx.z_tr, ctx.z_va)
        M[a:b, c + n_ae:c + n_ae + N_CHAIN] = _tail_slice(ctx, d, ctx.ch_tr, ctx.ch_va)
    c += n_ae + N_CHAIN
    M[:, c:c + len(RANK_COLS)] = rank_block(M, src.names, spans, 1)
    c += len(RANK_COLS)
    assert c == total
    return M


F16_MAX = float(np.finfo(np.float16).max)


def to_f16(M: np.ndarray) -> np.ndarray:
    """Clip to the f16 range before casting: base cols were already f16, but combo products
    can exceed 65504 in rare rows and would otherwise silently become inf on disk."""
    return np.clip(M, -F16_MAX, F16_MAX).astype(np.float16)


def write_pool(args: argparse.Namespace) -> None:
    src = load_source(Path(args.data))
    dates_all, _ = rows_for_dates(src.manifest, args.lo, args.hi)
    if args.limit_dates:
        dates_write = dates_all[:args.limit_dates]
    else:
        dates_write = dates_all

    ranges = src.manifest["date_row_ranges"]
    new_ranges: dict[str, tuple[int, int]] = {}
    c = 0
    for d in dates_write:  # cumulative, robust to gaps — do NOT assume source contiguity
        r0, r1 = ranges[str(d)]
        new_ranges[str(d)] = (c, c + (r1 - r0))
        c += r1 - r0
    n_out = c

    ctx = load_ctx(src, args, new_ranges)
    names_out, block_slices = build_feature_names(src, ctx)
    total = len(names_out)
    print(f"blocks: {'  '.join(f'{b}[{lo}:{hi}]' for b, lo, hi in block_slices)} = {total}",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    need = n_out * total * 2 + n_out * 30  # X.f16 + aux copies, ~5% headroom below
    free = shutil.disk_usage(out).free
    if free < need * 1.05:
        raise RuntimeError(f"need ~{need / 1e9:.1f} GB in {out}, only {free / 1e9:.1f} GB free")

    X_out = np.memmap(out / "X.f16.dat", dtype=np.float16, mode="w+", shape=(n_out, total))
    for i, d in enumerate(dates_write):
        p0, p1 = new_ranges[str(d)]
        X_out[p0:p1] = to_f16(assemble_dates([d], src, ctx, total))
        if (i + 1) % 50 == 0 or i + 1 == len(dates_write):
            X_out.flush()  # bound dirty pages: never more than ~50 dates x 18 MB unflushed
            print(f"  [{i + 1}/{len(dates_write)}] date {d} rows {p1 - p0}", flush=True)
    X_out.flush()
    del X_out
    gc.collect()

    # ---- aux copies restricted to the selected rows (order = per-date concatenation) -----
    sel = np.concatenate([np.arange(*ranges[str(d)], dtype=np.int64) for d in dates_write])
    for fname in ["y.f32.npy", "w.f32.npy", "symbols.i16.npy", "dates.i16.npy",
                  "times.i16.npy", "resp.f16.npy"]:
        arr = np.load(src.dir / fname, mmap_mode="r")
        np.save(out / fname, np.ascontiguousarray(arr[sel]))  # <=178 MB materialized per file
        del arr
        gc.collect()
    del sel
    # the frozen standardizer travels with the pool: train_from_memmap attaches it to the
    # pipeline so predict()/update() can re-standardize raw rows at serve time
    shutil.copy2(src.dir / "preprocessor.pkl", out / "preprocessor.pkl")

    m = src.manifest
    manifest = {
        "N": int(n_out), "K": int(total), "precision": "float16", "X_file": "X.f16.dat",
        "feature_cols": names_out,
        "aux_cols": m["aux_cols"],
        "lagged_responders": m["lagged_responders"],
        # base block is column-preserved at the front, so lag positions are unchanged
        "lag_cols": m["lag_cols"], "lag_col_indices": m["lag_col_indices"],
        "target": m["target"],
        "min_date": int(dates_write[0]), "max_date": int(dates_write[-1]),
        # chain/AE cols past the source train_until come from full-train stage-1/AE fits
        # (deploy-style tail) — the cap is what keeps trainers from fitting on leaky rows
        "train_until": int(min(m["train_until"], dates_write[-1])),
        "n_times": m["n_times"],
        "date_row_ranges": {k: [int(a), int(b)] for k, (a, b) in new_ranges.items()},
        "row_order": m["row_order"],
        "blocks": {b: [lo, hi] for b, lo, hi in block_slices},
        "combo_labels": [a["label"] for a in ctx.accepted],
        "enriched_from": str(src.dir),
        "latents": str(args.latents), "chain_cache": str(args.chain_cache),
        "partial": bool(args.limit_dates),
        "timestamp": datetime.now().isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}: X.f16.dat ({n_out:,} x {total}, "
          f"{n_out * total * 2 / 1e9:.1f} GB) + aux + manifest", flush=True)


def verify_pool(args: argparse.Namespace, n_sample: int = 3) -> None:
    """Sample dates, re-derive every block with assemble()'s multi-date call shape, and
    assert f16 equality with the memmap.

    WHY multi-date: the writer ran per-date; recomputing the sampled dates in ONE
    decomp/innov/rank call (exactly how rebuild_pool_ablation's assemble() calls them) proves
    the per-date == multi-date equivalence AND catches row-offset / ae-chain misalignment,
    the two real failure modes. The f32 pipeline is deterministic, so after the shared
    clip-and-cast the comparison is exact (allclose at f16 eps kept as the assert for slack).
    """
    out = Path(args.out)
    man = json.loads((out / "manifest.json").read_text())
    src = load_source(Path(args.data))
    dates_all, _ = rows_for_dates(src.manifest, args.lo, args.hi)

    new_ranges = {k: (int(a), int(b)) for k, (a, b) in man["date_row_ranges"].items()}
    ctx = load_ctx(src, args, new_ranges)
    names_out, block_slices = build_feature_names(src, ctx)
    total = len(names_out)
    if [names_out, total] != [man["feature_cols"], man["K"]]:
        raise AssertionError("manifest feature_cols/K do not match the code-derived layout")

    X_out = np.memmap(out / man["X_file"], dtype=np.float16, mode="r",
                      shape=(man["N"], man["K"]))
    have = [d for d in dates_all if str(d) in new_ranges]
    rng = np.random.default_rng(args.seed)
    sample = sorted(int(d) for d in rng.choice(have, size=min(n_sample, len(have)),
                                               replace=False))
    print(f"verify: dates {sample}", flush=True)

    ref16 = to_f16(assemble_dates(sample, src, ctx, total))  # one multi-date call, ~110 MB
    spans = date_spans_subsampled(src.manifest, sample, 1)
    ok = True
    for d, (a, b) in zip(sample, spans):
        p0, p1 = new_ranges[str(d)]
        disk = np.asarray(X_out[p0:p1], dtype=np.float32)
        ref = ref16[a:b].astype(np.float32)
        for block, lo, hi in block_slices:
            db, rb = disk[:, lo:hi], ref[:, lo:hi]
            close = np.allclose(db, rb, rtol=1e-3, atol=1e-3, equal_nan=True)
            exact = np.array_equal(db, rb, equal_nan=True)
            diff = float(np.nanmax(np.abs(db - rb))) if db.size else 0.0
            tag = "OK " if close else "FAIL"
            ok &= close
            print(f"  date {d}  {block:8s} {tag} exact={exact} max|diff|={diff:.2e}",
                  flush=True)
    if not ok:
        raise AssertionError("verification FAILED — see per-block report above")
    print(f"verify PASSED on dates {sample} at f16 tolerance", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--lo", type=int, default=1098)
    p.add_argument("--hi", type=int, default=1698)
    p.add_argument("--latents", default="artifacts/bench/ae_lab_sup/latents_500.npz")
    p.add_argument("--chain-cache", default="artifacts/bench/pool_rebuild_500/chain_cache.npz")
    p.add_argument("--combos", default="artifacts/bench/drw_lab_v2/combos.json")
    p.add_argument("--clusters", default="artifacts/bench/ablation_symbol.json")
    p.add_argument("--out", default="artifacts/precomputed/pool241")
    p.add_argument("--limit-dates", type=int, default=0,
                   help="write only the first N dates (smoke). Alignment asserts still run "
                        "on the FULL --lo/--hi selection; manifest is marked partial")
    p.add_argument("--verify", action="store_true",
                   help="verify an existing --out against a fresh multi-date re-derivation "
                        "(run with the SAME --lo/--hi/--latents/--chain-cache as the write)")
    p.add_argument("--seed", type=int, default=42, help="verify-mode date sampling seed")
    args = p.parse_args()
    if args.verify:
        verify_pool(args)
    else:
        write_pool(args)


if __name__ == "__main__":
    main()
