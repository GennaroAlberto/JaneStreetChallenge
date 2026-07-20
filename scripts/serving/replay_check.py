"""Correctness gate: offline ``prepare_dataset`` vs a streaming feature engine.

Builds the offline engineered frame for ``[--min-date, --max-date]``, then
streams the SAME raw rows one ``(date_id, time_id)`` batch at a time through
the selected engine (raw engineered values are compared, so feature-engineering
correctness is isolated from clip/standardize, which is a shared code path).
Reports max|Δ| and the fraction of cells within
``--tol-warn`` per column family, and exits nonzero if any family's max|Δ|
exceeds ``--tol-fail`` (a NaN on exactly one side counts as a failure; NaN on
both sides counts as agreement — that's the "rolling window not yet full"
state, which the standardizer maps to 0 on both paths).

``--engine`` picks the streaming implementation:

* ``feature_state`` (default) — the row-panel reference engine
  (``feature_state.FeatureState``, preprocessor=None). Its daily-refit
  assembly (``new_date`` → ``RefitBlock``) is exercised on every date but
  only sanity-checked for shape, not compared.
* ``kernel_state`` — the slate engine the Kaggle kernel serves with
  (``kernel_state.FeatureState``). ``push`` returns the FULL slot slate; the
  harness compares the rows of the symbols present in each batch
  (``X_slate[syms]``) and sanity-checks that ``day_store`` accumulated one
  record per pushed batch before each ``new_date`` clears it. Not supported
  with ``--rsig`` (the slate engine has no rsig columns).

Usage (caller runs this — it is NOT run at write time):

    uv run python scripts/serving/replay_check.py --min-date 1690 --max-date 1698
    uv run python scripts/serving/replay_check.py --min-date 1690 --max-date 1698 \
        --engine kernel_state

Notes
-----
* Both matrices are held in RAM: ~40 MB per date at K=134. Keep the window
  to <= ~15 dates on a loaded machine.
* Rolling stats need ``rolling_window`` rows per symbol (~1.05 dates at 968
  times/date) before they leave the NaN state, so any window of >= 3 dates
  exercises live rolling values. ``--rolling-window`` can shrink the window
  (consistently on BOTH paths, since it flows through cfg) for a faster gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_state import _DIFF_RE, _LAG_RE, _MAVG_RE, _RSIG, _STD_RE, FeatureState  # noqa: E402
from kernel_state import FeatureState as KernelFeatureState  # noqa: E402

from janestreet.config import COL_DATE, COL_ID, COL_TIME, COL_WEIGHT, Cfg  # noqa: E402
from janestreet.data.features import FeatureBuilder  # noqa: E402
from janestreet.pipeline import prepare_dataset  # noqa: E402

RESP_COLS = [f"responder_{i}" for i in range(9)]


def family_of(name: str) -> str:
    if _DIFF_RE.match(name):
        return "diff_rolling_avg"
    if _STD_RE.match(name):
        return "rolling_std"
    if _MAVG_RE.match(name):
        return "avg_per_date_time"
    if _LAG_RE.match(name):
        return "responder_lag1d"
    if name == "feature_time_id":
        return "feature_time_id"
    if name in _RSIG:
        return "rsig"
    return "raw"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--min-date", type=int, required=True)
    p.add_argument("--max-date", type=int, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--rolling-window", type=int, default=None,
                   help="override cfg.rolling_window (applied to BOTH paths)")
    p.add_argument("--lagged", default="0,1,2,3,4,5,6,7,8",
                   help="comma list of lagged responders (default = 134-col recipe)")
    p.add_argument("--rsig", action="store_true",
                   help="also enable responder_signal_features (139-col recipe)")
    p.add_argument("--engine", choices=("feature_state", "kernel_state"),
                   default="feature_state",
                   help="streaming engine to validate: the row-panel reference "
                        "(feature_state) or the Kaggle kernel's slate engine "
                        "(kernel_state)")
    p.add_argument("--tol-warn", type=float, default=1e-4)
    p.add_argument("--tol-fail", type=float, default=1e-3)
    p.add_argument("--out", default=None, help="optional JSON report path")
    args = p.parse_args()

    if args.max_date < args.min_date:
        p.error("--max-date must be >= --min-date")
    if args.engine == "kernel_state" and args.rsig:
        p.error("--engine kernel_state does not implement the rsig columns "
                "(134-col recipe only)")

    cfg = Cfg(
        min_date_id=args.min_date,
        max_date_id=args.max_date,
        lagged_responders=[int(t) for t in args.lagged.split(",") if t != ""],
        responder_signal_features=args.rsig,
        verbose=False,
    )
    if args.rolling_window is not None:
        cfg.rolling_window = args.rolling_window
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)

    # ---- offline path ----------------------------------------------------
    df = prepare_dataset(cfg, downcast=True)
    if df.height == 0:
        print("no rows in the requested date window", file=sys.stderr)
        return 2
    feature_cols = FeatureBuilder(cfg).feature_columns()
    print(f"window {args.min_date}..{args.max_date}: {df.height} rows, "
          f"K={len(feature_cols)} feature columns")

    df = df.sort([COL_DATE, COL_TIME, COL_ID])
    offline = df.select(feature_cols).to_numpy().astype(np.float32)
    syms_all = df[COL_ID].to_numpy().astype(np.int64)

    # ---- streaming path --------------------------------------------------
    if args.engine == "kernel_state":
        # The slate engine reads the weight column on every push (only into
        # w_slate, never into features) — synthesize it if the frame lacks it.
        if COL_WEIGHT not in df.columns:
            df = df.with_columns(pl.lit(1.0, dtype=pl.Float32).alias(COL_WEIGHT))
        fs = KernelFeatureState(feature_cols, roll_window=cfg.rolling_window)
    else:
        fs = FeatureState(
            preprocessor=None,
            feature_cols=feature_cols,
            lagged_responders=cfg.lagged_responders,
            rolling_window=cfg.rolling_window,
        )
    online = np.empty_like(offline)

    by_date = {int(part[COL_DATE][0]): part
               for part in df.partition_by(COL_DATE, maintain_order=True)}
    pos = 0
    n_refit_blocks = 0
    n_batches_prev = 0
    for d in sorted(by_date):
        prev = by_date.get(d - 1)
        lags = (
            prev.select([COL_ID, COL_TIME, *RESP_COLS])
            if prev is not None else None
        )
        if args.engine == "kernel_state":
            # Refit assembly lives in the kernel notebook; here we only check
            # the day_store cache it would consume: one record per batch.
            assert len(fs.day_store) == n_batches_prev, (
                f"date {d}: day_store has {len(fs.day_store)} records, "
                f"expected {n_batches_prev} (one per pushed batch)"
            )
            if n_batches_prev and lags is not None:
                n_refit_blocks += 1
            fs.new_date(lags)   # returns None — installs lags, clears day_store
        else:
            block = fs.new_date(lags)
            if block is not None:
                n_refit_blocks += 1
                expect = block.n_times * block.symbol_ids.size
                assert block.X.shape == (expect, len(feature_cols)), (
                    f"date {d}: refit block shape {block.X.shape} != "
                    f"({expect}, {len(feature_cols)})"
                )
                assert block.y.shape == (expect,) and block.w.shape == (expect,)
        n_batches_prev = 0
        for batch in by_date[d].partition_by(COL_TIME, maintain_order=True):
            syms = batch[COL_ID].to_numpy().astype(np.int64)
            if args.engine == "kernel_state":
                # Full slate out; compare the present symbols' rows, in batch
                # (= offline sort) order. Absent slots are the kernel's concern.
                x_slate, _w_slate = fs.push(batch)
                x_step = x_slate[syms]
                assert fs.active[syms].all(), (
                    f"date {d}: pushed symbols missing from the active set"
                )
            else:
                x_step, syms = fs.push(batch)
            s = syms.size
            if not np.array_equal(syms, syms_all[pos:pos + s]):
                print(f"row alignment broke at date {d}, "
                      f"time {int(batch[COL_TIME][0])}", file=sys.stderr)
                return 2
            online[pos:pos + s] = x_step
            pos += s
            n_batches_prev += 1
    assert pos == offline.shape[0], f"streamed {pos} rows, offline {offline.shape[0]}"
    print(f"[{args.engine}] streamed {pos} rows; "
          f"{n_refit_blocks} refit blocks assembled")

    # ---- compare per column family ---------------------------------------
    fam_cols: dict[str, list[int]] = {}
    for j, name in enumerate(feature_cols):
        fam_cols.setdefault(family_of(name), []).append(j)

    print(f"\n{'family':<20} {'cols':>4} {'max|d|':>10} "
          f"{'frac<={:g}'.format(args.tol_warn):>12} {'nan_1side':>9} {'nan_both':>9}")
    report: dict[str, dict] = {}
    failed: list[str] = []
    for fam in sorted(fam_cols):
        idx = fam_cols[fam]
        a = offline[:, idx]
        b = online[:, idx]
        nan_a = np.isnan(a)
        nan_b = np.isnan(b)
        both = nan_a & nan_b
        one = nan_a ^ nan_b
        fin = ~(nan_a | nan_b)
        d = np.abs(a[fin] - b[fin])
        max_d = float(d.max()) if d.size else 0.0
        if one.any():
            max_d = float("inf")
        n_cells = a.size
        within = int(both.sum()) + int((d <= args.tol_warn).sum())
        frac = within / n_cells if n_cells else 1.0
        report[fam] = {
            "n_cols": len(idx),
            "n_cells": int(n_cells),
            "max_abs_delta": max_d,
            "frac_within_tol_warn": frac,
            "n_nan_one_sided": int(one.sum()),
            "n_nan_both": int(both.sum()),
        }
        if max_d > args.tol_fail:
            failed.append(fam)
        print(f"{fam:<20} {len(idx):>4} {max_d:>10.3e} {frac:>12.6f} "
              f"{int(one.sum()):>9} {int(both.sum()):>9}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "engine": args.engine,
            "min_date": args.min_date,
            "max_date": args.max_date,
            "rolling_window": cfg.rolling_window,
            "n_rows": int(offline.shape[0]),
            "n_features": len(feature_cols),
            "tol_warn": args.tol_warn,
            "tol_fail": args.tol_fail,
            "families": report,
            "failed_families": failed,
        }, indent=2))
        print(f"→ {args.out}")

    if failed:
        print(f"\nFAIL: families over tol_fail={args.tol_fail:g}: {failed}",
              file=sys.stderr)
        return 1
    print(f"\nOK: all families within tol_fail={args.tol_fail:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
