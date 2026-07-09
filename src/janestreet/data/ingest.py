"""Lazy ingestion of the partitioned parquet dataset.

We never hold all 10 partitions in memory by default — callers either ask for
a date range or pass a ``LazyFrame`` straight into the feature builder, which
collects only what it needs.
"""

from __future__ import annotations

import polars as pl

from janestreet.config import COL_DATE, Cfg


def scan_train(cfg: Cfg) -> pl.LazyFrame:
    """Scan all partitions as a single lazy frame. No materialization yet."""
    lf = pl.scan_parquet(str(cfg.train_parquet / "**" / "*.parquet"))
    if "partition_id" in lf.collect_schema().names():
        lf = lf.drop("partition_id")
    return lf


def scan_train_dates(
    cfg: Cfg,
    min_date: int | None = None,
    max_date: int | None = None,
) -> pl.LazyFrame:
    """Lazy frame restricted to a date_id range (inclusive).

    Defaults pull from ``cfg.min_date_id`` / ``cfg.max_date_id``.
    """
    lo = cfg.min_date_id if min_date is None else min_date
    hi = cfg.max_date_id if max_date is None else max_date
    lf = scan_train(cfg)
    lf = lf.filter(pl.col(COL_DATE) >= lo)
    if hi is not None:
        lf = lf.filter(pl.col(COL_DATE) <= hi)
    return lf


def load_train(
    cfg: Cfg,
    min_date: int | None = None,
    max_date: int | None = None,
) -> pl.DataFrame:
    """Materialize a date-bounded slice. Use this only when you must collect."""
    return scan_train_dates(cfg, min_date=min_date, max_date=max_date).collect()


def date_id_range(cfg: Cfg) -> tuple[int, int]:
    """Min / max date_id across the full dataset (cheap stats scan)."""
    df = (
        scan_train(cfg)
        .select(
            pl.col(COL_DATE).min().alias("lo"),
            pl.col(COL_DATE).max().alias("hi"),
        )
        .collect()
    )
    return int(df["lo"][0]), int(df["hi"][0])
