"""Feature engineering — Volkova baseline + extensions.

We reproduce her three engineered groups (rolling per-symbol mean / std on 16
high-correlation features, market average per (date, time), and the raw
``time_id`` as a feature) and synthesize her two auxiliary responders. We
also expose hooks for additional feature groups so a Transformer / signature
model can stack extras without disturbing the baseline.
"""

from __future__ import annotations

import polars as pl

from janestreet.config import (
    COL_DATE,
    COL_ID,
    COL_TIME,
    COLS_FEATURES_CAT,
    COLS_FEATURES_CORR,
    COLS_FEATURES_INIT,
    Cfg,
)


class FeatureBuilder:
    """Build the Volkova feature set.

    Parameters
    ----------
    cfg:
        Project config.
    extra_groups:
        Optional list of names from ``{"rank_per_time", "ewma_per_symbol"}``
        to opportunistically add extra signals. They are zero-overhead when
        the lazy frame is filtered, so default to off.
    """

    def __init__(self, cfg: Cfg, extra_groups: list[str] | None = None) -> None:
        self.cfg = cfg
        self.extra_groups = extra_groups or []

    # ---- public API ------------------------------------------------------
    def feature_columns(self) -> list[str]:
        """The full list of model-input columns AFTER feature engineering."""
        t = self.cfg.rolling_window
        cols = list(COLS_FEATURES_INIT)
        cols += [f"{c}_diff_rolling_avg_{t}" for c in COLS_FEATURES_CORR]
        cols += [f"{c}_rolling_std_{t}" for c in COLS_FEATURES_CORR]
        cols += [f"{c}_avg_per_date_time" for c in COLS_FEATURES_CORR]
        cols += ["feature_time_id"]
        if "rank_per_time" in self.extra_groups:
            cols += [f"{c}_rank_per_time" for c in COLS_FEATURES_CORR]
        if "ewma_per_symbol" in self.extra_groups:
            cols += [f"{c}_ewma_30" for c in COLS_FEATURES_CORR]
        # Lagged (previous-day) responders — appended last so their indices
        # are stable and easy to hand to the TimeXer endogenous stream.
        cols += [f"responder_{r}_lag1d" for r in self.cfg.lagged_responders]
        if self.cfg.responder_signal_features:
            cols += self._responder_signal_names()
        cols = [c for c in cols if c not in COLS_FEATURES_CAT]
        return cols

    @staticmethod
    def _responder_signal_names() -> list[str]:
        return [
            "rsig_momA_short", "rsig_momA_med",   # exchange A: SMA4-SMA20, SMA20-SMA120
            "rsig_momB_short", "rsig_momB_med",   # exchange B: SMA4-SMA20, SMA20-SMA120
            "rsig_spread_fine",                    # A-B at the finest (SMA4) scale
        ]

    def lagged_responder_columns(self) -> list[str]:
        """Just the lagged-responder feature names (TimeXer endogenous set)."""
        return [f"responder_{r}_lag1d" for r in self.cfg.lagged_responders]

    # ------------------------------------------------------------------
    def build_train(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the full feature pipeline to a materialized training frame.

        Adds:
          - responder_9 = responder_8 + responder_8.shift(-4) per symbol
          - responder_10 = responder_6 + shift(-20) + shift(-40) per symbol
          - rolling mean / std on the 16-feature corr subset (T = 1000)
          - market average per (date, time)
          - ``feature_time_id`` mirror of ``time_id``
          - optional extras
        """
        df = self._add_synthetic_responders(df)
        if self.cfg.realized_aux:
            df = self._add_realized_responders(df)
        df = self._add_rolling_per_symbol(df)
        df = self._add_market_avg(df)
        df = df.with_columns(pl.col(COL_TIME).cast(pl.Float32).alias("feature_time_id"))
        if "rank_per_time" in self.extra_groups:
            df = self._add_rank_per_time(df)
        if "ewma_per_symbol" in self.extra_groups:
            df = self._add_ewma_per_symbol(df)
        if self.cfg.lagged_responders:
            df = self._add_lagged_responders(df)
        if self.cfg.responder_signal_features:
            df = self._add_responder_signal_features(df)
        return df

    # ---- internals -------------------------------------------------------
    @staticmethod
    def _add_synthetic_responders(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (
                pl.col("responder_8")
                + pl.col("responder_8").shift(-4).over(COL_ID)
            ).fill_null(0.0).alias("responder_9"),
            (
                pl.col("responder_6")
                + pl.col("responder_6").shift(-20).over(COL_ID)
                + pl.col("responder_6").shift(-40).over(COL_ID)
            ).fill_null(0.0).alias("responder_10"),
        )

    @staticmethod
    def _add_realized_responders(df: pl.DataFrame) -> pl.DataFrame:
        """Backward-realized responders as aux TARGETS (nowcast heads).

        The dataset's responder_r at row t is the forward SMA over ~(t..t+w)
        (#555562). Its value at row t-w is therefore the SMA that *completed*
        at t — a realized quantity the features describe with high R². These
        are training targets only (never model inputs), so like the forward
        synthetic responders they may be null-filled at day edges; the same
        1-date embargo logic applies (backward shifts reach into the previous
        date's tail, which is training-side, not validation-side).
        """
        return df.with_columns(
            pl.col("responder_6").shift(20).over(COL_ID)
            .fill_null(0.0).alias("responder_11"),
            pl.col("responder_8").shift(4).over(COL_ID)
            .fill_null(0.0).alias("responder_12"),
        )

    def _add_rolling_per_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
        t = self.cfg.rolling_window
        mean_exprs = [
            pl.col(c).rolling_mean(window_size=t).over(COL_ID).alias(f"{c}_rolling_avg_{t}")
            for c in COLS_FEATURES_CORR
        ]
        std_exprs = [
            pl.col(c).rolling_std(window_size=t).over(COL_ID).alias(f"{c}_rolling_std_{t}")
            for c in COLS_FEATURES_CORR
        ]
        df = df.with_columns(mean_exprs + std_exprs)
        diff_exprs = [
            (pl.col(c) - pl.col(f"{c}_rolling_avg_{t}")).alias(f"{c}_diff_rolling_avg_{t}")
            for c in COLS_FEATURES_CORR
        ]
        df = df.with_columns(diff_exprs)
        df = df.drop([f"{c}_rolling_avg_{t}" for c in COLS_FEATURES_CORR])
        return df

    @staticmethod
    def _add_market_avg(df: pl.DataFrame) -> pl.DataFrame:
        exprs = [
            pl.col(c).mean().over([COL_DATE, COL_TIME]).alias(f"{c}_avg_per_date_time")
            for c in COLS_FEATURES_CORR
        ]
        return df.with_columns(exprs)

    def _add_lagged_responders(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add previous-day responders as input features (backward-looking).

        For each configured responder r, ``responder_{r}_lag1d`` at
        ``(symbol, date d, time t)`` equals ``responder_r`` at
        ``(symbol, date d-1, time t)``. Implemented as a self-join on
        ``(symbol, date+1, time)`` — order-independent and unambiguous:
        we take each responder value, advance its date by +1, and join it
        back so it lands on the *following* day. This is exactly the
        lags.parquet signal the competition provides at inference, so it is
        deployable and cannot leak the future (it only reaches backward).

        The earliest date in the slice gets nulls (no prior day) → filled 0.
        """
        lag_cols = [f"responder_{r}" for r in self.cfg.lagged_responders]
        lagged = (
            df.select([COL_ID, COL_DATE, COL_TIME, *lag_cols])
            .with_columns((pl.col(COL_DATE) + 1).alias(COL_DATE))
            .rename({c: f"{c}_lag1d" for c in lag_cols})
        )
        df = df.join(lagged, on=[COL_ID, COL_DATE, COL_TIME], how="left")
        return df.with_columns(
            [pl.col(f"{c}_lag1d").fill_null(0.0) for c in lag_cols]
        )

    def _add_responder_signal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Multi-scale momentum + venue spread from the lagged responders.

        Window/exchange mapping (from Kaggle #555562, confirmed by
        responders.csv tags): within a group the responders are SMA-20/120/4
        of a common signal —
            exchange A (target): r6=SMA20, r7=SMA120, r8=SMA4
            exchange B         : r3=SMA20, r4=SMA120, r5=SMA4
        Cross-window differences of yesterday's values are the short/medium
        trend of the underlying; the finest-scale cross-exchange difference is
        the venue spread. Every term is a difference of already-lagged
        (previous-day) columns, so nothing reaches into today or the future.
        """
        need = {3, 4, 5, 6, 7, 8}
        if not need.issubset(set(self.cfg.lagged_responders)):
            raise ValueError(
                "responder_signal_features needs lagged_responders to include "
                f"{sorted(need)}; got {self.cfg.lagged_responders}"
            )

        def lag(r: int) -> pl.Expr:
            return pl.col(f"responder_{r}_lag1d")

        return df.with_columns(
            (lag(8) - lag(6)).alias("rsig_momA_short"),   # SMA4 - SMA20, exch A
            (lag(6) - lag(7)).alias("rsig_momA_med"),     # SMA20 - SMA120, exch A
            (lag(5) - lag(3)).alias("rsig_momB_short"),   # SMA4 - SMA20, exch B
            (lag(3) - lag(4)).alias("rsig_momB_med"),     # SMA20 - SMA120, exch B
            (lag(8) - lag(5)).alias("rsig_spread_fine"),  # A - B at SMA4 (finest)
        )

    @staticmethod
    def _add_rank_per_time(df: pl.DataFrame) -> pl.DataFrame:
        """Cross-sectional rank of each feature per (date, time) — strong signal."""
        exprs = [
            pl.col(c)
            .rank(method="average")
            .over([COL_DATE, COL_TIME])
            .cast(pl.Float32)
            .alias(f"{c}_rank_per_time")
            for c in COLS_FEATURES_CORR
        ]
        return df.with_columns(exprs)

    @staticmethod
    def _add_ewma_per_symbol(df: pl.DataFrame) -> pl.DataFrame:
        exprs = [
            pl.col(c)
            .ewm_mean(span=30, adjust=False)
            .over(COL_ID)
            .alias(f"{c}_ewma_30")
            for c in COLS_FEATURES_CORR
        ]
        return df.with_columns(exprs)
