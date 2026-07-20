"""Top-level orchestration.

``FullPipeline`` wraps a single model + optional preprocessor and exposes
``fit`` / ``predict`` / ``update`` over polars frames. ``run_cv`` drives
expanding-window CV with the online-refit protocol Volkova used (and we
generalize to any ``BaseModel``).
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch  # noqa: F401  (kept so torch-using models import cleanly when serialized)
from tqdm.auto import tqdm

from janestreet.config import (
    COL_DATE,
    COL_ID,
    COL_TIME,
    COL_WEIGHT,
    Cfg,
)
from janestreet.data.features import FeatureBuilder
from janestreet.data.ingest import load_train
from janestreet.data.scaler import OnlineStandardizer, QuantileClipper
from janestreet.models import build_model
from janestreet.models.base import BaseModel, FitData
from janestreet.training.cv import TimeSeriesDateSplit
from janestreet.training.metrics import r2_weighted


@dataclass
class Preprocessor:
    """Fit-time clipper + standardizer. ``feature_cols`` is fixed at fit time."""

    feature_cols: list[str]
    clipper: QuantileClipper
    scaler: OnlineStandardizer

    @classmethod
    def fit(cls, df: pl.DataFrame, feature_cols: list[str]) -> Preprocessor:
        X = df.select(feature_cols).to_numpy()
        clipper = QuantileClipper().fit(X)
        X = clipper.transform(X)
        scaler = OnlineStandardizer().fit(X)
        return cls(feature_cols=feature_cols, clipper=clipper, scaler=scaler)

    def transform(self, df: pl.DataFrame, refit: bool = False) -> np.ndarray:
        X = df.select(self.feature_cols).to_numpy()
        X = self.clipper.transform(X)
        if refit:
            # Update running stats with the new block — keeps the scaler in
            # tune with the test distribution at inference time.
            self.scaler.partial_fit(X)
        return self.scaler.transform(X)


class FullPipeline:
    """Bundles a model + preprocessor and exposes polars-frame I/O."""

    def __init__(
        self,
        cfg: Cfg,
        model: BaseModel,
        feature_cols: list[str],
        aux_cols: list[str],
        target_col: str,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.feature_cols = feature_cols
        self.aux_cols = aux_cols
        self.target_col = target_col
        self.preprocessor = preprocessor

    # ------------------------------------------------------------------
    def _to_fitdata(self, df: pl.DataFrame, refit: bool) -> FitData:
        X = self.preprocessor.transform(df, refit=refit) if self.preprocessor else (
            df.select(self.feature_cols).to_numpy()
        )
        resp = (
            df.select(self.aux_cols).to_numpy()
            if self.aux_cols
            else np.zeros((df.height, 0), dtype=np.float32)
        )
        y = df.select(self.target_col).to_series().to_numpy().astype(np.float32)
        w = df.select(COL_WEIGHT).to_series().to_numpy().astype(np.float32)
        symbols = df.select(COL_ID).to_series().to_numpy()
        dates = df.select(COL_DATE).to_series().to_numpy()
        times = df.select(COL_TIME).to_series().to_numpy()
        return FitData(
            X=X.astype(np.float32),
            resp=resp.astype(np.float32),
            y=y,
            w=w,
            symbols=symbols,
            dates=dates,
            times=times,
        )

    def fit(
        self,
        df: pl.DataFrame,
        df_valid: pl.DataFrame | None = None,
        verbose: bool = False,
        warm_start: bool = False,
        epoch_save_dir: object = None,
        resume_from: object = None,
    ) -> None:
        """Fit the wrapped model.

        ``warm_start`` — keep the model's current weights (loaded via
        ``FullPipeline.load``) and continue training.

        ``epoch_save_dir`` / ``resume_from`` — mid-training checkpointing.
        See ``RecurrentModel.fit`` docstring for the semantics. Both
        are opt-in; only RNN family currently implements them (other
        backends ignore silently).
        """
        if self.preprocessor is None:
            self.preprocessor = Preprocessor.fit(df, self.feature_cols)
        train = self._to_fitdata(df, refit=False)
        valid = self._to_fitdata(df_valid, refit=False) if df_valid is not None else None
        try:
            self.model.fit(
                train, valid, verbose=verbose, warm_start=warm_start,
                epoch_save_dir=epoch_save_dir, resume_from=resume_from,
            )
        except TypeError:
            # Model's fit doesn't accept the newer kwargs — fall back.
            try:
                self.model.fit(train, valid, verbose=verbose, warm_start=warm_start)
            except TypeError:
                self.model.fit(train, valid, verbose=verbose)

    def predict(self, df: pl.DataFrame, refit: bool = False) -> np.ndarray:
        fit = self._to_fitdata(df, refit=refit)
        if self.model.sequence_model:
            n_times = df.select(COL_TIME).n_unique()
            preds, _ = self.model.predict(fit.X, n_times=n_times)
        else:
            preds, _ = self.model.predict(fit.X)
        return np.clip(preds, -5.0, 5.0)

    def update(self, df: pl.DataFrame) -> None:
        fit = self._to_fitdata(df, refit=True)
        n_times = df.select(COL_TIME).n_unique() if self.model.sequence_model else 1
        self.model.update(fit.X, fit.y, fit.w, n_times)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Pickle the pipeline (preprocessor + model + feature schema).

        Note on XGB: ``xgb.Booster`` cannot be round-tripped through pickle
        on macOS arm64 in xgboost 3.2 — torch and xgboost ship competing
        copies of libomp and pickling an XGBPerHorizon in a process that
        has also imported torch yields a segfault on ``pickle.load``. We
        skip XGB altogether: ``save`` raises ``NotImplementedError`` for it,
        and downstream scripts work off ``.npz`` prediction dumps instead.
        """
        if hasattr(self.model, "dump_booster"):  # XGBPerHorizon sentinel
            raise NotImplementedError(
                "XGBPerHorizon checkpoint disabled (libomp conflict with torch). "
                "Use the bench's .npz prediction dumps for downstream blending."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "feature_cols": self.feature_cols,
                    "aux_cols": self.aux_cols,
                    "target_col": self.target_col,
                    "preprocessor": self.preprocessor,
                    "model": self.model,
                    "cfg": self.cfg,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> FullPipeline:
        with Path(path).open("rb") as f:
            blob = pickle.load(f)  # noqa: S301  trusted local artifacts
        return cls(
            cfg=blob["cfg"],
            model=blob["model"],
            feature_cols=blob["feature_cols"],
            aux_cols=blob["aux_cols"],
            target_col=blob["target_col"],
            preprocessor=blob["preprocessor"],
        )


# ===========================================================================
def make_pipeline(
    cfg: Cfg,
    feature_cols: list[str] | None = None,
    aux_cols: list[str] | None = None,
) -> FullPipeline:
    """Build a fresh pipeline from a config (no fitting yet)."""
    fb = FeatureBuilder(cfg)
    feature_cols = feature_cols if feature_cols is not None else fb.feature_columns()
    aux_cols = aux_cols if aux_cols is not None else cfg.aux_targets
    model = build_model(cfg.model_name, **cfg.model_kwargs)
    return FullPipeline(
        cfg=cfg,
        model=model,
        feature_cols=feature_cols,
        aux_cols=aux_cols,
        target_col=cfg.target,
    )


# ===========================================================================
def prepare_dataset(
    cfg: Cfg, downcast: bool = True, storage_precision: str = "float32",
) -> pl.DataFrame:
    """Load the date-bounded slice and apply Volkova's feature engineering.

    Parameters
    ----------
    downcast
        Cast Float64 → Float32 at the end (default True). Halves the frame
        footprint of Volkova-style column ops.
    storage_precision
        ``"float32"`` (default) or ``"float16"``. Applied *after* all feature
        engineering (rolling stats, market averages) so the numerically
        sensitive computations still run in Float32. Float16 halves the frame
        again (Float32 → 2 B per element). Roundtrip error to Float32 at
        model-input time is ~1e-4 for standardised features — well below the
        noise floor — so it's safe as a pure storage optimization.

        Memory footprint on the 500-train + 100-valid slice (~22 M rows,
        ~150 float cols):

        * Float32: ~13 GB polars frame
        * Float16: ~6.5 GB polars frame  ← what to use on Colab free tier

        (Downstream ``_to_fitdata`` still materialises the numpy X in
        Float32 for the model, so peak-during-fit doesn't halve — but the
        *persistent* frame does. See notebooks/colab_train_500d.ipynb.)
    """
    if storage_precision not in ("float32", "float16"):
        raise ValueError(f"storage_precision must be float32 or float16, got {storage_precision!r}")
    df = load_train(cfg)
    fb = FeatureBuilder(cfg)
    df = fb.build_train(df)
    if downcast:
        df = df.with_columns(
            [pl.col(c).cast(pl.Float32) for c, dt in df.schema.items() if dt == pl.Float64]
        )
    if storage_precision == "float16":
        df = df.with_columns(
            [pl.col(c).cast(pl.Float16) for c, dt in df.schema.items() if dt == pl.Float32]
        )
    return df


# ===========================================================================
def run_cv(cfg: Cfg, df: pl.DataFrame | None = None) -> list[float]:
    """Run expanding-window CV; return per-fold weighted-R²."""
    df = df if df is not None else prepare_dataset(cfg)
    dates = df.select(pl.col(COL_DATE).unique().sort()).to_series().to_numpy()

    splitter = TimeSeriesDateSplit(
        n_splits=cfg.n_splits, test_size=cfg.test_size_dates, gap=cfg.cv_gap_dates,
    )

    scores: list[float] = []
    for fold, (train_dates, valid_dates) in enumerate(splitter.split(dates)):
        if cfg.verbose:
            print("=" * 70)
            print(
                f"Fold {fold}: train {train_dates[0]}->{train_dates[-1]} "
                f"({len(train_dates)} dates) | valid {valid_dates[0]}->{valid_dates[-1]} "
                f"({len(valid_dates)} dates)"
            )

        df_train = df.filter(pl.col(COL_DATE).is_in(train_dates))
        df_valid = df.filter(pl.col(COL_DATE).is_in(valid_dates))

        pipe = make_pipeline(cfg)
        pipe.fit(df_train, df_valid, verbose=cfg.verbose)

        # Day-by-day predict + online refit (matches Volkova's loop).
        # ``pipe`` is not reused after this point, so the deepcopy is only
        # defensive — and XGBPerHorizon cannot survive it: its __getstate__
        # nulls the booster (deliberately, to dodge the torch+libomp pickle
        # segfault on macOS), so deepcopy *succeeds* and silently returns a
        # model that asserts on predict. Detect via the same dump_booster
        # sentinel FullPipeline.save uses and reuse the pipeline directly.
        if hasattr(pipe.model, "dump_booster"):
            pipe_eval = pipe
        else:
            try:
                pipe_eval = copy.deepcopy(pipe)
            except Exception:
                pipe_eval = pipe
        preds_list: list[np.ndarray] = []
        for i, dt in enumerate(tqdm(valid_dates, disable=not cfg.verbose, desc=f"fold{fold}-eval")):
            day = df_valid.filter(pl.col(COL_DATE) == dt)
            if i > 0:
                prev = df.filter(pl.col(COL_DATE) == dt - 1)
                if prev.height > 0:
                    pipe_eval.update(prev)
            preds_list.append(pipe_eval.predict(day))
        preds = np.concatenate(preds_list)

        y = df_valid.fill_null(0.0).select(cfg.target).to_series().to_numpy()
        w = df_valid.select(COL_WEIGHT).to_series().to_numpy()
        score = r2_weighted(y, preds, w)
        scores.append(score)
        if cfg.verbose:
            print(f"Fold {fold} weighted-R² = {score:.5f}")

    if cfg.verbose:
        print(f"\nCV mean R² = {np.mean(scores):.5f} (folds: {scores})")
    return scores
