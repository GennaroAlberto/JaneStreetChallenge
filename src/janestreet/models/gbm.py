"""XGBoost adapter — non-DL baseline.

Trains one model on the flat (N, K) feature matrix. Predicts and updates do
not need ``n_times`` since each row is independent under this model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xgboost as xgb

from janestreet.models.base import BaseModel, FitData


class XGBPerHorizon(BaseModel):
    sequence_model = False

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.5,
        min_child_weight: float = 10,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        tree_method: str = "hist",
        max_bin: int = 256,
        # macOS' bundled libomp deadlocks XGBoost when n_jobs > 1 in some
        # environments. Default to 1 — set higher explicitly when you know
        # your stack is OK with it (e.g. Linux + libomp from system pkg).
        n_jobs: int = 1,
        seed: int = 42,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            max_bin=max_bin,
            n_jobs=n_jobs,
            random_state=seed,
            objective="reg:squarederror",
        )
        self.early_stopping_rounds = early_stopping_rounds
        # `model` is the sklearn wrapper (used during fit; nice eval-set API).
        # `booster` is the underlying ``xgb.Booster`` — the source of truth
        # for predict and the only thing we serialize, since the sklearn
        # wrapper's pickle path segfaults on macOS arm64 in xgboost 3.2.
        self.model: xgb.XGBRegressor | None = None
        self.booster: xgb.Booster | None = None
        # Holds the raw booster bytes when the object came from unpickle —
        # we lazy-load on first predict because calling Booster.load_model
        # *during* ``__setstate__`` segfaults silently on macOS arm64.
        self._pending_booster_bytes: bytes | None = None
        self.n_features_in_: int | None = None

    def fit(self, train: FitData, valid: FitData | None = None, verbose: bool = False) -> None:
        eval_set = None
        sample_weight_eval_set = None
        if valid is not None:
            eval_set = [(valid.X, valid.y)]
            sample_weight_eval_set = [valid.w]
        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )
        self.model.fit(
            train.X, train.y,
            sample_weight=train.w,
            eval_set=eval_set,
            sample_weight_eval_set=sample_weight_eval_set,
            verbose=verbose,
        )
        # Snapshot the booster — this is what we use to predict + serialize.
        self.booster = self.model.get_booster()
        self.n_features_in_ = int(train.X.shape[1])

    def _ensure_booster(self) -> None:
        """Lazy-load the booster from cached bytes (deferred from __setstate__)."""
        if self.booster is None and self._pending_booster_bytes is not None:
            # Persist bytes to a temp file then load via file path. On macOS
            # arm64 + xgboost 3.2, calling ``load_model(bytearray)`` after the
            # XGBPerHorizon object has been through pickle + any subsequent
            # numpy work in the process segfaults silently. The file-path
            # variant goes through a separate xgb code path that survives.
            with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tf:
                tf.write(self._pending_booster_bytes)
                tmp = tf.name
            try:
                b = xgb.Booster()
                b.load_model(tmp)
                self.booster = b
                self._pending_booster_bytes = None
            finally:
                Path(tmp).unlink(missing_ok=True)

    def predict(
        self, X: np.ndarray, n_times: int | None = None, state: object | None = None
    ) -> tuple[np.ndarray, object | None]:
        self._ensure_booster()
        assert self.booster is not None
        # ``inplace_predict`` skips the DMatrix construction and is much faster
        # on numpy input; works identically to sklearn-wrapper predict for
        # regression objectives.
        return self.booster.inplace_predict(X), None

    # ------------------------------------------------------------------
    # Custom pickle: drop the heavy XGB state entirely. ``xgb.Booster``
    # cannot be reliably round-tripped through pickle on macOS arm64 +
    # xgboost 3.2 (silent C-level crash; reproducible only when other
    # numpy work has run in the process between unpickle and predict).
    # We instead persist XGB models via a side channel (see ``dump_xgb_*``
    # helpers); FullPipeline.save retains the params so the model is
    # rebuildable, but does not try to ship the booster through pickle.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["model"] = None
        state["booster"] = None
        state["_pending_booster_bytes"] = None
        return state

    # ------------------------------------------------------------------
    # Side-channel serialization (use these from FullPipeline.save / load).
    def dump_booster(self, path) -> None:
        """Write the trained booster to a side file (UBJSON)."""
        if self.booster is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    def load_booster(self, path) -> None:
        """Restore the booster from a side file."""
        b = xgb.Booster()
        b.load_model(str(path))
        self.booster = b
        self._pending_booster_bytes = None
