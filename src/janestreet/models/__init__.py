"""Model registry — looks up models by name from the CLI / pipeline.

Every model implements :class:`janestreet.models.base.BaseModel`. To plug in a
new architecture, write the class, then add an entry to ``REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from janestreet.models.ae_mlp import AEMLPModel
from janestreet.models.base import BaseModel
from janestreet.models.gbm import XGBPerHorizon
from janestreet.models.itransformer import InvertedTransformerModel
from janestreet.models.mlp import MLPModel
from janestreet.models.mlp_sig import MLPWithSignatureModel
from janestreet.models.patchtst import PatchTSTModel
from janestreet.models.recurrent import RecurrentModel
from janestreet.models.timexer import TimeXerModel
from janestreet.models.transformer import SignatureTransformerModel, TransformerModel

REGISTRY: dict[str, Callable[..., BaseModel]] = {
    # Volkova replica — 4 branches of GRU + aux-head linear combiner.
    "gru_modelr": lambda **kw: RecurrentModel(model_type="gru", aux_branches=True, **kw),
    "lstm_modelr": lambda **kw: RecurrentModel(model_type="lstm", aux_branches=True, **kw),
    # Single-branch (no aux heads).
    "gru": lambda **kw: RecurrentModel(model_type="gru", aux_branches=False, **kw),
    "lstm": lambda **kw: RecurrentModel(model_type="lstm", aux_branches=False, **kw),
    # MLP baseline (no time recurrence, no signature).
    "mlp": lambda **kw: MLPModel(**kw),
    # MLP + signature — stateless in time; temporal info lives entirely in
    # the signature features. Orthogonal ensemble candidate vs RNN.
    "mlp_sig": lambda **kw: MLPWithSignatureModel(**kw),
    # Transformer — vanilla and signature-augmented variants.
    "transformer": lambda **kw: TransformerModel(**kw),
    "sig_transformer": lambda **kw: SignatureTransformerModel(**kw),
    # Gradient boosting per-horizon (non-DL baseline).
    "xgb": lambda **kw: XGBPerHorizon(**kw),
    # Causal iTransformer/TimeXer-inspired: inverted (cross-feature) attention
    # + endogenous global-token cross-attention + causal temporal mixing.
    "itransformer": lambda **kw: InvertedTransformerModel(**kw),
    # Causal PatchTST: attention over overlapping time patches (16/8) instead
    # of timesteps; per-timestep head reads the last *complete* patch, so no
    # within-patch future leaks. Channel independence dropped (see patchtst.py).
    "patchtst": lambda **kw: PatchTSTModel(**kw),
    # TimeXer-for-JS: lagged-responder endogenous patches + global token,
    # causal exogenous stream over today's features, cross-attention bridge.
    # Needs cfg.lagged_responders set + endo_channels passed.
    "timexer": lambda **kw: TimeXerModel(**kw),
    # ModelR + cross-sectional attention: at each timestep, symbols attend
    # to each other's hidden states — the one axis no other model mixes
    # (contemporaneous, hence deployable). EDA motivation: per-symbol R²
    # spans -0.003..+0.022; weak symbols should borrow strength.
    "gru_modelr_xsec": lambda **kw: RecurrentModel(
        model_type="gru", aux_branches=True, **{"xsec_heads": 4, **kw}),
    "lstm_modelr_xsec": lambda **kw: RecurrentModel(
        model_type="lstm", aux_branches=True, **{"xsec_heads": 4, **kw}),
    # Supervised denoising AE + MLP, end to end — the family that won the
    # 2021 Jane Street competition; row-wise, so structurally decorrelated
    # from the RNN streams.
    "ae_mlp": lambda **kw: AEMLPModel(**kw),
}


def build_model(name: str, **kwargs: Any) -> BaseModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "REGISTRY",
    "BaseModel",
    "InvertedTransformerModel",
    "MLPModel",
    "MLPWithSignatureModel",
    "PatchTSTModel",
    "RecurrentModel",
    "SignatureTransformerModel",
    "TimeXerModel",
    "TransformerModel",
    "XGBPerHorizon",
    "build_model",
]
