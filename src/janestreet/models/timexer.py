"""TimeXer-for-Jane-Street — faithful endogenous/exogenous split, made causal.

Provenance: TimeXer (Wang et al., 2024, arXiv:2402.19072). Its idea: separate
the **endogenous** target series (patchified, with a learnable global token)
from **exogenous** covariates (one inverted token per variate), and let a
global-token **cross-attention** bridge import covariate info into the target
representation.

Jane Street mapping
-------------------

* **Endogenous** = the target's own recent history. Within a trading day you
  do NOT observe responder_6's past, but the competition hands you the
  **previous day's** responder path (``responder_6_lag1d`` etc., via
  lags.parquet). That entire path is known at day-start, so it can be
  patchified and self-attended **non-causally** — it is all past data.
* **Exogenous** = today's features (feature_00…, rolling, market-avg). These
  stream in during the day, so anything touching them must stay **causal**
  (prediction at time t sees today's features only up to t).

Causality-forced adaptation of the bridge
------------------------------------------

TimeXer's global endo token *queries* the exo tokens. But turning today's
features into iTransformer-style whole-series variate tokens would embed the
future within the day → leakage. Since here the endogenous stream is the
fully-known one and the exogenous stream is the causal one, we **invert the
bridge**: the causal today-stream (query) cross-attends to the fully-known
yesterday-endogenous tokens (key/value). This imports yesterday's target
dynamics into today's causal prediction without ever seeing today's future.

Flow (input x: (B, T, K); endo_channels index the lagged responders):
  1. Endogenous: patchify the lagged-responder series → patch tokens +
     learnable global token; non-causal self-attention (all yesterday).
  2. Exogenous: causal temporal Transformer over the full feature vector.
  3. Bridge: causal per-t exo representation cross-attends to the endo
     tokens (patches + global).
  4. Head → responder_6 per timestep.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from janestreet.models.transformer import _BaseTransformerModel


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention block. ``causal`` toggles the triangular mask."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        h = self.norm1(x)
        mask = None
        if causal:
            t = x.shape[1]
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=causal)
        x = x + a
        return x + self.ff(self.norm2(x))


class TimeXerJSNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        endo_channels: list[int],
        patch_len: int = 44,
        d_model: int = 96,
        n_heads: int = 4,
        n_endo_layers: int = 2,
        n_exo_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.endo_channels = endo_channels
        self.patch_len = patch_len
        e = len(endo_channels)

        # (1) endogenous: patch embedding + learnable global token
        self.patch_embed = nn.Linear(patch_len * e, d_model)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.patch_pos = nn.Parameter(torch.zeros(1, max_len // patch_len + 2, d_model))
        self.endo_blocks = nn.ModuleList(
            _SelfAttnBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_endo_layers)
        )

        # (2) exogenous: causal temporal encoder over the full feature vector
        self.in_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.exo_blocks = nn.ModuleList(
            _SelfAttnBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_exo_layers)
        )

        # (3) bridge: causal exo stream (query) cross-attends to endo tokens
        self.cross_norm_q = nn.LayerNorm(d_model)
        self.cross_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # (4) head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _encode_endo(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        endo = x[..., self.endo_channels]                        # (B, T, E)
        # pad time up to a multiple of patch_len, then fold into patches
        pad = (-t) % self.patch_len
        if pad:
            endo = F.pad(endo, (0, 0, 0, pad))                   # pad the time axis
        p = endo.shape[1] // self.patch_len
        patches = endo.reshape(b, p, self.patch_len * endo.shape[-1])  # (B, P, L*E)
        tok = self.patch_embed(patches)                          # (B, P, D)
        glb = self.glb_token.expand(b, 1, tok.shape[-1])
        tok = torch.cat([tok, glb], dim=1) + self.patch_pos[:, : p + 1, :]
        for blk in self.endo_blocks:                             # non-causal: all yesterday
            tok = blk(tok, causal=False)
        return tok                                               # (B, P+1, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        endo_tokens = self._encode_endo(x)                       # (B, P+1, D)

        # exogenous causal temporal stream
        h = self.in_proj(x) + self.pos[:, :t, :]
        for blk in self.exo_blocks:
            h = blk(h, causal=True)

        # bridge: today's causal reps (query) attend to yesterday's endo tokens.
        # No mask needed on the cross-attention — every endo token is prior-day
        # data, fully available at every time t.
        q = self.cross_norm_q(h)
        kv = self.cross_norm_kv(endo_tokens)
        a, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h = h + a

        return self.head(self.head_norm(h)).squeeze(-1)


class TimeXerModel(_BaseTransformerModel):
    """TimeXer-for-JS: lagged-responder endogenous stream + causal exogenous.

    Requires the dataset to include lagged responders (set
    ``cfg.lagged_responders``); pass their feature indices as
    ``endo_channels``. Reuses the base transformer train/predict/refit loop.
    """

    def __init__(self, **kw) -> None:
        self.endo_channels = kw.pop("endo_channels", None)
        self.patch_len = kw.pop("patch_len", 44)
        self.n_endo_layers = kw.pop("n_endo_layers", 2)
        self.n_exo_layers = kw.pop("n_exo_layers", 2)
        kw.setdefault("signature_channels", None)
        super().__init__(**kw)
        if not self.endo_channels:
            raise ValueError(
                "TimeXerModel needs endo_channels (indices of the lagged-responder "
                "features). Set cfg.lagged_responders and pass their indices."
            )

    def _build(self, input_size: int) -> nn.Module:
        self.input_size = input_size
        bad = [i for i in self.endo_channels if i >= input_size]
        if bad:
            raise ValueError(f"endo_channels out of range for input_size={input_size}: {bad}")
        return TimeXerJSNet(
            input_size=input_size, endo_channels=self.endo_channels,
            patch_len=self.patch_len, d_model=self.d_model, n_heads=self.n_heads,
            n_endo_layers=self.n_endo_layers, n_exo_layers=self.n_exo_layers,
            ff_mult=self.ff_mult, dropout=self.dropout, max_len=self.max_len,
        ).to(self.device)
