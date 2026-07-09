"""Causal inverted Transformer — iTransformer + TimeXer ideas, made deployable.

Design provenance
-----------------

* **iTransformer** (Liu et al., 2023, arXiv:2310.06625): *invert* the token
  axis — self-attention runs over **variates (features)** rather than time,
  so it directly models cross-feature correlations; a per-token FFN handles
  the temporal representation.

* **TimeXer** (Wang et al., 2024, arXiv:2402.19072): separate **endogenous**
  (the target's own signal) from **exogenous** covariates; a learnable
  **global endogenous token** cross-attends to the exogenous variate tokens
  to import their information into the target representation.

The one adaptation both papers force on us
-------------------------------------------

Jane Street scores per ``time_id`` causally — the prediction at time t may
use only data at times ≤ t (the gateway streams one time_id at a time). The
vanilla iTransformer/TimeXer embed a *whole* window per variate token, which
is non-causal and would leak future-within-day. We therefore keep the ideas
but make them causal:

1. **Inverted feature attention is applied per timestep.** At each time t we
   treat a configurable subset of features as variate tokens and self-attend
   across them. Attention within a single timestep can't see the future, so
   this is causal by construction. (This is the iTransformer inversion.)
2. **A global endogenous token** (learnable query) cross-attends over those
   per-timestep feature tokens — the TimeXer bridge — producing one
   cross-feature summary vector per timestep.
3. **A causal temporal Transformer** (triangular mask) mixes those summaries
   over time — this is the only place time is mixed, and it is strictly
   causal, so the model is deployable one time_id at a time.

Cost note: the inverted attention is over ``len(variate_channels)`` tokens
(default the AR-selected subset, ~8), not all 125 features, so it stays cheap
on CPU. The full feature vector still feeds the temporal backbone via a
standard input projection.
"""

from __future__ import annotations

import torch
from torch import nn

from janestreet.models.transformer import _BaseTransformerModel
from janestreet.theory.signatures import SignatureBlock


class _InvertedFeatureBlock(nn.Module):
    """Per-timestep self-attention across a set of feature (variate) tokens.

    Input  : (B, T, V, D) — V variate tokens per timestep, D-dim each.
    Output : (B, T, V, D) — same, after attention over the V axis + FFN.

    Attention is computed independently at each (B, T) position, over the V
    tokens, so no information crosses the time axis here (causal-safe).
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, v, d = x.shape
        h = x.reshape(b * t, v, d)          # attention over V, per (B,T) position
        a, _ = self.attn(self.norm1(h), self.norm1(h), self.norm1(h), need_weights=False)
        h = h + a
        h = h + self.ff(self.norm2(h))
        return h.reshape(b, t, v, d)


class _CausalTemporalBlock(nn.Module):
    """Standard pre-norm causal self-attention over the time axis."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False, is_causal=True)
        x = x + a
        return x + self.ff(self.norm2(x))


class CausalInvertedTransformer(nn.Module):
    """iTransformer/TimeXer-inspired causal encoder for per-timestep regression.

    Flow (input ``x``: (B, T, K)):
      1. Exogenous variate tokens: pick ``variate_idx`` features; embed each
         scalar to D via a per-variate affine map → (B, T, V, D).
      2. ``n_inv_layers`` of inverted feature attention over the V tokens.
      3. Global endogenous token cross-attends over the V tokens per timestep
         → (B, T, D) cross-feature summary (TimeXer bridge).
      4. Fuse with a projection of the *full* feature vector → temporal stream.
      5. ``n_temporal_layers`` of causal temporal attention over T.
      6. Linear head → (B, T).
    """

    def __init__(
        self,
        input_size: int,
        variate_idx: list[int],
        d_model: int = 96,
        n_heads: int = 4,
        n_inv_layers: int = 2,
        n_temporal_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.variate_idx = variate_idx
        v = len(variate_idx)

        # (1) per-variate scalar embedding: value * w_v + b_v, (V,D) params each.
        self.var_w = nn.Parameter(torch.randn(v, d_model) * 0.02)
        self.var_b = nn.Parameter(torch.zeros(v, d_model))
        self.var_id = nn.Parameter(torch.randn(v, d_model) * 0.02)  # variate identity

        # (2) inverted feature attention
        self.inv_blocks = nn.ModuleList(
            _InvertedFeatureBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_inv_layers)
        )

        # (3) global endogenous query + cross-attention over variate tokens
        self.endo_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # (4) full-feature projection → temporal stream, fused with the summary
        self.in_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.fuse_norm = nn.LayerNorm(d_model)

        # (5) causal temporal attention
        self.temporal_blocks = nn.ModuleList(
            _CausalTemporalBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_temporal_layers)
        )

        # (6) head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        # (1) exogenous variate tokens from the selected features
        xv = x[..., self.variate_idx]                       # (B, T, V)
        tokens = xv.unsqueeze(-1) * self.var_w + self.var_b + self.var_id  # (B,T,V,D)

        # (2) inverted attention across the V feature tokens (per timestep)
        for blk in self.inv_blocks:
            tokens = blk(tokens)

        # (3) global endogenous token cross-attends over the variate tokens
        d = tokens.shape[-1]
        q = self.endo_query.expand(b * t, 1, d)             # (B*T, 1, D)
        kv = tokens.reshape(b * t, len(self.variate_idx), d)
        summary, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B*T,1,D)
        summary = self.cross_norm(summary).reshape(b, t, d)         # (B, T, D)

        # (4) fuse cross-feature summary with the full-feature temporal stream
        h = self.in_proj(x) + self.pos[:, :t, :] + summary
        h = self.fuse_norm(h)

        # (5) causal temporal mixing
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        for blk in self.temporal_blocks:
            h = blk(h, mask)

        # (6) head
        return self.head(self.head_norm(h)).squeeze(-1)


class InvertedTransformerModel(_BaseTransformerModel):
    """Causal iTransformer/TimeXer-inspired model.

    Reuses ``_BaseTransformerModel``'s per-date causal training / predict /
    online-refit loop; only the backbone differs. ``variate_channels`` picks
    which features become the inverted (exogenous) variate tokens — default is
    the AR-selected subset (see scripts/select_signature_channels_ar.py).
    Signature augmentation is off by default here (the inversion is the point).
    """

    # AR-selected feature indices (feature_16,66,45,36,73,25,23,59) — the same
    # subset that worked best for the signature models.
    _DEFAULT_VARIATES = [13, 63, 42, 33, 70, 22, 20, 56]

    def __init__(self, **kw) -> None:
        self.variate_channels = kw.pop("variate_channels", None) or self._DEFAULT_VARIATES
        self.n_inv_layers = kw.pop("n_inv_layers", 2)
        self.n_temporal_layers = kw.pop("n_temporal_layers", 2)
        kw.setdefault("signature_channels", None)  # no signature by default
        super().__init__(**kw)

    def _build(self, input_size: int) -> nn.Module:
        # Honour an optional signature pre-block (adds sig dims to input_size),
        # exactly like the base class, so the two ideas can be combined.
        if self.signature_channels:
            self.sig_block = SignatureBlock(
                channels=self.signature_channels, window=self.signature_window,
                depth=self.signature_depth, hurst=self.signature_hurst,
                mode=self.signature_mode,
            ).to(self.device)
            input_size = input_size + self.sig_block.sig_dim
        self.input_size = input_size
        # Guard: variate indices must be within the (possibly augmented) input.
        bad = [i for i in self.variate_channels if i >= input_size]
        if bad:
            raise ValueError(f"variate_channels out of range for input_size={input_size}: {bad}")
        return CausalInvertedTransformer(
            input_size=input_size,
            variate_idx=self.variate_channels,
            d_model=self.d_model, n_heads=self.n_heads,
            n_inv_layers=self.n_inv_layers,
            n_temporal_layers=self.n_temporal_layers,
            ff_mult=self.ff_mult, dropout=self.dropout, max_len=self.max_len,
        ).to(self.device)
