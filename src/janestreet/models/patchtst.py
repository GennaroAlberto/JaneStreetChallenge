"""Causal PatchTST for per-timestep day-sequence regression.

Provenance: PatchTST (Nie et al., 2023, arXiv:2211.14730). Its two ideas:
(1) **patching** — embed short sub-series (patch_len steps) as tokens so
attention runs over ~T/stride patch tokens instead of T timesteps, giving a
longer effective receptive field per FLOP; (2) **channel independence** — one
shared univariate Transformer applied to each input channel separately.

Two honest deviations from the paper (and why)
----------------------------------------------

1. **Channel independence is dropped.** It is designed for forecasting many
   *homogeneous* series with one shared model. Here the ~134 input channels
   are heterogeneous *descriptors of one target* (features + lagged
   responders), not 134 parallel targets, and running each through its own
   token stream would mean attention over 134 x ~121 ≈ 16k tokens per
   symbol-day — madness on CPU for zero inductive-bias payoff. Instead we
   linear-project the K features of each timestep to d_model (exactly the
   house ``CausalTransformer`` front-end), so **patching is the one PatchTST
   mechanism under test** in the synthetic-world bench.

2. **Per-timestep causal outputs, not per-window forecasts.** Standard
   PatchTST reads a whole lookback window and emits one forecast at its end —
   offline and non-causal for our setting, where the prediction at time t may
   use only data at times ≤ t (the gateway streams one time_id at a time; see
   itransformer.py for the same constraint). Emitting at patch boundaries and
   forward-filling inside a patch would leak the current patch's future steps
   into earlier timesteps. Solution, airtight by construction:

   * patch tokens embed the *projected* stream over [i*stride, i*stride+L);
     a triangular mask over patch tokens means patch i attends only to
     patches ≤ i, i.e. only to timesteps ≤ its own end index;
   * the per-timestep head consumes ``[h_t, ctx_t]`` where ``h_t`` is the
     (pointwise, hence causal) projection of x_t plus a time-of-day embedding
     and ``ctx_t`` is the encoder output of the **last patch that ends at or
     before t**. Timesteps before the first complete patch get a learnable
     start token instead. No path from any timestep > t reaches output t.

Memory arithmetic (16 GB Mac; real data D_symbols≈39, T=968, K≈134, D=64):
input day tensor 39*968*134*4 B ≈ 20 MB; projected stream 39*968*64*4 B
≈ 9.7 MB; patch tokens 39*121*64*4 B ≈ 1.2 MB; attention weights
4 heads * 39 * 122² * 4 B ≈ 9 MB — patch attention is O(P²)≈15k entries vs
O(T²)≈937k for the vanilla causal transformer, ~60x cheaper. Everything is
tens of MB; peak RAM is dominated by the dataset, not this model.
"""

from __future__ import annotations

import torch
from torch import nn

from janestreet.models.transformer import _BaseTransformerModel


class CausalPatchTSTNet(nn.Module):
    """Patch-token causal encoder with a per-timestep two-stream head.

    Flow (input ``x``: (B, T, K), output (B, T)):
      1. Pointwise stream: ``h = in_proj(x) + pos`` — depends only on x_t and
         the clock, so it is causal by construction (and carries time-of-day,
         which patch tokens alone would blur).
      2. Patching: unfold ``h`` into P = floor((T-L)/stride)+1 overlapping
         patches of length L, flatten, embed to d_model. Patch i covers
         timesteps [i*stride, i*stride+L), ending at end_i = i*stride+L-1.
      3. A learnable start token is prepended (index 0) and a triangular mask
         over the P+1 tokens enforces patch-level causality: with a fixed
         positive stride, end_i ≤ end_j iff i ≤ j, so token j only ever sees
         timesteps ≤ end_{j-1}.
      4. Per-timestep gather: timestep t reads the context of the **last
         complete patch** — index floor((t-L+1)/stride), or the start token
         when t < L-1 — so the current (incomplete) patch, which contains
         future steps, is never consulted.
      5. Head: MLP on ``[h_t, ctx_t]`` → scalar.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 2,
        dropout: float = 0.1,
        patch_len: int = 16,
        patch_stride: int = 8,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError(f"patch_len/patch_stride must be positive, got {patch_len}/{patch_stride}")
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.in_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Patch embed reads the projected stream (L*d_model), not raw features
        # (L*K): keeps the patch pathway independent of K and reuses in_proj.
        self.patch_embed = nn.Linear(patch_len * d_model, d_model)
        max_patches = max_len // patch_stride + 2          # +1 start token, +1 slack
        self.patch_pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def _ctx_index(self, t: int, n_patches: int, device: torch.device) -> torch.Tensor:
        """Token index (into [start] + patches) of the last patch ending ≤ each timestep.

        last_complete(t) = floor((t - L + 1) / stride); -1 (→ start token, index 0)
        while no patch has completed. Clamped above for tail timesteps past the
        final patch end when (T - L) % stride != 0.
        """
        ts = torch.arange(t, device=device)
        last = torch.div(ts - (self.patch_len - 1), self.patch_stride, rounding_mode="floor")
        return last.clamp(min=-1, max=n_patches - 1) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :t, :]                       # (B, T, D)
        d = h.shape[-1]

        if t >= self.patch_len:
            patches = h.unfold(1, self.patch_len, self.patch_stride)   # (B, P, D, L)
            p = patches.shape[1]
            tok = self.patch_embed(patches.reshape(b, p, d * self.patch_len))
        else:                                                          # day shorter than one patch
            p = 0
            tok = h.new_zeros(b, 0, d)
        tok = torch.cat([self.start_token.expand(b, 1, d), tok], dim=1)
        tok = tok + self.patch_pos[:, : p + 1, :]

        mask = torch.triu(torch.ones(p + 1, p + 1, device=x.device, dtype=torch.bool), diagonal=1)
        ctx = self.ctx_norm(self.encoder(tok, mask=mask, is_causal=True))  # (B, P+1, D)

        ctx_t = ctx[:, self._ctx_index(t, p, x.device), :]             # (B, T, D)
        return self.head(torch.cat([h, ctx_t], dim=-1)).squeeze(-1)


class PatchTSTModel(_BaseTransformerModel):
    """Causal PatchTST — patch-token attention as the temporal inductive bias.

    Reuses ``_BaseTransformerModel``'s per-date training / predict / online
    ``lr_refit`` loop (same hooks as transformer / itransformer / timexer);
    only the backbone differs. ``patch_len``/``patch_stride`` control the
    patch grid (defaults 16/8 → 120 overlapping patches over a 968-step day,
    fresh context at worst stride-1 = 7 steps stale).
    """

    def __init__(self, **kw) -> None:
        self.patch_len = int(kw.pop("patch_len", 16))
        self.patch_stride = int(kw.pop("patch_stride", 8))
        kw.setdefault("signature_channels", None)      # patching is the point here
        super().__init__(**kw)

    def _build(self, input_size: int) -> nn.Module:
        self.input_size = input_size
        return CausalPatchTSTNet(
            input_size=input_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            max_len=self.max_len,
        ).to(self.device)


# ---------------------------------------------------------------------------
# causality self-test
# ---------------------------------------------------------------------------

def causality_self_test(verbose: bool = True) -> None:
    """Perturb future timesteps on random tensors; assert exactly-zero past deltas.

    Why exact zero (not a tolerance): masked softmax assigns weight exactly
    0.0 to future patch tokens and 0.0 * v = 0.0 in IEEE floats, while every
    other op (LayerNorm, Linear, gather) is per-position — so a causal graph
    yields *bitwise identical* past outputs, and any nonzero delta is a leak.
    Probe points straddle patch boundaries (L=16, stride=8 → ends at
    15, 23, 31, …) where an off-by-one in the last-complete-patch index or
    forward-fill-style leakage would show up first.
    """
    torch.manual_seed(0)
    net = CausalPatchTSTNet(
        input_size=10, d_model=32, n_heads=4, n_layers=2, ff_mult=2,
        dropout=0.1, patch_len=16, patch_stride=8, max_len=128,
    )
    net.eval()
    b, t, k = 3, 96, 10
    x = torch.randn(b, t, k)
    with torch.no_grad():
        base = net(x)
        assert base.shape == (b, t)
        short = net(x[:, :12, :])                      # exercise the T < patch_len path
        assert short.shape == (b, 12)
        for t0 in (1, 8, 15, 16, 17, 23, 24, 40, 63, 95):
            xp = x.clone()
            xp[:, t0:, :] += 100.0 * torch.randn(b, t - t0, k)
            pert = net(xp)
            delta = (pert[:, :t0] - base[:, :t0]).abs().max().item() if t0 > 0 else 0.0
            assert delta == 0.0, (
                f"LEAK: perturbing t >= {t0} changed predictions at t < {t0} "
                f"(max |delta| = {delta:.3e})"
            )
            moved = (pert[:, t0:] - base[:, t0:]).abs().max().item()
            assert moved > 0.0, f"degenerate: perturbing t >= {t0} changed nothing"
            if verbose:
                print(f"  t0={t0:>3}: past delta = 0.0 exactly, future moved by {moved:.3e}  ok")
    if verbose:
        print("causality self-test PASSED (bitwise-zero past deltas at all probe points)")


if __name__ == "__main__":
    causality_self_test()
