"""Volterra-style path signature features.

We compute a truncated, time-augmented path signature on rolling windows of
the price-like channels. Three flavors:

* ``compute_signature``: the standard tensor algebra signature of the path
  ``X̃: [0, 1] -> R^{d+1}`` with the first coordinate being ``t``. Depth M.
  Output dimension = (d+1) + (d+1)^2 + ... + (d+1)^M.

* ``compute_volterra_signature``: weights each tensor word by a power-law
  kernel ``(t - s)^{H - 1/2}``. The Hurst index ``H`` lets us capture the
  *rough Volterra* behavior the Jaber/Hager school exploits — small ``H``
  emphasizes recent updates, ``H = 1/2`` recovers the classical signature.

* ``compute_log_signature`` / ``compute_volterra_log_signature``: the
  log of the signature tensor series, truncated to the same depth. The
  log-signature lives in the free Lie algebra rather than the free tensor
  algebra, so it has *much smaller, linearly independent* components — the
  shuffle redundancies of the signature collapse. At depth 2 the level-2
  component is exactly the Lévy area, antisymmetric in its two indices,
  with only ``d(d-1)/2`` independent entries (vs the signature's ``d²``).

  Formula (truncated log of ``1 + S₁ + S₂ + ... + S_M``):

      L₁ = S₁
      L₂ = S₂ − S₁⊗S₁ / 2
      L₃ = S₃ − (S₁⊗S₂ + S₂⊗S₁)/2 + S₁⊗S₁⊗S₁ / 3

  At depth 2 we additionally extract the strict upper triangle of L₂
  (the level-2 log-signature is antisymmetric) so the final feature vector
  has size ``d + d(d-1)/2`` instead of ``d + d²``. At depth 3 we keep the
  full d³ tensor — entries are linearly independent in the Lie-algebra
  sense, but representing them in the Lyndon basis is more involved; the
  downstream linear layer handles the (small) further redundancy.

The implementation is a clean Chen-iteration: we maintain the running
signature ``S_n`` as a list of tensors and update incrementally on each
linear increment. Vectorized across windows and channels. ``M <= 3`` is
the realistic regime for our channel counts.

These are PyTorch tensors so the computation is GPU-friendly when batched.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# Optional Lyndon-basis log-signature via iisignature. Heavily-optimized C++,
# returns the true Lie-algebra-dim log-sig (no shuffle redundancies left at
# any level). Falls back gracefully — see SignatureBlock(mode=...).
try:
    import iisignature as _iisig  # type: ignore[import-untyped]

    _HAS_IISIG = True
except ImportError:
    _iisig = None
    _HAS_IISIG = False


def _chen_step(prev: list[torch.Tensor], delta: torch.Tensor, depth: int) -> list[torch.Tensor]:
    """Update list of signature tensors ``[level1, ..., level_depth]`` by one increment.

    ``prev`` may also be a partial running signature; ``delta`` is shape
    ``(..., d)``. Levels are kept symmetric tensors of growing rank.
    """
    new = [prev[0] + delta]
    for k in range(2, depth + 1):
        # level_k of (X + δ) up to chen: sum_{j=0..k} (1/j!) level_{k-j}(X) ⊗ δ^{⊗j}
        # We unroll for clarity (small k).
        acc = prev[k - 1].clone() if len(prev) >= k else torch.zeros_like(_outer(prev[0], k))
        for j in range(1, k + 1):
            coeff = 1.0 / math.factorial(j)
            outer = _outer(delta, j)
            lower = prev[k - 1 - j] if (k - 1 - j) >= 0 else None
            if lower is None:
                # j == k: pure delta tensor at level k
                acc = acc + coeff * outer
            else:
                acc = acc + coeff * _tensor_product(lower, outer)
        new.append(acc)
    return new


def _outer(v: torch.Tensor, k: int) -> torch.Tensor:
    """k-fold outer product of v along the last axis."""
    out = v
    for _ in range(k - 1):
        out = torch.einsum("...i,...j->...ij", out, v)
        # flatten the last two dims so subsequent einsum dims line up
        out = out.flatten(-2)
    return out


def _tensor_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tensor (Kronecker) product across the last dim: (..., D_a, D_b) -> (..., D_a*D_b)."""
    da = a.shape[-1]
    db = b.shape[-1]
    out = torch.einsum("...i,...j->...ij", a, b).reshape(*a.shape[:-1], da * db)
    return out


def compute_signature(path: torch.Tensor, depth: int = 2) -> torch.Tensor:
    """Truncated path signature of ``path`` of shape (batch, n, d).

    Returns a flat feature vector of length ``sum_{k=1..depth} d^k`` per batch
    item.

    Parameters
    ----------
    path:
        Tensor of shape (B, N, d). Successive rows along dim 1 are the path
        vertices; differences are the linear increments.
    depth:
        Truncation level. ``depth=1`` is just the displacement; ``depth=2``
        adds the Lévy-area-like second moments; ``depth=3`` adds skewness.
    """
    b, n, d = path.shape
    delta = path[:, 1:] - path[:, :-1]  # (B, N-1, d)
    sig: list[torch.Tensor] = [torch.zeros(b, d, device=path.device, dtype=path.dtype)]
    for k in range(2, depth + 1):
        sig.append(torch.zeros(b, d ** k, device=path.device, dtype=path.dtype))
    for i in range(delta.shape[1]):
        sig = _chen_step(sig, delta[:, i, :], depth)
    return torch.cat(sig, dim=-1)


def compute_volterra_signature(
    path: torch.Tensor, depth: int = 2, hurst: float = 0.1
) -> torch.Tensor:
    """Time-weighted (Volterra-rough) signature.

    Each increment ``δ_i`` is rescaled by ``(t_N - t_i)^{H - 0.5}`` before
    being fed into the Chen iteration. ``H = 1/2`` reduces to the standard
    signature; ``H < 1/2`` (rough regime) gives more weight to recent steps.
    """
    b, n, d = path.shape
    delta = path[:, 1:] - path[:, :-1]
    # time grid in [0, 1] — uniform spacing assumption.
    t = torch.linspace(0.0, 1.0, n, device=path.device, dtype=path.dtype)
    weights = (1.0 - t[:-1] + 1e-6) ** (hurst - 0.5)
    delta = delta * weights.view(1, -1, 1)
    sig: list[torch.Tensor] = [torch.zeros(b, d, device=path.device, dtype=path.dtype)]
    for k in range(2, depth + 1):
        sig.append(torch.zeros(b, d ** k, device=path.device, dtype=path.dtype))
    for i in range(delta.shape[1]):
        sig = _chen_step(sig, delta[:, i, :], depth)
    return torch.cat(sig, dim=-1)


# ---------------------------------------------------------------------------
# Log-signature helpers
# ---------------------------------------------------------------------------
def _logsig_dim(d: int, depth: int) -> int:
    """Output dimension of ``compute_log_signature`` for d channels at given depth.

    * Depth 1: d (vector).
    * Depth 2: d + d(d-1)/2 — level 2 collapses to its strict upper triangle
      (the antisymmetric Lévy area is the only nonzero log-sig component at L₂).
    * Depth ≥ 3: d + d(d-1)/2 + d³ — we keep the full L₃ tensor since the
      Lie-algebra basis at level 3 is more involved (Witt/Möbius dim
      ``(d³ − d)/3``) and we let the downstream linear layer absorb the
      residual redundancy.

    TODO(run-3): project L₃ onto a Lyndon basis. At d=9, that would shrink
    the level-3 component from 729 to (d³-d)/3 = 240 dims (further 67% cut
    on top of the depth-2 win). For now the L₃ shuffle-product entries are
    still zeroed by the (S₁⊗S₂ + S₂⊗S₁)/2 subtraction, so the components
    are linearly dependent but the redundancy is benign for ML use.
    """
    if depth < 1:
        return 0
    dim = d
    if depth >= 2:
        dim += (d * (d - 1)) // 2
    if depth >= 3:
        dim += d ** 3
    if depth > 3:
        raise NotImplementedError("log-signature only supported up to depth=3 here")
    return dim


def _signature_levels(
    path: torch.Tensor, depth: int, hurst: float | None,
) -> list[torch.Tensor]:
    """Run Chen iteration and return ``[S₁, S₂, ..., S_depth]`` separately.

    Shape of ``S_k`` is ``(B, d^k)``. ``hurst=None`` gives the standard
    signature; numeric ``hurst`` triggers the Volterra-rough weighting.
    """
    b, n, d = path.shape
    delta = path[:, 1:] - path[:, :-1]
    if hurst is not None:
        t = torch.linspace(0.0, 1.0, n, device=path.device, dtype=path.dtype)
        weights = (1.0 - t[:-1] + 1e-6) ** (hurst - 0.5)
        delta = delta * weights.view(1, -1, 1)
    sig: list[torch.Tensor] = [torch.zeros(b, d, device=path.device, dtype=path.dtype)]
    for k in range(2, depth + 1):
        sig.append(torch.zeros(b, d ** k, device=path.device, dtype=path.dtype))
    for i in range(delta.shape[1]):
        sig = _chen_step(sig, delta[:, i, :], depth)
    return sig


def compute_log_signature(
    path: torch.Tensor, depth: int = 2, hurst: float | None = None,
) -> torch.Tensor:
    """Truncated log-signature of ``path`` of shape ``(B, N, d)``.

    Computes the signature levels then forms

        L₁ = S₁
        L₂ = S₂ − S₁⊗S₁ / 2
        L₃ = S₃ − (S₁⊗S₂ + S₂⊗S₁)/2 + S₁⊗S₁⊗S₁ / 3

    At depth 2 we project L₂ to its strict upper triangle (antisymmetric
    Lévy area is the only nonzero log-sig component). Output dim per item
    follows ``_logsig_dim``.

    ``hurst=None`` reduces to the classical signature; numeric ``hurst``
    feeds the Chen iteration with the Volterra-rough weights, identical
    to ``compute_volterra_signature``.
    """
    sig = _signature_levels(path, depth, hurst)
    s1 = sig[0]  # (B, d)
    out: list[torch.Tensor] = [s1]
    b, d = s1.shape
    if depth >= 2:
        s2 = sig[1].view(b, d, d)
        # L₂ = S₂ − ½ S₁⊗S₁ — explicitly antisymmetric (shuffle id).
        s1_outer = torch.einsum("bi,bj->bij", s1, s1)
        l2 = s2 - 0.5 * s1_outer
        iu = torch.triu_indices(d, d, offset=1)
        out.append(l2[:, iu[0], iu[1]])  # (B, d*(d-1)/2)
    if depth >= 3:
        s3 = sig[2].view(b, d, d, d)
        # L₃ = S₃ − ½ (S₁⊗S₂ + S₂⊗S₁) + ⅓ S₁⊗S₁⊗S₁
        s1_s2 = torch.einsum("bi,bjk->bijk", s1, s2)
        s2_s1 = torch.einsum("bij,bk->bijk", s2, s1)
        s1_s1_s1 = torch.einsum("bi,bj,bk->bijk", s1, s1, s1)
        l3 = s3 - 0.5 * (s1_s2 + s2_s1) + (1.0 / 3.0) * s1_s1_s1
        out.append(l3.reshape(b, d ** 3))
    if depth > 3:
        raise NotImplementedError("log-signature only supported up to depth=3 here")
    return torch.cat(out, dim=-1)


def compute_volterra_log_signature(
    path: torch.Tensor, depth: int = 2, hurst: float = 0.1,
) -> torch.Tensor:
    """Hurst-weighted log-signature (rough-Volterra weighting in the Chen step)."""
    return compute_log_signature(path, depth=depth, hurst=hurst)


# ---------------------------------------------------------------------------
# Minimal (Lyndon-basis) log-signature via iisignature.
# ---------------------------------------------------------------------------
def _iisig_logsig_dim(d: int, depth: int) -> int:
    """Lie-algebra (Lyndon-basis) dim of the truncated log-signature."""
    if not _HAS_IISIG:
        raise ImportError(
            "iisignature is required for the minimal log-signature. "
            "Install with: uv sync --extra logsig-minimal"
        )
    return int(_iisig.logsiglength(d, depth))


# Cache iisignature's "prepare" output (the Lyndon basis precomputation) per
# (d, depth) — it's expensive (~ms) and the same call happens on every
# forward pass for every (sym, time-id).
_IISIG_PREPARE_CACHE: dict[tuple[int, int], object] = {}


def _iisig_prepare(d: int, depth: int) -> object:
    key = (d, depth)
    if key not in _IISIG_PREPARE_CACHE:
        if not _HAS_IISIG:
            raise ImportError("iisignature missing")
        # Method letters (iisignature 0.24):
        #   'C' — compiled per (d, depth), fastest but overflows an internal
        #         buffer at d≥9, depth=3 on this build
        #   'D' — default numeric — same overflow
        #   'S' — symbolic — slower but the only one that actually works at
        #         our config. Gives the true Lyndon-basis output.
        _IISIG_PREPARE_CACHE[key] = _iisig.prepare(d, depth, "S")
    return _IISIG_PREPARE_CACHE[key]


def _hurst_reweight_path(path: torch.Tensor, hurst: float) -> torch.Tensor:
    """Replace path X with the cumulated path of Hurst-weighted increments.

    ``compute_volterra_signature`` rescales each increment δ_i by
    ``(1 − t_i)^{H − 0.5}`` before the Chen step. To get the same
    Volterra-weighted log-signature out of iisignature (which only knows
    about the *path*, not the increments), we reconstruct the path whose
    increments are the weighted ones: ``Y_t = ∫₀^t w(s) dX_s``.
    """
    b, n, d = path.shape
    delta = path[:, 1:] - path[:, :-1]
    t = torch.linspace(0.0, 1.0, n, device=path.device, dtype=path.dtype)
    weights = (1.0 - t[:-1] + 1e-6) ** (hurst - 0.5)
    delta_w = delta * weights.view(1, -1, 1)
    # Y_0 = X_0; Y_i = X_0 + sum_{k<i} weighted_delta_k.
    head = path[:, :1, :]
    tail = head + delta_w.cumsum(dim=1)
    return torch.cat([head, tail], dim=1)


def compute_log_signature_minimal(
    path: torch.Tensor, depth: int = 3, hurst: float | None = 0.1,
) -> torch.Tensor:
    """Lyndon-basis truncated log-signature via iisignature.

    Output dim = ``iisignature.logsiglength(d, depth)`` — the true Lie-algebra
    dim. At d=9, depth=3 this is ``9 + 36 + 240 = 285`` (vs the naive
    ``compute_log_signature``'s 774 and the classical signature's 819).

    Hurst weighting (Volterra-rough variant): when ``hurst`` is not None we
    rebuild the path so its increments carry the power-law weights, then
    call iisignature on the new path. Mathematically equivalent to
    pre-weighting the increments in the Chen iteration.

    Notes
    -----
    * iisignature is a numpy/C++ library — it does NOT carry autograd. We
      detach the input, call iisignature in numpy, and wrap the result back
      in a torch tensor. This is safe here because ``SignatureBlock`` has
      no learnable parameters and its input is a leaf tensor (no upstream
      params want gradients through the signature).
    * The cached ``_iisig_prepare`` call is reused across windows / batches.
    """
    if not _HAS_IISIG:
        raise ImportError(
            "iisignature is required for compute_log_signature_minimal. "
            "Install with: uv sync --extra logsig-minimal"
        )
    if hurst is not None:
        path = _hurst_reweight_path(path, hurst)
    b, n, d = path.shape
    s = _iisig_prepare(d, depth)
    arr = path.detach().cpu().to(torch.float64).numpy()
    # iisignature.logsig supports a leading batch dim: shape (B, N, d) → (B, K)
    out_np = _iisig.logsig(arr, s)
    return torch.from_numpy(np.ascontiguousarray(out_np)).to(
        device=path.device, dtype=path.dtype,
    )


# ---------------------------------------------------------------------------
class SignatureBlock(nn.Module):
    """Drop-in feature extractor: takes (D, T, K) and emits (D, T, K + sig_dim).

    For each time step it computes the signature (or log-signature) of the
    most-recent ``window`` increments along ``channels`` columns of the input.
    Earlier than ``window`` we pad with zeros.

    This is a *learning-free* feature extractor; the model on top decides how
    to use it. Cost is O(T · window · depth · K^depth), so keep ``channels``
    small (4–8) and depth ≤ 3.

    ``mode``:
        ``"signature"``              — classical truncated signature; size
                                       grows as ``d + d² + d³ + …`` (highly
                                       redundant via shuffle products).
        ``"log_signature"``          — log of the signature tensor series.
                                       Level-2 projected to its strict upper
                                       triangle (Lévy area); level-3 kept as
                                       a full d³ tensor. Pure-PyTorch impl,
                                       no external deps. At depth 2 this is
                                       ~half the size of the signature; at
                                       depth 3 the saving is small (~5 %).
        ``"log_signature_minimal"``  — Lyndon-basis log-signature via
                                       iisignature (optional dep). Realises
                                       the full dim reduction at all depths
                                       — e.g. at d=9 depth=3, 819 → 285
                                       (−65 %). Requires ``iisignature``;
                                       see pyproject's logsig-minimal
                                       optional group.
    """

    _ALLOWED_MODES = ("signature", "log_signature", "log_signature_minimal")

    def __init__(
        self,
        channels: list[int],
        window: int = 32,
        # Depth 3 is the project-wide default — depth 2 is "below the
        # interesting regime" for the Jane Street task (Lévy area alone
        # underfits) and the Phase C₁ run already confirmed depth=3 has
        # the highest static R² of any sig variant. Override per-spec for
        # the legacy depth-2 experiments.
        depth: int = 3,
        hurst: float | None = 0.1,
        include_time: bool = True,
        mode: str = "signature",
    ) -> None:
        super().__init__()
        if mode not in self._ALLOWED_MODES:
            raise ValueError(
                f"mode must be one of {self._ALLOWED_MODES}, got {mode!r}"
            )
        if mode == "log_signature_minimal" and not _HAS_IISIG:
            raise ImportError(
                "mode='log_signature_minimal' requires iisignature — install "
                "with `uv sync --extra logsig-minimal`"
            )
        self.channels = channels
        self.window = window
        self.depth = depth
        self.hurst = hurst
        self.include_time = include_time
        self.mode = mode
        d = len(channels) + (1 if include_time else 0)
        if mode == "signature":
            self.sig_dim = sum(d ** k for k in range(1, depth + 1))
        elif mode == "log_signature":
            self.sig_dim = _logsig_dim(d, depth)
        else:  # log_signature_minimal
            self.sig_dim = _iisig_logsig_dim(d, depth)
        self.d = d

    def _compute(self, seg: torch.Tensor) -> torch.Tensor:
        if self.mode == "signature":
            return (
                compute_volterra_signature(seg, self.depth, self.hurst)
                if self.hurst is not None
                else compute_signature(seg, self.depth)
            )
        if self.mode == "log_signature":
            return compute_log_signature(seg, self.depth, self.hurst)
        # log_signature_minimal — iisignature returns the Lyndon basis
        return compute_log_signature_minimal(seg, self.depth, self.hurst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_b, t, _ = x.shape
        sigs = torch.zeros(d_b, t, self.sig_dim, device=x.device, dtype=x.dtype)
        sel = x[..., self.channels]  # (D, T, c)
        if self.include_time:
            tt = torch.linspace(0.0, 1.0, t, device=x.device, dtype=x.dtype)
            sel = torch.cat([tt.view(1, t, 1).expand(d_b, t, 1), sel], dim=-1)
        for end in range(1, t + 1):
            start = max(0, end - self.window)
            seg = sel[:, start:end, :]
            if seg.shape[1] < 2:
                continue
            sigs[:, end - 1, :] = self._compute(seg)
        return torch.cat([x, sigs], dim=-1)
