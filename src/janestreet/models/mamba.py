"""Mamba (selective state-space) model — pure-PyTorch, no CUDA scan.

Why this exists
---------------

Our current lineup is LSTM/GRU (bounded hidden state, gated) + transformer
(O(T²) attention). Mamba (Gu & Dao, 2023) is a third family: a linear
state-space model with *data-dependent* transition/input matrices —
mathematically closer to a continuous-time linear system than either RNN
or transformer. It typically has

* Longer effective context than a fixed-size LSTM hidden state
* Lower compute than attention at long T (linear, not quadratic)
* Different inductive bias — the "selective" gating decides which inputs
  advance the state

We register it here as a first-class model. The upstream CUDA-accelerated
selective_scan is unavailable on the Mac; we replace it with a plain
sequential Python scan over T. Same asymptotic cost as our LSTM path
(O(T · d_state · d_inner)) — acceptable for our T = 968.

Ref
---
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Gu & Dao, 2023. https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from janestreet.data.dataset import DateBatchDataset, flatten_collate_fn
from janestreet.models._torch_utils import (
    auto_device,
    reshape_flat_to_sequence,
    reshape_sequence_to_flat,
)
from janestreet.models.base import BaseModel, FitData
from janestreet.training.loss import WeightedR2Loss
from janestreet.training.metrics import r2_weighted_torch


class MambaBlock(nn.Module):
    """Single selective-SSM block.

    Follows the S6 formulation from the Mamba paper:

    1. Input projection ``x → (u, res)`` — u feeds the SSM, res is a gate.
    2. Local (causal) 1D conv on ``u`` — mixes a small time window before
       the state.
    3. Data-dependent projection ``u → (Δ, B, C)``. Δ ∈ R^{d_inner} is the
       discretisation step; B, C ∈ R^{d_state} are input/output matrices.
    4. Discretise the fixed A: ``Ā = exp(Δ ⊙ A)``, ``B̄ = Δ ⊙ B``.
    5. Sequential scan: ``hₜ = Ā_t · h_{t-1} + B̄_t · uₜ``, then
       ``yₜ = C · hₜ + D · uₜ``.
    6. Gate with ``silu(res)`` and project back to ``d_model``. Residual.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # (1) input → (u, res)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # (2) causal depthwise conv over T
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,      # left-pad amount; we truncate right side
            groups=self.d_inner,
        )

        # (3) data-dependent projection to (dt_raw, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.d_inner + 2 * d_state, bias=False,
        )
        # Per-channel Δ projection (typical Mamba init: bias for a warm-start Δ)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        with torch.no_grad():
            # softplus^{-1}(range) so post-softplus Δ ∈ [dt_min, dt_max] initially
            dt = torch.exp(
                torch.rand(self.d_inner) * (np.log(dt_max) - np.log(dt_min))
                + np.log(dt_min)
            )
            inv_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus)

        # (4) A is fixed, stored via log for positivity: A = -exp(A_log) < 0
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        # D: direct feed-through per channel
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # (6) output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model). Output: (B, L, d_model), residual added.
        b, ell, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2·d_inner)
        u, res = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal conv over L. Conv1d with padding=k-1 emits L+k-1 samples;
        # keep the first L (causal alignment).
        u = u.transpose(1, 2)                     # (B, d_inner, L)
        u = self.conv1d(u)[..., :ell]             # (B, d_inner, L)
        u = F.silu(u)
        u = u.transpose(1, 2)                     # (B, L, d_inner)

        # Data-dependent (dt_raw, B, C)
        xdbl = self.x_proj(u)  # (B, L, d_inner + 2·d_state)
        dt_raw, B_x, C_x = torch.split(
            xdbl, [self.d_inner, self.d_state, self.d_state], dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw))       # (B, L, d_inner)

        A = -torch.exp(self.A_log)                  # (d_inner, d_state)

        # Ā = exp(dt ⊙ A)      (B, L, d_inner, d_state)
        # B̄ = dt ⊙ B_x         (B, L, d_inner, d_state) — B_x broadcast over d_inner
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_x.unsqueeze(2)

        # Sequential scan (pure PyTorch — no selective_scan CUDA kernel here).
        # h has shape (B, d_inner, d_state).
        h = torch.zeros(
            b, self.d_inner, self.d_state, device=x.device, dtype=x.dtype,
        )
        ys = []
        for t in range(ell):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            # y_t = h · C  + D · u
            y_t = (h * C_x[:, t].unsqueeze(1)).sum(dim=-1) + self.D * u[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                  # (B, L, d_inner)

        # Gate & output projection & residual
        y = y * F.silu(res)
        return self.out_proj(y) + x


class MambaStack(nn.Module):
    """Stack of MambaBlocks with input/output projections and a scalar head."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 96,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk, norm in zip(self.blocks, self.norms, strict=True):
            # Pre-norm variant — Mamba's own paper uses this
            h = blk(norm(self.dropout(h)))
        return self.head(self.final_norm(h)).squeeze(-1)


class MambaModel(BaseModel):
    """Selective SSM (Mamba) model — same interface as Recurrent/Transformer variants."""

    sequence_model = True

    def __init__(
        self,
        d_model: int = 96,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        epochs: int = 12,
        batch_size: int = 1,
        early_stopping_patience: int = 4,
        grad_clip: float = 1.0,
        lr_refit: float = 3e-4,
        n_times: int = 968,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        self.lr_refit = lr_refit
        self.n_times = n_times
        self.seed = seed
        self.device = auto_device(device)
        self.criterion = WeightedR2Loss()
        self.model: nn.Module | None = None
        self.input_size: int | None = None

    def _build(self, input_size: int) -> nn.Module:
        self.input_size = input_size
        return MambaStack(
            input_size, self.d_model, self.n_layers,
            self.d_state, self.d_conv, self.expand, self.dropout,
        ).to(self.device)

    def fit(self, train: FitData, valid: FitData | None = None, verbose: bool = False) -> None:
        torch.manual_seed(self.seed)
        train_ds = DateBatchDataset(
            train.X, train.resp, train.y, train.w,
            train.symbols, train.dates, train.times,
            n_times=self.n_times, on_batch=False,
        )
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=flatten_collate_fn,
        )
        val_dl = None
        if valid is not None:
            val_ds = DateBatchDataset(
                valid.X, valid.resp, valid.y, valid.w,
                valid.symbols, valid.dates, valid.times,
                n_times=self.n_times, on_batch=True,
            )
            val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=flatten_collate_fn)

        self.model = self._build(train.X.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        if verbose:
            print(f"[mamba] device={self.device} d_model={self.d_model} n_layers={self.n_layers}")
        best_r2 = -np.inf
        best_state: dict | None = None
        no_improve = 0
        for epoch in range(self.epochs):
            self.model.train()
            tot = 0.0
            ys, ws, ps = [], [], []
            for x, _resp, y, w in train_dl:
                x, y, w = [t.to(self.device) for t in (x, y, w)]
                opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred.flatten(), y.flatten(), w.flatten())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                opt.step()
                tot += loss.item()
                ys.append(y.detach().flatten().cpu())
                ws.append(w.detach().flatten().cpu())
                ps.append(pred.detach().flatten().cpu())
            sched.step()
            tr_r2 = float(r2_weighted_torch(torch.cat(ys), torch.cat(ps), torch.cat(ws)))

            v_r2 = float("nan")
            if val_dl is not None:
                v_r2 = self._validate(val_dl)
            if verbose:
                print(
                    f"epoch {epoch+1:>3} | loss {tot/max(len(train_dl),1):.4f} | "
                    f"train_R² {tr_r2:.4f} | val_R² {v_r2:.4f}"
                )
            v_metric = v_r2 if val_dl is not None else tr_r2
            if v_metric > best_r2:
                best_r2 = v_metric
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.early_stopping_patience + 1:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _validate(self, dl: DataLoader) -> float:
        model = copy.deepcopy(self.model)
        ys, ws, ps = [], [], []
        for x, _resp, y, w in dl:
            x, y, w = [t.to(self.device) for t in (x, y, w)]
            model.eval()
            with torch.no_grad():
                pred = model(x)
                ys.append(y.flatten().cpu())
                ws.append(w.flatten().cpu())
                ps.append(pred.flatten().cpu())
            if self.lr_refit > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=self.lr_refit, weight_decay=self.weight_decay)
                opt.zero_grad()
                model.train()
                pred = model(x)
                loss = self.criterion(pred, y, w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                opt.step()
        return float(r2_weighted_torch(torch.cat(ys), torch.cat(ps), torch.cat(ws)))

    def predict(
        self, X: np.ndarray, n_times: int | None = None, state: object | None = None
    ) -> tuple[np.ndarray, object | None]:
        assert self.model is not None
        n_times = n_times if n_times is not None else self.n_times
        X_t = torch.from_numpy(X).float()
        X_t = reshape_flat_to_sequence(X_t, n_times).to(self.device)
        X_t = torch.nan_to_num(X_t, 0.0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t)
        return reshape_sequence_to_flat(pred).cpu().numpy(), None

    def update(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, n_times: int) -> None:
        if self.lr_refit <= 0 or self.model is None:
            return
        X_t = torch.from_numpy(X).float()
        y_t = torch.from_numpy(y).float()
        w_t = torch.from_numpy(w).float()
        X_t = reshape_flat_to_sequence(X_t, n_times).to(self.device)
        y_t = y_t.view(n_times, -1).swapaxes(0, 1).to(self.device)
        w_t = w_t.view(n_times, -1).swapaxes(0, 1).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr_refit, weight_decay=self.weight_decay)
        self.model.train()
        opt.zero_grad()
        pred = self.model(X_t)
        loss = self.criterion(pred.flatten(), y_t.flatten(), w_t.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        opt.step()
