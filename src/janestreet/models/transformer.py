"""Transformer time-series model + Volterra-signature-augmented variant.

The vanilla ``TransformerModel`` is a per-symbol causal-attention encoder
over the (T, K) per-date sequence. We treat each symbol's daily timeline as
one example, exactly like the recurrent baseline.

The signature variant ``SignatureTransformerModel`` prepends a
``SignatureBlock`` so each time step's input is the raw feature vector
concatenated with the truncated Volterra signature of the most recent window
(over a configurable subset of channels). The hypothesis: the signature
gives the attention layer a compact representation of recent path geometry
(quadratic variation, mixed integrals, rough kernel weighting) that a
recurrent state has to discover from scratch.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from janestreet.data.dataset import DateBatchDataset, flatten_collate_fn
from janestreet.models._torch_utils import (
    auto_device,
    reshape_flat_to_sequence,
    reshape_sequence_to_flat,
)
from janestreet.models.base import BaseModel, FitData
from janestreet.models.signature import SignatureBlock
from janestreet.training.loss import WeightedR2Loss
from janestreet.training.metrics import r2_weighted_torch


class CausalTransformer(nn.Module):
    """Causal-attention encoder per symbol-day sequence."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
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
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, t, _ = x.shape
        h = self.in_proj(x) + self.pos[:, :t, :]
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.head(h).squeeze(-1)


class _BaseTransformerModel(BaseModel):
    """Shared training / predict / update loop for the Transformer variants."""

    sequence_model = True

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        epochs: int = 25,
        batch_size: int = 1,
        early_stopping_patience: int = 6,
        grad_clip: float = 1.0,
        lr_refit: float = 3e-4,
        n_times: int = 968,
        max_len: int = 1024,
        seed: int = 42,
        device: str = "auto",
        # Optional signature pre-block. Depth 3 is the project-wide default
        # (see signature.SignatureBlock); override per-spec to revert to 2.
        signature_channels: list[int] | None = None,
        signature_window: int = 32,
        signature_depth: int = 3,
        signature_hurst: float | None = 0.1,
        # ``signature_mode``: "signature" (classical, default) or
        # "log_signature" (Lie-algebra-projected, ~half the dim at depth 2,
        # components linearly independent — see signature.py).
        signature_mode: str = "signature",
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip
        self.lr_refit = lr_refit
        self.n_times = n_times
        self.max_len = max_len
        self.seed = seed
        self.device = auto_device(device)
        self.signature_channels = signature_channels
        self.signature_window = signature_window
        self.signature_depth = signature_depth
        self.signature_hurst = signature_hurst
        self.signature_mode = signature_mode

        self.criterion = WeightedR2Loss()
        self.sig_block: SignatureBlock | None = None
        self.model: nn.Module | None = None
        self.input_size: int | None = None

    # ------------------------------------------------------------------
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.sig_block is None:
            return x
        return self.sig_block(x)

    def _build(self, input_size: int) -> nn.Module:
        if self.signature_channels:
            self.sig_block = SignatureBlock(
                channels=self.signature_channels,
                window=self.signature_window,
                depth=self.signature_depth,
                hurst=self.signature_hurst,
                mode=self.signature_mode,
            ).to(self.device)
            input_size = input_size + self.sig_block.sig_dim
        self.input_size = input_size
        return CausalTransformer(
            input_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_mult=self.ff_mult,
            dropout=self.dropout,
            max_len=self.max_len,
        ).to(self.device)

    # ------------------------------------------------------------------
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
            print(f"[transformer] device={self.device} input={self.input_size}")
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
                x_aug = self._augment(x)
                pred = self.model(x_aug)
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
                x_aug = self._augment(x)
                pred = model(x_aug)
                ys.append(y.flatten().cpu())
                ws.append(w.flatten().cpu())
                ps.append(pred.flatten().cpu())
            if self.lr_refit > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=self.lr_refit, weight_decay=self.weight_decay)
                opt.zero_grad()
                model.train()
                x_aug = self._augment(x)
                pred = model(x_aug)
                loss = self.criterion(pred, y, w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                opt.step()
        return float(r2_weighted_torch(torch.cat(ys), torch.cat(ps), torch.cat(ws)))

    # ------------------------------------------------------------------
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
            x_aug = self._augment(X_t)
            pred = self.model(x_aug)
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
        pred = self.model(self._augment(X_t))
        loss = self.criterion(pred.flatten(), y_t.flatten(), w_t.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        opt.step()


class TransformerModel(_BaseTransformerModel):
    """Vanilla causal Transformer over the per-date sequence."""

    def __init__(self, **kw) -> None:
        kw.setdefault("signature_channels", None)
        super().__init__(**kw)


class SignatureTransformerModel(_BaseTransformerModel):
    """Volterra-signature-augmented Transformer.

    By default takes the signature of the 16 high-correlation feature
    indices (in the augmented matrix layout). Override
    ``signature_channels`` if your feature ordering differs.
    """

    def __init__(self, **kw) -> None:
        if "signature_channels" not in kw or kw["signature_channels"] is None:
            # Default: the first 16 channels — matches the COLS_FEATURES_CORR
            # block as long as the feature builder hasn't reordered them.
            kw["signature_channels"] = list(range(16))
        super().__init__(**kw)
