"""MLP + Volterra signature — a non-recurrent path-aware baseline.

Why this exists
---------------

Our ensemble already has:

* XGB (tree, per-row, no signature)
* GRU-ModelR / LSTM-ModelR (recurrent, implicit path memory in the hidden
  state)
* Signature-Transformer (attention + explicit signature)

All three model temporal structure through hidden state or attention. This
module adds a **stateless, per-timestep MLP** that gets the temporal info
*only* through an explicit signature feature. Concretely, at each
``(symbol, date, time_id)`` we feed the MLP:

    [ raw_feature_vec_at_t  ||  signature(rolling_window_ending_at_t) ]

and predict ``responder_6_t``. No cross-time modeling other than the
signature — a very different geometry than what any of our other models
exploit, so the predictions ought to be less correlated with the RNN
streams.

Implementation
--------------

Mirrors the sig-augmentation + per-date-sequence machinery of
``_BaseTransformerModel`` (same online-refit hook, same fit/predict/update
signatures, same interaction with ``FitData``) but replaces the causal
transformer with a plain feed-forward net applied independently at each
timestep.
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


class _PerTimestepMLP(nn.Module):
    """Applies a plain MLP to the last (feature) dim of a (D, T, K) tensor.

    No cross-time modeling — the network is completely stateless in the
    time axis. Any temporal info the caller cares about must be in the
    input features themselves (which is what the signature block gives us).
    """

    def __init__(
        self, input_size: int, hidden_sizes: list[int], dropout: float
    ) -> None:
        super().__init__()
        dims = [input_size, *hidden_sizes]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (D, T, K) -> (D, T)
        return self.net(x).squeeze(-1)


class MLPWithSignatureModel(BaseModel):
    """Signature-augmented MLP — stateless in the time axis.

    Kwargs mirror the transformer variants where they overlap
    (``signature_channels``, ``signature_window``, ``signature_depth``,
    ``signature_hurst``, ``signature_mode``, ``lr_refit``) so the same
    channel / depth / mode choices port over cleanly.
    """

    sequence_model = True  # the pipeline needs to know n_times to reshape

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        epochs: int = 20,
        batch_size: int = 1,  # one date at a time — same as recurrent / transformer
        early_stopping_patience: int = 4,
        grad_clip: float = 1.0,
        lr_refit: float = 3e-5,
        n_times: int = 968,
        seed: int = 42,
        device: str = "auto",
        # Signature — mirrors _BaseTransformerModel keyword surface.
        signature_channels: list[int] | None = None,
        signature_window: int = 16,
        signature_depth: int = 3,
        signature_hurst: float | None = 0.1,
        signature_mode: str = "signature",
    ) -> None:
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [512, 256, 128]
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
        return _PerTimestepMLP(input_size, self.hidden_sizes, self.dropout).to(self.device)

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
            print(f"[mlp_sig] device={self.device} input={self.input_size}")
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
