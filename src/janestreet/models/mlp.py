"""MLP baseline — tabular, no time recurrence.

Used as a sanity check: a competent MLP on the engineered features should
score noticeably worse than the recurrent ``ModelR`` on the same input. If
it doesn't, something is wrong upstream.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from janestreet.models._torch_utils import auto_device
from janestreet.models.base import BaseModel, FitData
from janestreet.training.loss import WeightedR2Loss
from janestreet.training.metrics import r2_weighted_torch


class MLPNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        dims = [input_size, *hidden_sizes]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPModel(BaseModel):
    sequence_model = False

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        epochs: int = 20,
        batch_size: int = 8192,
        early_stopping_patience: int = 3,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [512, 512, 256]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.device = auto_device(device)
        self.criterion = WeightedR2Loss()
        self.model: nn.Module | None = None

    def fit(self, train: FitData, valid: FitData | None = None, verbose: bool = False) -> None:
        torch.manual_seed(self.seed)
        self.model = MLPNet(train.X.shape[1], self.hidden_sizes, self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.from_numpy(train.X).float()
        y_t = torch.from_numpy(train.y).float()
        w_t = torch.from_numpy(train.w).float()
        n = X_t.shape[0]

        Xv: torch.Tensor | None = None
        yv: torch.Tensor | None = None
        wv: torch.Tensor | None = None
        if valid is not None:
            Xv = torch.from_numpy(valid.X).float().to(self.device)
            yv = torch.from_numpy(valid.y).float().to(self.device)
            wv = torch.from_numpy(valid.w).float().to(self.device)

        best_r2 = -np.inf
        best_state: dict | None = None
        no_improve = 0
        for epoch in range(self.epochs):
            self.model.train()
            idx = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                sl = idx[i:i + self.batch_size]
                x_b = X_t[sl].to(self.device)
                y_b = y_t[sl].to(self.device)
                w_b = w_t[sl].to(self.device)
                opt.zero_grad()
                pred = self.model(x_b)
                loss = self.criterion(pred, y_b, w_b)
                loss.backward()
                opt.step()
                tot += loss.item() * x_b.shape[0]

            v_r2 = float("nan")
            if Xv is not None and yv is not None and wv is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_v = self.model(Xv)
                v_r2 = float(r2_weighted_torch(yv, pred_v, wv))
            if verbose:
                print(f"epoch {epoch+1}: train_loss={tot/n:.4f}  val_R2={v_r2:.4f}")

            if v_r2 > best_r2:
                best_r2 = v_r2
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.early_stopping_patience + 1:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(
        self, X: np.ndarray, n_times: int | None = None, state: object | None = None
    ) -> tuple[np.ndarray, object | None]:
        assert self.model is not None
        self.model.eval()
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy(), None
