"""GRU / LSTM models, including the Volkova ``ModelR`` 4-branch architecture.

Two layouts:

* ``aux_branches=False``: a single ``ModelRBase`` stack of recurrent + FC
  layers — straightforward.
* ``aux_branches=True``: four parallel ``ModelRBase`` stacks, each predicting
  one auxiliary responder, with a linear combiner producing the final ``y``.
  This is the Volkova replica.

Training loss = sum of weighted-R² on the primary target and on each aux.
Online refit uses only the primary head's loss with ``lr_refit``.
"""

from __future__ import annotations

import copy
from pathlib import Path

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
from janestreet.training.loss import WeightedR2Loss
from janestreet.training.metrics import r2_weighted_torch


class ModelRBase(nn.Module):
    """Stack of recurrent layers followed by a small FC head."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout_rates: list[float],
        hidden_sizes_linear: list[int],
        dropout_rates_linear: list[float],
        model_type: str = "gru",
    ) -> None:
        super().__init__()
        self.num_layers = len(hidden_sizes)
        self.rnns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = input_size if i == 0 else hidden_sizes[i - 1]
            cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(model_type)
            if cls is None:
                raise ValueError(f"model_type must be gru or lstm, got {model_type}")
            self.rnns.append(cls(in_dim, hidden_sizes[i], num_layers=1, batch_first=True))
            self.dropouts.append(nn.Dropout(dropout_rates[i]))
        n_in = hidden_sizes[-1] if hidden_sizes else input_size

        layers: list[nn.Module] = []
        if hidden_sizes_linear:
            for i, h in enumerate(hidden_sizes_linear):
                layers.append(nn.Linear(n_in if i == 0 else hidden_sizes_linear[i - 1], h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rates_linear[i]))
            layers.append(nn.Linear(hidden_sizes_linear[-1], 1))
        else:
            layers.append(nn.Linear(n_in, 1))
        self.fc = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, hidden: list | None = None
    ) -> tuple[torch.Tensor, list]:
        d, t, _ = x.shape
        if hidden is None:
            hidden = [None] * self.num_layers
        for i, rnn in enumerate(self.rnns):
            x, h = rnn(x, hidden[i])
            x = self.dropouts[i](x)
            hidden[i] = h
        x = self.fc(x.reshape(d * t, -1)).reshape(d, t)
        return x, hidden


class ModelR(nn.Module):
    """Four parallel ``ModelRBase`` branches + linear combiner — Volkova style."""

    def __init__(self, input_size: int, num_aux: int = 4, **base_kwargs) -> None:
        super().__init__()
        self.num_aux = num_aux
        self.branches = nn.ModuleList(
            [ModelRBase(input_size, **base_kwargs) for _ in range(num_aux)]
        )
        self.out = nn.Linear(num_aux, 1)

    def forward(
        self, x: torch.Tensor, hidden: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        d, t, _ = x.shape
        if hidden is None:
            hidden = [None] * self.num_aux
        aux: list[torch.Tensor] = []
        for i, br in enumerate(self.branches):
            z, h = br(x, hidden[i])
            aux.append(z.reshape(d * t, -1))
            hidden[i] = h
        out_resp = torch.cat(aux, dim=-1)  # (D*T, num_aux)
        y = self.out(out_resp).reshape(d, t)
        out_resp = out_resp.reshape(d, t, -1)
        return y, out_resp, hidden


# ---------------------------------------------------------------------------
class RecurrentModel(BaseModel):
    """GRU / LSTM model with optional aux-target heads.

    Hyperparameters mirror Volkova's NN class but are exposed as kwargs.
    """

    sequence_model = True

    def __init__(
        self,
        model_type: str = "gru",
        aux_branches: bool = True,
        num_aux: int = 4,
        hidden_sizes: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        hidden_sizes_linear: list[int] | None = None,
        dropout_rates_linear: list[float] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 1,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        early_stopping: bool = True,
        grad_clip: float = 1.0,
        lr_refit: float = 3e-4,
        n_times: int = 968,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.model_type = model_type
        self.aux_branches = aux_branches
        self.num_aux = num_aux
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 128, 128]
        self.dropout_rates = (
            dropout_rates if dropout_rates is not None else [0.1] * len(self.hidden_sizes)
        )
        self.hidden_sizes_linear = (
            hidden_sizes_linear if hidden_sizes_linear is not None else []
        )
        self.dropout_rates_linear = (
            dropout_rates_linear if dropout_rates_linear is not None else []
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = early_stopping
        self.grad_clip = grad_clip
        self.lr_refit = lr_refit
        self.n_times = n_times
        self.seed = seed
        self.device = auto_device(device)

        self.criterion = WeightedR2Loss()
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.best_epoch: int | None = None

    # ------------------------------------------------------------------
    def _build(self, input_size: int) -> nn.Module:
        if self.aux_branches:
            return ModelR(
                input_size,
                num_aux=self.num_aux,
                hidden_sizes=self.hidden_sizes,
                dropout_rates=self.dropout_rates,
                hidden_sizes_linear=self.hidden_sizes_linear,
                dropout_rates_linear=self.dropout_rates_linear,
                model_type=self.model_type,
            )
        return ModelRBase(
            input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rates=self.dropout_rates,
            hidden_sizes_linear=self.hidden_sizes_linear,
            dropout_rates_linear=self.dropout_rates_linear,
            model_type=self.model_type,
        )

    # ------------------------------------------------------------------
    def fit(
        self, train: FitData, valid: FitData | None = None, verbose: bool = False,
        warm_start: bool = False,
        epoch_save_dir: object = None,
        resume_from: object = None,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        warm_start
            Keep the current ``self.model`` weights (loaded via
            ``FullPipeline.load`` from a prior run). Adam moments are
            lost — pair with a smaller ``lr`` (2e-4 range).
        epoch_save_dir
            If set (``pathlib.Path`` or ``str``), after every epoch write
            ``latest.pt`` + ``epoch_XXX.pt`` under that directory containing
            model state, optimizer state, current epoch, and best-so-far
            metric. Colab-disconnect insurance — resume via ``resume_from``.
            The FullPipeline pickle (model + preprocessor + schema) is
            handled separately; save it once at the start.
        resume_from
            If set, load a mid-training checkpoint produced by
            ``epoch_save_dir`` and continue from the next epoch. Model
            architecture must match. Overrides ``warm_start`` — the
            checkpoint's state is authoritative.
        """
        torch.manual_seed(self.seed)

        # For memmap-backed X we must stream per-batch (on_batch=True);
        # eager presort would materialise the whole file. For regular
        # numpy arrays we keep the fast presort path.
        _train_on_batch = isinstance(train.X, np.memmap)
        train_ds = DateBatchDataset(
            train.X, train.resp, train.y, train.w,
            train.symbols, train.dates, train.times,
            n_times=self.n_times, on_batch=_train_on_batch,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=flatten_collate_fn,
            num_workers=0,
        )
        val_dl = None
        if valid is not None:
            val_ds = DateBatchDataset(
                valid.X, valid.resp, valid.y, valid.w,
                valid.symbols, valid.dates, valid.times,
                n_times=self.n_times, on_batch=True,
            )
            val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=flatten_collate_fn)

        if warm_start and self.model is not None:
            # Reuse loaded weights; only re-materialise on the target device
            # in case we're switching Mac CPU → Colab GPU.
            self.model = self.model.to(self.device)
        else:
            self.model = self._build(train.X.shape[1]).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_r2 = -np.inf
        best_state: dict | None = None
        no_improve = 0
        start_epoch = 0

        # Optional: resume from a mid-training checkpoint. This restores
        # model + optimizer + best-so-far + epoch counter — losing zero
        # progress from a Colab disconnect (unlike warm_start which only
        # brings the weights over and loses Adam moments).
        if resume_from is not None:
            ckpt = torch.load(Path(resume_from), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_r2 = float(ckpt["best_r2"])
            best_state = ckpt.get("best_state")
            self.best_epoch = int(ckpt.get("best_epoch", start_epoch))
            if verbose:
                print(f"[recurrent] resumed at epoch {start_epoch} (best so far R²={best_r2:.4f})")

        if epoch_save_dir is not None:
            save_dir = Path(epoch_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = None

        if verbose:
            print(f"[recurrent] device={self.device} | input={train.X.shape[1]}")
            print(f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>9} | {'train_R2':>8} | {'val_R2':>7}")

        for epoch in range(start_epoch, self.epochs):
            tr_loss, tr_r2 = self._train_one_epoch(train_dl)
            v_loss, v_r2 = (None, None)
            if val_dl is not None:
                v_loss, v_r2 = self._validate_one_epoch(val_dl)

            if verbose:
                print(
                    f"{epoch+1:>5} | {tr_loss:>10.4f} | "
                    f"{(v_loss if v_loss is not None else float('nan')):>9.4f} | "
                    f"{tr_r2:>8.4f} | {(v_r2 if v_r2 is not None else float('nan')):>7.4f}"
                )

            v_metric = v_r2 if v_r2 is not None else tr_r2
            if v_metric > best_r2:
                best_r2 = v_metric
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                self.best_epoch = epoch + 1
            else:
                no_improve += 1

            # Colab-disconnect insurance: dump full training state after
            # every epoch. Latest.pt is always the freshest; epoch_XXX.pt
            # is a rolling history (useful if you want to revert).
            if save_dir is not None:
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_r2": best_r2,
                    "best_epoch": getattr(self, "best_epoch", epoch + 1),
                    "best_state": best_state,
                }
                torch.save(ckpt, save_dir / "latest.pt")
                torch.save(ckpt, save_dir / f"epoch_{epoch + 1:03d}.pt")

            if self.early_stopping and no_improve >= self.early_stopping_patience + 1:
                if verbose:
                    print(f"early stop at epoch {epoch+1} (best={self.best_epoch}, R²={best_r2:.4f})")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    def _forward(self, x: torch.Tensor, hidden=None):
        out = self.model(x, hidden)
        if self.aux_branches:
            y, aux, hidden = out
            return y, aux, hidden
        y, hidden = out
        return y, None, hidden

    def _loss_for_batch(
        self,
        x: torch.Tensor,
        resp: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_pred, aux, _ = self._forward(x, None)
        loss = self.criterion(y_pred.flatten(), y.flatten(), w.flatten())
        if self.aux_branches and aux is not None and resp.shape[-1] >= self.num_aux:
            # Volkova uses the last ``num_aux`` columns of the responder matrix
            # — in column order these correspond to (r7, r8, r9, r10) with our
            # ordering (see FitData construction in the pipeline).
            for i in range(self.num_aux):
                loss = loss + self.criterion(
                    aux[:, :, i].flatten(),
                    resp[:, :, -(i + 1)].flatten(),
                    w.flatten(),
                )
        return loss, y_pred

    def _train_one_epoch(self, dl: DataLoader) -> tuple[float, float]:
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        total_loss = 0.0
        ys, ws, ps = [], [], []
        for x, resp, y, w in dl:
            x, resp, y, w = [t.to(self.device) for t in (x, resp, y, w)]
            self.optimizer.zero_grad()
            loss, y_pred = self._loss_for_batch(x, resp, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            ys.append(y.detach().flatten().cpu())
            ws.append(w.detach().flatten().cpu())
            ps.append(y_pred.detach().flatten().cpu())
        y_all = torch.cat(ys); w_all = torch.cat(ws); p_all = torch.cat(ps)
        return total_loss / max(len(dl), 1), float(r2_weighted_torch(y_all, p_all, w_all))

    def _validate_one_epoch(self, dl: DataLoader) -> tuple[float, float]:
        assert self.model is not None
        # We mimic Volkova's "validate-with-update" by performing one refit
        # step per date if ``lr_refit > 0``. This is closer to the production
        # online protocol than a frozen validation pass.
        model = copy.deepcopy(self.model)
        model.train(False)
        losses, ys, ws, ps = [], [], [], []
        for x, resp, y, w in dl:
            x, resp, y, w = [t.to(self.device) for t in (x, resp, y, w)]
            model.eval()
            with torch.no_grad():
                pred, _, _ = self._forward_for_model(model, x, None)
                losses.append(self.criterion(pred.flatten(), y.flatten(), w.flatten()).item())
                ys.append(y.flatten().cpu()); ws.append(w.flatten().cpu()); ps.append(pred.flatten().cpu())
            if self.lr_refit > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=self.lr_refit, weight_decay=self.weight_decay)
                opt.zero_grad()
                model.train()
                pred, _, _ = self._forward_for_model(model, x, None)
                loss = self.criterion(pred, y, w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                opt.step()
        y_all = torch.cat(ys); w_all = torch.cat(ws); p_all = torch.cat(ps)
        return float(np.mean(losses)), float(r2_weighted_torch(y_all, p_all, w_all))

    def _forward_for_model(self, model: nn.Module, x: torch.Tensor, hidden):
        out = model(x, hidden)
        if self.aux_branches:
            return out
        y, h = out
        return y, None, h

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        n_times: int | None = None,
        state: object | None = None,
    ) -> tuple[np.ndarray, object | None]:
        assert self.model is not None
        n_times = n_times if n_times is not None else self.n_times
        X_t = torch.from_numpy(X).float()
        X_t = reshape_flat_to_sequence(X_t, n_times).to(self.device)
        X_t = torch.nan_to_num(X_t, 0.0)
        self.model.eval()
        with torch.no_grad():
            y, _, hidden = self._forward(X_t, state)  # type: ignore[arg-type]
        return reshape_sequence_to_flat(y).cpu().numpy(), hidden

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
        y_pred, _, _ = self._forward(X_t, None)
        loss = self.criterion(y_pred.flatten(), y_t.flatten(), w_t.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        opt.step()

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict() if self.model is not None else None}

    def load_state_dict(self, state: dict) -> None:
        if self.model is None or state.get("model") is None:
            return
        self.model.load_state_dict(state["model"])
