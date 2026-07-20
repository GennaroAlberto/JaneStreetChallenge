"""Supervised autoencoder + MLP — the family that won Jane Street 2021.

Yirun Zhang's 1st-place solution to the previous Jane Street competition
trained, end to end: a denoising autoencoder over the features, with the
bottleneck supervised by the target, and an MLP consuming [raw features,
bottleneck] for the final prediction. Our feature-level cousin of this
(bottleneck extracted, fed to XGB) is the campaign's best confirmed block
(+0.00058); this module is the full end-to-end version, with this
competition's aux responders as additional supervision.

Loss = WeightedR2(target head)
     + aux_weight   * mean_i WeightedR2(aux head_i)
     + recon_weight * MSE(decoder(z), x)

Row-wise (``sequence_model = False``) — temporal context must arrive via
the engineered features. That makes it an ensemble-diversity candidate by
construction: its errors cannot share the RNNs' temporal machinery.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from janestreet.models.base import BaseModel, FitData
from janestreet.theory.torch_utils import auto_device
from janestreet.training.loss import WeightedR2Loss
from janestreet.training.metrics import r2_weighted_torch


class AEMLPNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent: int = 16,
        enc_hidden: int = 128,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.2,
        num_aux: int = 4,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        mlp_hidden = mlp_hidden if mlp_hidden is not None else [256, 256, 128]
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_size, enc_hidden), nn.GELU(),
            nn.Linear(enc_hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, enc_hidden), nn.GELU(),
            nn.Linear(enc_hidden, input_size),
        )
        dims = [input_size + latent, *mlp_hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(),
                           nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.head_y = nn.Linear(dims[-1], 1)
        self.head_aux = nn.Linear(dims[-1], num_aux) if num_aux > 0 else None

    def forward(self, x: torch.Tensor, add_noise: bool = False):
        xin = x + self.noise_std * torch.randn_like(x) if (
            add_noise and self.noise_std > 0) else x
        z = self.encoder(xin)
        recon = self.decoder(z)
        h = self.trunk(torch.cat([x, z], dim=-1))
        y = self.head_y(h).squeeze(-1)
        aux = self.head_aux(h) if self.head_aux is not None else None
        return y, aux, recon


class AEMLPModel(BaseModel):
    sequence_model = False

    def __init__(
        self,
        latent: int = 16,
        enc_hidden: int = 128,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.2,
        noise_std: float = 0.1,
        num_aux: int = 4,
        aux_weight: float = 1.0,
        recon_weight: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        epochs: int = 20,
        batch_size: int = 8192,
        early_stopping_patience: int = 3,
        lr_refit: float = 0.0,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.latent = latent
        self.enc_hidden = enc_hidden
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout
        self.noise_std = noise_std
        self.num_aux = num_aux
        self.aux_weight = aux_weight
        self.recon_weight = recon_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.lr_refit = lr_refit
        self.seed = seed
        self.device = auto_device(device)
        self.criterion = WeightedR2Loss()
        self.model: AEMLPNet | None = None

    # ------------------------------------------------------------------
    def _loss(self, x, y, w, resp):
        pred, aux, recon = self.model(x, add_noise=self.model.training)
        loss = self.criterion(pred, y, w)
        if aux is not None and resp is not None and resp.shape[-1] >= self.num_aux:
            tgt = resp[:, -self.num_aux:]
            for i in range(self.num_aux):
                loss = loss + self.aux_weight / self.num_aux * self.criterion(
                    aux[:, i], tgt[:, i], w)
        loss = loss + self.recon_weight * nn.functional.mse_loss(recon, x)
        return loss

    def fit(self, train: FitData, valid: FitData | None = None,
            verbose: bool = False) -> None:
        torch.manual_seed(self.seed)
        self.model = AEMLPNet(
            train.X.shape[1], self.latent, self.enc_hidden, self.mlp_hidden,
            self.dropout, self.num_aux, self.noise_std,
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        X_t = torch.from_numpy(train.X).float()
        y_t = torch.from_numpy(train.y).float()
        w_t = torch.from_numpy(train.w).float()
        r_t = torch.from_numpy(train.resp).float()
        n = X_t.shape[0]

        # valid stays on CPU; the epoch-end eval streams chunks through the
        # device — a single full-tail forward pass allocates multi-GB
        # activations and OOMs the MPS shared pool
        Xv = yv = wv = None
        if valid is not None:
            Xv = torch.from_numpy(valid.X).float()
            yv = torch.from_numpy(valid.y).float()
            wv = torch.from_numpy(valid.w).float()

        best_r2, best_state, no_improve = -np.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            idx = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                sl = idx[i:i + self.batch_size]
                opt.zero_grad()
                loss = self._loss(X_t[sl].to(self.device), y_t[sl].to(self.device),
                                  w_t[sl].to(self.device), r_t[sl].to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(sl)

            v_r2 = float("nan")
            if Xv is not None:
                self.model.eval()
                num = den = 0.0
                with torch.no_grad():
                    for j in range(0, len(Xv), 262144):
                        pb, _, _ = self.model(Xv[j:j + 262144].to(self.device))
                        pb = pb.cpu()
                        yb, wb = yv[j:j + 262144], wv[j:j + 262144]
                        num += float((wb * (pb - yb) ** 2).sum())
                        den += float((wb * yb ** 2).sum())
                v_r2 = 1.0 - num / (den + 1e-38)
            if self.device == "mps":
                torch.mps.empty_cache()
            if verbose:
                print(f"epoch {epoch + 1:>3}: loss={tot / n:.4f}  val_R2={v_r2:+.4f}",
                      flush=True)
            if v_r2 > best_r2:
                best_r2, no_improve = v_r2, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                no_improve += 1
            if no_improve >= self.early_stopping_patience + 1:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray, n_times: int | None = None,
                state: object | None = None):
        assert self.model is not None
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 262144):
                xb = torch.from_numpy(X[i:i + 262144]).float().to(self.device)
                p, _, _ = self.model(xb)
                outs.append(p.cpu().numpy())
        return np.concatenate(outs), None

    def update(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
               n_times: int) -> None:
        if self.lr_refit <= 0 or self.model is None:
            return
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr_refit,
                                weight_decay=self.weight_decay)
        self.model.train()
        opt.zero_grad()
        x_t = torch.from_numpy(X).float().to(self.device)
        pred, _, _ = self.model(x_t)
        loss = self.criterion(pred, torch.from_numpy(y).float().to(self.device),
                              torch.from_numpy(w).float().to(self.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
