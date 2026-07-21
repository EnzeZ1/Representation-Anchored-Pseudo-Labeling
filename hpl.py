"""
Heteroscedastic Pseudo-Labels (HPL) — matching official implementation.

Uses `higher` for proper Adam unrolling in the bilevel meta-loop.
Separate optimizers for backbone, head, and uncertainty learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher


class UncertaintyLearner(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


@dataclass
class HPLConfig:
    w_ulb: float = 10.0
    lambda2: float = 0.1
    update_unc_every: int = 5


class HPLTrainer:

    def __init__(self, model, uncertainty, opt_feat, opt_fc, opt_unc, cfg=HPLConfig()):
        self.model = model
        self.uncertainty = uncertainty
        self.opt_feat = opt_feat
        self.opt_fc = opt_fc
        self.opt_unc = opt_unc
        self.cfg = cfg
        self.step_id = 0

    def step(self, x_l, y_l, x_u_w, x_u_s, x_meta, y_meta) -> Dict[str, float]:
        self.model.train()
        self.uncertainty.train()

        meta_loss_val = float('nan')
        if self.step_id % self.cfg.update_unc_every == 0:
            meta_loss_val = self._update_uncertainty(
                x_l, y_l, x_u_w, x_u_s, x_meta, y_meta)

        train_loss = self._update_model(x_l, y_l, x_u_w, x_u_s)
        self.step_id += 1

        return {
            "hpl_train_loss": float(train_loss),
            "hpl_meta_loss": meta_loss_val,
        }

    def _update_model(self, x_l, y_l, x_u_w, x_u_s):
        self.opt_fc.zero_grad(set_to_none=True)
        self.opt_feat.zero_grad(set_to_none=True)

        pred_l = self.model(x_l)
        loss_lb = F.mse_loss(pred_l, y_l)

        with torch.no_grad():
            pred_w = self.model(x_u_w).detach()
        pred_s = self.model(x_u_s)

        with torch.no_grad():
            unc_in = torch.stack([pred_s.detach() - pred_w,
                                  pred_s.detach()], dim=-1)
            weight = torch.exp(-self.uncertainty(unc_in)) / 2

        loss_ulb = torch.mean(weight.squeeze(-1) * (pred_s - pred_w) ** 2)
        loss = loss_lb + self.cfg.w_ulb * loss_ulb

        loss.backward()
        self.opt_fc.step()
        self.opt_feat.step()

        return loss.detach()

    def _update_uncertainty(self, x_l, y_l, x_u_w, x_u_s, x_meta, y_meta):
        self.opt_unc.zero_grad(set_to_none=True)

        with higher.innerloop_ctx(self.model.head, self.opt_fc) as (fhead, diffopt):
            # Freeze backbone, extract features with dropout
            with torch.no_grad():
                feat_l = self.model.drop(self.model.encode(x_l))
                feat_uw = self.model.drop(self.model.encode(x_u_w))
                feat_us = self.model.drop(self.model.encode(x_u_s))
                feat_meta = self.model.drop(self.model.encode(x_meta))

            # Inner supervised
            pred_l = fhead(feat_l).squeeze(-1)
            loss_lb = F.mse_loss(pred_l, y_l)

            # Inner pseudo-label
            with torch.no_grad():
                pred_w = fhead(feat_uw)
            pred_s = fhead(feat_us)

            # Uncertainty weight (inputs detached, unc_learner params live)
            unc_in = torch.cat([pred_s.detach() - pred_w,
                                pred_s.detach()], dim=-1)
            weight_raw = self.uncertainty(unc_in.detach())
            weight = torch.exp(-weight_raw) / 2

            unlabel_mse = (pred_s - pred_w) ** 2
            loss_ulb = torch.mean(weight * unlabel_mse)

            inner_loss = loss_lb + self.cfg.w_ulb * loss_ulb

            # Simulate Adam step on head via higher
            diffopt.step(inner_loss)

            # Meta evaluation with updated head
            pred_meta = fhead(feat_meta).squeeze(-1)

            # λ2 regularization on META batch uncertainty
            unc_in_meta = torch.cat([
                (pred_meta.unsqueeze(-1) - y_meta.unsqueeze(-1)).detach(),
                pred_meta.unsqueeze(-1).detach()
            ], dim=-1)
            weight_meta = self.uncertainty(unc_in_meta.detach())

            meta_loss = F.mse_loss(pred_meta, y_meta) \
                        - self.cfg.lambda2 * torch.mean(weight_meta)

            self.opt_unc.zero_grad(set_to_none=True)
            meta_loss.backward()
            self.opt_unc.step()

        return float(meta_loss.detach())