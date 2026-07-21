"""
Feature-Probe Filtered Consistency Training.

A frozen linear probe (trained on labeled data features) provides
an independent age estimate. When it disagrees with the model's
pseudo-label, the consistency loss is suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeConfig:
    lambda_u: float = 1.0
    max_grad_norm: float = 5.0


def train_probe(backbone, labeled_loader, device):
    """Train linear probe on pretrained features. Closed-form, instant."""
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in labeled_loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    n, d = feats.shape

    Z = torch.cat([feats, torch.ones(n, 1)], dim=1)
    sol = torch.linalg.lstsq(Z, labels.unsqueeze(1)).solution

    probe = nn.Linear(d, 1)
    probe.weight.data.copy_(sol[:-1, 0].unsqueeze(0))
    probe.bias.data.copy_(sol[-1:, 0])
    probe.requires_grad_(False)
    probe.eval()

    pred = feats @ sol[:-1, 0] + sol[-1, 0]
    mae = (pred - labels).abs().mean()
    print(f'Probe trained: MAE={mae:.3f} (normalized)')

    return probe


class ProbeFilteredTrainer:
    """Consistency training filtered by a frozen feature probe."""

    def __init__(self, model, frozen_backbone, probe, optimizer,
                 cfg=ProbeConfig()):
        self.model = model
        self.frozen_backbone = frozen_backbone
        self.probe = probe
        self.optimizer = optimizer
        self.cfg = cfg

        self.frozen_backbone.eval()
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)

    def step(self, x_l, y_l, x_u_w, x_u_s) -> Dict[str, float]:
        # Supervised loss
        pred_l = self.model(x_l)
        loss_sup = F.mse_loss(pred_l, y_l)

        # Pseudo-label from model (weak view)
        with torch.no_grad():
            pseudo = self.model(x_u_w).detach()

        # Probe's second opinion (frozen backbone, weak view)
        with torch.no_grad():
            z_u = self.frozen_backbone(x_u_w)
            probe_est = self.probe(z_u).squeeze(-1)

        # Trust score
        disagreement = (probe_est - pseudo).abs()
        r = (1.0 / (1.0 + disagreement)).detach()

        # Gated consistency loss (strong view)
        pred_s = self.model(x_u_s)
        loss_u = (r * (pred_s - pseudo).pow(2)).mean()

        loss = loss_sup + self.cfg.lambda_u * loss_u

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_u": float(loss_u.detach()),
            "mean_r": float(r.mean()),
            "mean_disagreement": float(disagreement.mean()),
            "grad_norm": float(grad_norm),
        }