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
    weighting_intervention: str = "rapl"
    formal_seed: int = 0
    trust_tau: float = 1.0


def train_probe(backbone, labeled_loader, device, progress_callback=None):
    """Train linear probe on pretrained features. Closed-form, instant."""
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in labeled_loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    if progress_callback:
        progress_callback('frozen-probe feature extraction completed')
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
    if progress_callback:
        progress_callback('probe fitting completed')

    return probe


class ProbeFilteredTrainer:
    """Consistency training filtered by a frozen feature probe."""

    def __init__(self, model, frozen_backbone, probe, optimizer,
                 cfg=ProbeConfig(), progress_callback=None):
        self.model = model
        self.frozen_backbone = frozen_backbone
        self.probe = probe
        self.optimizer = optimizer
        self.cfg = cfg
        self.progress_callback = progress_callback
        self._first_step = True
        self._training_step = 0

        self.frozen_backbone.eval()
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)

    def step(self, x_l, y_l, x_u_w, x_u_s) -> Dict[str, float]:
        # Supervised loss
        if x_l is None:
            loss_sup = torch.zeros((), device=x_u_w.device)
        else:
            pred_l = self.model(x_l)
            if self._first_step and self.progress_callback:
                self.progress_callback('target forward completed')
            loss_sup = F.mse_loss(pred_l, y_l)

        # Pseudo-label from model (weak view)
        with torch.no_grad():
            pseudo = self.model(x_u_w).detach()

        pred_s = None
        if self.cfg.weighting_intervention == "rapl_self_referential_weight":
            # The model's weak/strong disagreement replaces only the independent
            # anchor used to estimate reliability. Both sides of this reliability
            # branch are detached; the strong prediction used by the consistency
            # residual below remains live for gradient computation.
            pred_s = self.model(x_u_s)
            disagreement = (pseudo - pred_s.detach()).abs()
            r = (1.0 / (1.0 + disagreement)).detach()
        else:
            # Probe's second opinion (frozen backbone, weak view)
            with torch.no_grad():
                z_u = self.frozen_backbone(x_u_w)
                probe_est = self.probe(z_u).squeeze(-1)
            if self._first_step and self.progress_callback:
                self.progress_callback('frozen-probe forward completed')
            disagreement = (probe_est - pseudo).abs()
            r = (self.cfg.trust_tau / (self.cfg.trust_tau + disagreement)).detach()

        if self.cfg.weighting_intervention == "rapl_uniform_weight":
            r = torch.ones_like(r)
        elif self.cfg.weighting_intervention == "rapl_self_referential_weight":
            pass
        elif self.cfg.weighting_intervention == "rapl_constant_weight":
            r = r.mean().expand_as(r)
        elif self.cfg.weighting_intervention == "rapl_shuffled_weight":
            # A private CPU generator prevents the permutation from consuming or
            # perturbing model/data/augmentation RNG streams.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.cfg.formal_seed * 1_000_003 + self._training_step)
            permutation = torch.randperm(r.numel(), generator=generator).to(r.device)
            r = r.flatten()[permutation].reshape_as(r)
        elif self.cfg.weighting_intervention != "rapl":
            raise ValueError(f"Unknown weighting intervention: {self.cfg.weighting_intervention}")

        # Gated consistency loss (strong view)
        if pred_s is None:
            pred_s = self.model(x_u_s)
        loss_u = (r * (pred_s - pseudo).pow(2)).mean()

        loss = loss_sup + self.cfg.lambda_u * loss_u
        if self._first_step and self.progress_callback:
            self.progress_callback('loss construction completed')

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._first_step and self.progress_callback:
            self.progress_callback('backward completed')
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self._training_step += 1
        if self._first_step and self.progress_callback:
            self.progress_callback('optimizer step completed')
            self._first_step = False

        return {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_u": float(loss_u.detach()),
            "mean_r": float(r.mean()),
            "mean_disagreement": float(disagreement.mean()),
            "grad_norm": float(grad_norm),
        }
