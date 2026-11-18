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


def train_probe(backbone, labeled_loader, device, progress_callback=None):
    """Train linear probe on pretrained features. Closed-form, instant."""
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in labeled_loader:
            # Recycling Pilot B returns (clean, strong); the canonical frozen
            # probe consumes the deterministic clean branch only.
            if isinstance(x, (tuple, list)):
                x = x[0]
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
                 cfg=ProbeConfig(), progress_callback=None, uniform_weight=False):
        self.model = model
        self.frozen_backbone = frozen_backbone
        self.probe = probe
        self.optimizer = optimizer
        self.cfg = cfg
        self.progress_callback = progress_callback
        self.uniform_weight = uniform_weight
        self._first_step = True

        self.frozen_backbone.eval()
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)

    def step(self, x_l, y_l, x_u_w, x_u_s, *, x_l_strong=None,
             labeled_anchor_mode='canonical') -> Dict[str, float]:
        # Explicit labeled anchor. In clean+strong mode x_l and x_l_strong are
        # two views of the same recycled labeled batch at this optimizer step.
        pred_l = self.model(x_l)
        if self._first_step and self.progress_callback:
            self.progress_callback('target forward completed')
        loss_clean_sup = F.mse_loss(pred_l, y_l)
        loss_strong_sup = None
        if labeled_anchor_mode == 'recycled_strong_anchor':
            loss_sup = loss_clean_sup  # x_l is the freshly sampled strong view
            loss_clean_sup = None
            loss_strong_sup = loss_sup
        elif labeled_anchor_mode == 'recycled_clean_strong_anchor':
            if x_l_strong is None:
                raise ValueError('clean+strong labeled anchor requires x_l_strong')
            loss_strong_sup = F.mse_loss(self.model(x_l_strong), y_l)
            loss_sup = loss_clean_sup + loss_strong_sup
        elif labeled_anchor_mode == 'canonical':
            loss_sup = loss_clean_sup
        else:
            raise ValueError(f'Unknown labeled_anchor_mode: {labeled_anchor_mode}')

        # Pseudo-label from model (weak view)
        with torch.no_grad():
            pseudo = self.model(x_u_w).detach()

        # Probe's second opinion (frozen backbone, weak view)
        with torch.no_grad():
            z_u = self.frozen_backbone(x_u_w)
            probe_est = self.probe(z_u).squeeze(-1)
        if self._first_step and self.progress_callback:
            self.progress_callback('frozen-probe forward completed')

        # Trust score
        disagreement = (probe_est - pseudo).abs()
        r = (1.0 / (1.0 + disagreement)).detach()
        if self.uniform_weight:
            # Formal Uniform control: probe machinery may execute for path matching,
            # but no uncertainty or reliability can affect optimization.
            r = torch.ones_like(r)

        # Gated consistency loss (strong view)
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
        if self._first_step and self.progress_callback:
            self.progress_callback('optimizer step completed')
            self._first_step = False

        return {
            "loss_total": float(loss.detach()),
            "loss_sup": float(loss_sup.detach()),
            "loss_clean_sup": (0.0 if loss_clean_sup is None else float(loss_clean_sup.detach())),
            "loss_strong_sup": (0.0 if loss_strong_sup is None else float(loss_strong_sup.detach())),
            "loss_u": float(loss_u.detach()),
            "mean_r": float(r.mean()),
            "mean_disagreement": float(disagreement.mean()),
            "mean_prediction_gap": float((pred_s - pseudo).abs().mean().detach()),
            "grad_norm": float(grad_norm),
        }
