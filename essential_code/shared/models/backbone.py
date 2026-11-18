"""
Backbones for semi-supervised regression experiments.
- ResNet50Regressor: image regression backbone close to Heteroscedastic Pseudo-Labels.
- MixtureRegressor: distributional regression provider for the proposed method.

Expected input for image models: x of shape [B, 3, H, W].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:  # torchvision may be unavailable in some environments
    resnet50 = None
    ResNet50_Weights = None


@dataclass
class ModelOutput:
    pred: torch.Tensor
    feat: torch.Tensor


class ResNet50Regressor(nn.Module):
    """ResNet-50 + linear regression head.

    This mirrors the common image setup in the HPL paper:
    ImageNet-pretrained ResNet-50, feature dim 2048, final head 2048 -> 1.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if resnet50 is None:
            raise ImportError("torchvision is required for ResNet50Regressor")

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet50(weights=weights)
        self.feature_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        feat = self.drop(self.encode(x))
        pred = self.head(feat).squeeze(-1)
        if return_feat:
            return pred, feat
        return pred

    def head_forward(self, feat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(feat, weight, bias).squeeze(-1)


class MLPFeatureBackbone(nn.Module):
    """Small non-image fallback backbone for quick debugging."""

    def __init__(self, input_dim: int, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(nn.Module):
    """MLP regression model with the same interface as ResNet50Regressor."""

    def __init__(self, input_dim: int, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = MLPFeatureBackbone(input_dim, feature_dim)
        self.head = nn.Linear(feature_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        feat = self.encode(x)
        pred = self.head(feat).squeeze(-1)
        if return_feat:
            return pred, feat
        return pred

    def head_forward(self, feat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return F.linear(feat, weight, bias).squeeze(-1)


class MixtureRegressor(nn.Module):
    """Distributional pseudo-label provider / final regressor.

    For each x, outputs K Gaussian modes:
      weights a_k(x), means m_k(x), variances s_k^2(x).
    """

    def __init__(self, feature_backbone: nn.Module, feature_dim: int, num_modes: int = 3):
        super().__init__()
        self.backbone = feature_backbone
        self.feature_dim = feature_dim
        self.num_modes = num_modes
        self.logit_head = nn.Linear(feature_dim, num_modes)
        self.mean_head = nn.Linear(feature_dim, num_modes)
        self.logvar_head = nn.Linear(feature_dim, num_modes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encode(x)
        logits = self.logit_head(feat)
        weights = torch.softmax(logits, dim=-1)
        means = self.mean_head(feat)
        log_vars = self.logvar_head(feat).clamp(min=-8.0, max=6.0)
        return weights, means, log_vars, feat

    def point_prediction(self, x: torch.Tensor) -> torch.Tensor:
        weights, means, _, _ = self.forward(x)
        return (weights * means).sum(dim=-1)
