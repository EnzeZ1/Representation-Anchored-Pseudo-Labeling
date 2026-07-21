"""
DINOv2 Regressor — drop-in replacement for ResNet50Regressor.
Same interface: backbone, head, drop, encode(), forward().
Usage: model = DINOv2Regressor(size='small')  # or 'base', 'large'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Regressor(nn.Module):
    """DINOv2 + linear regression head.
    Same interface as ResNet50Regressor."""

    MODELS = {
        'small': ('dinov2_vits14', 384),
        'base':  ('dinov2_vitb14', 768),
        'large': ('dinov2_vitl14', 1024),
    }

    def __init__(self, size='small', dropout=0.2):
        super().__init__()
        model_name, feat_dim = self.MODELS[size]
        self.feature_dim = feat_dim
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        feat = self.drop(self.encode(x))
        return self.head(feat).squeeze(-1)
