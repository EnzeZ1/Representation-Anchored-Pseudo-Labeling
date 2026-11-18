"""Backbone adapter exposing DINOv2 through official HPL's model contract."""

from __future__ import annotations

import torch
import torch.nn as nn


MODEL_NAME = "dinov2_vits14"
WEIGHT_IDENTIFIER = "DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
FEATURE_DIMENSION = 384


class DINOv2HPLRegressor(nn.Module):
    """DINOv2 ViT-S/14 plus HPL's existing scalar regression-head contract."""

    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.fc_m = nn.Linear(FEATURE_DIMENSION, 1)

    def forward(self, value):
        features = self.backbone(value)
        prediction = self.fc_m(self.dropout(features))
        return prediction, features
