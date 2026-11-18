import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Regressor(nn.Module):
    """DINOv2 drop-in for SimRegMatch's prediction/representation contract."""
    model_name = "dinov2_vits14"
    feature_dim = 384
    pretrained_identifier = "DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)"

    def __init__(self, dropout=0.1):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.dropout = float(dropout)
        self.linear = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        encoding = self.backbone(x)
        prediction = self.linear(F.dropout(encoding, p=self.dropout, training=True))
        return prediction, encoding


def dinov2_vits14(dropout=0.1):
    return DINOv2Regressor(dropout=dropout)
