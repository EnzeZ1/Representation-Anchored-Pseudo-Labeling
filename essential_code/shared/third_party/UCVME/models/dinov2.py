import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2UncertaintyRegressor(nn.Module):
    """DINOv2 ViT-S/14 implementation of UCVME's mean/variance contract."""
    model_name = "dinov2_vits14"
    feature_dim = 384
    pretrained_identifier = "DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)"

    def __init__(self, drp_p=0.05):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.fc_m = nn.Linear(self.feature_dim, 1)
        self.fc_v = nn.Linear(self.feature_dim, 1)
        self.drop_rate = float(drp_p)

    def forward(self, x):
        features = self.backbone(x)
        return (self.fc_m(F.dropout(features, p=self.drop_rate, training=True)),
                self.fc_v(F.dropout(features, p=self.drop_rate, training=True)))


def dinov2_vits14_unc(drp_p=0.05):
    return DINOv2UncertaintyRegressor(drp_p=drp_p)
