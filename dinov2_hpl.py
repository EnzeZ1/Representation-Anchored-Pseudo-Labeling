"""DINOv2 model matching HPL's expected interface."""

import torch
import torch.nn as nn


class DINOv2_unc(nn.Module):
    def __init__(self, size='small', drp_p=0.05):
        super().__init__()
        models = {
            'small': ('dinov2_vits14', 384),
            'base': ('dinov2_vitb14', 768),
        }
        model_name, feat_dim = models[size]
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.fc_m = nn.Linear(feat_dim, 1)
        self.fc_v = nn.Linear(feat_dim, 1)
        self.drop = nn.Dropout(p=drp_p)

    def forward(self, x):
        feat = self.backbone(x)
        x_feat_m = self.drop(feat)
        x_m = self.fc_m(x_feat_m)
        return x_m, x_feat_m


def dinov2_unc(pretrained=True, progress=True, **kwargs):
    return DINOv2_unc(**kwargs)