"""Exact DINOv2 regressor used by the gpu03 canonical RSNA Uniform campaign."""
import torch
import torch.nn as nn
from pathlib import Path


class DINOv2Regressor(nn.Module):
    MODELS = {'small': ('dinov2_vits14', 384), 'base': ('dinov2_vitb14', 768), 'large': ('dinov2_vitl14', 1024)}
    HUB_REPOSITORY = 'facebookresearch/dinov2'
    WEIGHT_IDENTIFIERS = {
        'small': 'DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)',
        'base': 'DINOv2 ViT-B/14 LVD-142M (dinov2_vitb14)',
        'large': 'DINOv2 ViT-L/14 LVD-142M (dinov2_vitl14)',
    }
    WEIGHT_URLS = {
        'small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
    }

    def __init__(self, size='small', dropout=0.2):
        super().__init__()
        model_name, feat_dim = self.MODELS[size]
        self.feature_dim, self.size, self.model_name = feat_dim, size, model_name
        self.weight_identifier = self.WEIGHT_IDENTIFIERS[size]
        self.weight_url = self.WEIGHT_URLS[size]
        cached_repository = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
        if cached_repository.is_dir():
            self.backbone = torch.hub.load(str(cached_repository), model_name, source='local')
        else:
            self.backbone = torch.hub.load(self.HUB_REPOSITORY, model_name)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1)

    def encode(self, x): return self.backbone(x)
    def forward(self, x): return self.head(self.drop(self.encode(x))).squeeze(-1)
