"""Official DINOv3 ConvNeXt-Tiny regression backbone adapter."""
import torch.nn as nn
from transformers import AutoModel

MODEL_IDENTIFIER = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
MODEL_REVISION = "10d30274b4d445111e2d5bf75ac93bbd94db274b"
WEIGHT_SHA256 = "bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"
FEATURE_DIMENSION = 768

class DINOv3ConvNextTinyFeatureBackbone(nn.Module):
    """Return only the official final normalized, globally pooled representation."""
    def __init__(self):
        super().__init__()
        self.model=AutoModel.from_pretrained(MODEL_IDENTIFIER,revision=MODEL_REVISION,local_files_only=True)
    def forward(self,value):
        return self.model(pixel_values=value).pooler_output

class DINOv3ConvNextTinyRegressor(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.model_name=self.model_identifier=MODEL_IDENTIFIER
        self.weight_identifier=MODEL_IDENTIFIER
        self.weight_revision=MODEL_REVISION
        self.weight_checksum_sha256=WEIGHT_SHA256
        self.feature_dim=FEATURE_DIMENSION
        self.backbone=DINOv3ConvNextTinyFeatureBackbone()
        self.drop=nn.Dropout(dropout)
        self.head=nn.Linear(FEATURE_DIMENSION,1)
    def encode(self,value):
        return self.backbone(value)
    def forward(self,value):
        return self.head(self.drop(self.encode(value))).squeeze(-1)
