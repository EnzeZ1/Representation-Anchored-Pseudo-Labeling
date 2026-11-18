"""Official DINOv3 ConvNeXt-Tiny adapter for the HPL model contract."""
import torch.nn as nn
from transformers import AutoModel
MODEL_IDENTIFIER="facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
MODEL_REVISION="10d30274b4d445111e2d5bf75ac93bbd94db274b"
WEIGHT_SHA256="bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2"
FEATURE_DIMENSION=768

class DINOv3ConvNextTinyHPLRegressor(nn.Module):
    def __init__(self,dropout=0.05):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(MODEL_IDENTIFIER,revision=MODEL_REVISION,local_files_only=True)
        self.dropout=nn.Dropout(dropout)
        self.fc_m=nn.Linear(FEATURE_DIMENSION,1)
    def forward(self,value):
        features=self.backbone(pixel_values=value).pooler_output
        return self.fc_m(self.dropout(features)),features
