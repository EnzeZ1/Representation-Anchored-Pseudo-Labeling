"""Reusable generic RoBERTa-base sentence-pair regression backbone."""

from __future__ import annotations

import torch
from torch import nn

ROBERTA_BASE_IDENTIFIER = "FacebookAI/roberta-base"


class RobertaPairRegressor(nn.Module):
    def __init__(self, pretrained_identifier: str = ROBERTA_BASE_IDENTIFIER, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.pretrained_identifier = pretrained_identifier
        self.backbone = AutoModel.from_pretrained(pretrained_identifier)
        self.feature_dim = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.feature_dim, 1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0]

    def forward_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(input_ids, attention_mask))).squeeze(-1)
