"""HPL-compatible BiLSTM/GloVe sentence-pair regressor."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMPairRegressor(nn.Module):
    """Two-layer BiLSTM with max pooling and InferSent-style pair features."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        padding_index: int,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.2,
        train_embeddings: bool = False,
    ):
        super().__init__()
        self.padding_index = int(padding_index)
        self.hidden_size = int(hidden_size)
        self.feature_dim = 8 * self.hidden_size
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.float(), freeze=not train_embeddings, padding_idx=self.padding_index
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            input_size=embeddings.shape[1],
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.feature_dim, 1)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))
        encoded, _ = self.encoder(embedded)
        encoded = self.dropout(encoded)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), torch.finfo(encoded.dtype).min)
        return encoded.max(dim=1).values

    def forward_features(
        self,
        sentence1: torch.Tensor,
        sentence2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        first = self.encode(sentence1, mask1)
        second = self.encode(sentence2, mask2)
        return torch.cat((first, second, (first - second).abs(), first * second), dim=-1)

    def forward(
        self,
        sentence1: torch.Tensor,
        sentence2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.forward_features(sentence1, sentence2, mask1, mask2)).squeeze(-1)
