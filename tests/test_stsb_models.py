import torch

from models.stsb_bilstm import BiLSTMPairRegressor


def test_bilstm_feature_api():
    embeddings = torch.randn(20, 8)
    model = BiLSTMPairRegressor(
        embeddings, padding_index=0, hidden_size=4, num_layers=1, dropout=0
    )
    first = torch.tensor([[1, 2, 0], [3, 4, 5]])
    second = torch.tensor([[6, 0], [7, 8]])
    features = model.forward_features(first, second, first.ne(0), second.ne(0))
    assert features.shape == (2, 32)
    assert model(first, second, first.ne(0), second.ne(0)).shape == (2,)
