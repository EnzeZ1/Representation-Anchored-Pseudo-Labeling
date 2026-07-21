"""
STS-B (Semantic Textual Similarity Benchmark) support.
Sentence pairs → similarity score (0-5, continuous regression).
Uses sentence-transformers as frozen backbone.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class STSBDataset(Dataset):
    """STS-B dataset: sentence pairs with similarity scores."""
    def __init__(self, encodings, labels=None, unlabeled=False):
        self.encodings = encodings  # [N, D] precomputed features
        self.labels = labels
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        feat = self.encodings[i]
        if self.unlabeled:
            # Return (weak, strong) — for text, "strong" = dropout noise
            noise = torch.randn_like(feat) * 0.1
            return feat, feat + noise
        return feat, self.labels[i]


class TextRegressor(nn.Module):
    """MLP regressor on precomputed text features.
    Exposes backbone/head/drop/encode matching ResNet50Regressor interface."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.feature_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        feat = self.drop(self.encode(x))
        return self.head(feat).squeeze(-1)


def encode_stsb(split_data, model_name='all-MiniLM-L6-v2'):
    """Encode sentence pairs using sentence-transformer."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    sent1 = [item['sentence1'] for item in split_data]
    sent2 = [item['sentence2'] for item in split_data]
    labels = [item['label'] for item in split_data]

    print(f'  Encoding {len(sent1)} sentence pairs...')
    emb1 = model.encode(sent1, batch_size=256, show_progress_bar=False,
                        convert_to_tensor=True)
    emb2 = model.encode(sent2, batch_size=256, show_progress_bar=False,
                        convert_to_tensor=True)

    # Combine: [emb1, emb2, |emb1-emb2|, emb1*emb2]
    features = torch.cat([emb1, emb2, (emb1 - emb2).abs(), emb1 * emb2], dim=1)

    labels = torch.tensor(labels, dtype=torch.float32)
    return features.cpu(), labels


def make_data_stsb(args):
    """Load STS-B and create dataloaders."""
    from datasets import load_dataset

    print('Loading STS-B dataset...')
    dataset = load_dataset('glue', 'stsb')

    train_data = [x for x in dataset['train'] if x['label'] >= 0]
    # GLUE test labels are hidden, use validation as test
    test_data = [x for x in dataset['validation'] if x['label'] >= 0]

    print(f'  Train: {len(train_data)}, Test (from val): {len(test_data)}')

    # Encode all splits
    print('Encoding with sentence-transformer...')
    feat_train, labels_train = encode_stsb(train_data)
    feat_test, labels_test = encode_stsb(test_data)

    feat_dim = feat_train.shape[1]
    print(f'  Feature dim: {feat_dim}')

    # Normalize labels
    y_mean = float(labels_train.mean())
    y_std = float(labels_train.std() + 1e-6)
    labels_train = (labels_train - y_mean) / y_std
    labels_test = (labels_test - y_mean) / y_std

    # Split train into labeled/unlabeled/val
    n = len(feat_train)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = int(0.1 * n)
    n_lab = max(1, int(args.labeled_ratio * (n - n_val)))

    val_idx = idx[:n_val]
    lab_idx = idx[n_val:n_val + n_lab]
    unlab_idx = idx[n_val + n_lab:]

    lab_ds = STSBDataset(feat_train[lab_idx], labels_train[lab_idx])
    unlab_ds = STSBDataset(feat_train[unlab_idx], unlabeled=True)
    val_ds = STSBDataset(feat_train[val_idx], labels_train[val_idx])
    test_ds = STSBDataset(feat_test, labels_test)

    # Store feat_dim as attribute on lab_ds for retrieval
    lab_ds.feat_dim = feat_dim

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                         num_workers=0, pin_memory=True, drop_last=drop)

    print(f'STS-B: labeled={len(lab_idx)}, unlabeled={len(unlab_idx)}, '
          f'val={len(val_idx)}, test={len(feat_test)}')
    print(f'label scaler: mean={y_mean:.2f}, std={y_std:.2f}')

    return loader(lab_ds), loader(unlab_ds), loader(val_ds, False, False), loader(test_ds, False, False), y_mean, y_std