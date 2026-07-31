"""Shared manifest-backed weak/strong STS-B datasets for SSL methods."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from data_processing.text_augmentation import TextAugmenter, weak_view


class STSBUnlabeledViews(Dataset):
    """Return deterministic weak/strong pairs without exposing hidden labels."""

    def __init__(self, cohort, indices, formal_seed: int, augmenter: TextAugmenter | None = None):
        self.cohort = cohort
        self.indices = list(indices)
        self.formal_seed = int(formal_seed)
        self.epoch = 0
        self.augmenter = augmenter or TextAugmenter()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        record = self.cohort["records"][index]
        weak = weak_view(record["sentence1"], record["sentence2"])
        strong = self.augmenter.augment_pair(
            record["sentence1"], record["sentence2"], self.formal_seed,
            self.epoch, record["stable_id"],
        )
        return {
            "weak_sentence1": weak[0], "weak_sentence2": weak[1],
            "strong_sentence1": strong[0], "strong_sentence2": strong[1],
            "cohort_index": index, "stable_id": record["stable_id"],
        }


class SSLPairCollator:
    """Apply the same backbone tokenizer/collator independently to both views."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    @staticmethod
    def _view(examples, prefix):
        return [{
            "sentence1": example[f"{prefix}_sentence1"],
            "sentence2": example[f"{prefix}_sentence2"],
            # Synthetic placeholder: the unlabeled target is never read from the cohort.
            "target": torch.tensor(0.0),
            "cohort_index": example["cohort_index"],
            "stable_id": example["stable_id"],
        } for example in examples]

    def __call__(self, examples):
        weak = self.base_collator(self._view(examples, "weak"))
        strong = self.base_collator(self._view(examples, "strong"))
        weak.pop("target", None)
        strong.pop("target", None)
        return {"weak": weak, "strong": strong}
