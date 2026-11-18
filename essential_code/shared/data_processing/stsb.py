"""Shared STS-B-DIR protocol and datasets.

The canonical cohort is the balanced STS-B-DIR table distributed by the
official Heteroscedastic-Pseudo-Labels repository.  The official train/dev/test
membership is immutable; only the nested labeled subset is seed-dependent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from data_processing.utkface_protocol import (
    dataloader_generator,
    dataloader_seed,
    label_scaler,
    loader_metadata,
    seed_dataloader_worker,
)

PROTOCOL_VERSION = "stsb-benchmark-v1"
AUGMENTATION_VERSION = "rapl-text-augmentation-v1"
COHORT_VERSION = "stsb-dir-cohort-v1"
CANONICAL_TSV_SHA256 = "be27c303d716552475d6856e286d0a0d2b13a81f328e65e3c842db211ef80437"
EXPECTED_SPLIT_COUNTS = {"train": 5249, "validation": 1000, "test": 1000}
TARGET_RANGE = (0.0, 5.0)
MAX_SEQUENCE_LENGTH = 40


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        f"{record['stable_id']}\t{record['split']}\t"
        f"{float(record['score']):.17g}\t{record['sentence1']}\t{record['sentence2']}\n"
        for record in records
    ).encode("utf-8")


def cohort_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(records)).hexdigest()


def build_canonical_cohort(tsv_path: str | Path) -> dict[str, Any]:
    path = Path(tsv_path).resolve()
    if file_sha256(path) != CANONICAL_TSV_SHA256:
        raise ValueError("STS-B-DIR table does not match the pinned official HPL source")
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    records = []
    split_ordinals = {"train": 0, "validation": 0, "test": 0}
    # Match the official loader's literal tab-column semantics. Some STS-B
    # sentences begin with an unmatched quote, so CSV quote interpretation
    # would incorrectly merge otherwise valid physical rows.
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            row = line.rstrip("\n").split("\t")
            if len(row) < 11:
                raise ValueError("Malformed STS-B-DIR row")
            split = split_map.get(row[10])
            if split is None:
                raise ValueError(f"Unexpected STS-B split: {row[10]!r}")
            score = float(row[9])
            if not np.isfinite(score) or not TARGET_RANGE[0] <= score <= TARGET_RANGE[1]:
                raise ValueError(f"Invalid STS-B score: {score}")
            ordinal = split_ordinals[split]
            split_ordinals[split] += 1
            records.append(
                {
                    "stable_id": f"{split}:{ordinal:05d}",
                    "path": f"{split}/{ordinal:05d}",
                    "source_index": str(row[0]),
                    "split": split,
                    "sentence1": row[7],
                    "sentence2": row[8],
                    "score": score,
                }
            )
    cohort = {
        "cohort_version": COHORT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "augmentation_version": AUGMENTATION_VERSION,
        "source": {
            "repository": "https://github.com/sxq11/Heteroscedastic-Pseudo-Labels.git",
            "commit": "89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20",
            "relative_path": "sts/glue_data/STS-B/sts.tsv",
            "tsv_sha256": CANONICAL_TSV_SHA256,
        },
        "cohort_size": len(records),
        "cohort_sha256": cohort_digest(records),
        "target_range": list(TARGET_RANGE),
        "records": records,
    }
    validate_cohort(cohort)
    return cohort


def validate_cohort(cohort: Mapping[str, Any]) -> None:
    records = cohort.get("records", [])
    if len(records) != 7249 or cohort.get("cohort_size") != len(records):
        raise ValueError("Unexpected STS-B-DIR cohort size")
    ids = [record["stable_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate STS-B stable identifiers")
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in EXPECTED_SPLIT_COUNTS
    }
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected STS-B split counts: {counts}")
    if any(not record["sentence1"] or not record["sentence2"] for record in records):
        raise ValueError("Empty STS-B sentence")
    if cohort_digest(records) != cohort.get("cohort_sha256"):
        raise ValueError("STS-B cohort digest mismatch")


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_cohort(path: str | Path) -> dict[str, Any]:
    cohort = json.loads(Path(path).read_text())
    validate_cohort(cohort)
    return cohort


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_manifest(
    cohort: Mapping[str, Any], seed: int, labeled_ratio: float
) -> dict[str, Any]:
    validate_cohort(cohort)
    if labeled_ratio not in {0.05, 0.10, 0.20, 1.00}:
        raise ValueError("Formal STS-B ratios are 0.05, 0.10, 0.20, and 1.00")
    splits = {
        split: [i for i, record in enumerate(cohort["records"]) if record["split"] == split]
        for split in EXPECTED_SPLIT_COUNTS
    }
    permutation = splits["train"].copy()
    random.Random(int(seed)).shuffle(permutation)
    labeled_count = len(permutation) if labeled_ratio == 1.0 else int(len(permutation) * labeled_ratio)
    labeled = permutation[:labeled_count]
    unlabeled = permutation[labeled_count:]
    mean, std = label_scaler([cohort["records"][index]["score"] for index in labeled])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "augmentation_version": AUGMENTATION_VERSION,
        "cohort_sha256": cohort["cohort_sha256"],
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "splits": splits,
        "train_permutation": permutation,
        "labeled_indices": labeled,
        "unlabeled_indices": unlabeled,
        "counts": {
            "cohort": len(cohort["records"]),
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
            "labeled": len(labeled),
            "unlabeled": len(unlabeled),
        },
        "label_scaler": {
            "mean": mean,
            "std": std,
            "source": "full training split" if labeled_ratio == 1.0 else "labeled subset only",
        },
        "loader_seeds": {
            role: dataloader_seed(seed, role)
            for role in ("labeled", "unlabeled", "validation", "test")
        },
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    validate_manifest(manifest, cohort)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], cohort: Mapping[str, Any]) -> None:
    validate_cohort(cohort)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("STS-B protocol version mismatch")
    if manifest.get("augmentation_version") != AUGMENTATION_VERSION:
        raise ValueError("STS-B augmentation version mismatch")
    if manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise ValueError("STS-B cohort mismatch")
    expected = {
        split: [i for i, record in enumerate(cohort["records"]) if record["split"] == split]
        for split in EXPECTED_SPLIT_COUNTS
    }
    if manifest["splits"] != expected:
        raise ValueError("Canonical STS-B split membership changed")
    labeled = list(manifest["labeled_indices"])
    unlabeled = list(manifest["unlabeled_indices"])
    if labeled + unlabeled != list(manifest["train_permutation"]):
        raise ValueError("STS-B labeled/unlabeled ordering is invalid")
    if sorted(labeled + unlabeled) != expected["train"] or set(labeled) & set(unlabeled):
        raise ValueError("STS-B labeled/unlabeled membership is invalid")
    expected_labeled = (
        len(expected["train"])
        if float(manifest["labeled_ratio"]) == 1.0
        else int(len(expected["train"]) * float(manifest["labeled_ratio"]))
    )
    if len(labeled) != expected_labeled:
        raise ValueError("STS-B labeled count mismatch")
    mean, std = label_scaler([cohort["records"][index]["score"] for index in labeled])
    if mean != float(manifest["label_scaler"]["mean"]) or std != float(manifest["label_scaler"]["std"]):
        raise ValueError("STS-B label scaler mismatch")
    if manifest_digest(manifest) != manifest.get("manifest_sha256"):
        raise ValueError("STS-B manifest checksum mismatch")


def load_manifest(path: str | Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest, cohort)
    return manifest


def validate_manifest_family(
    cohort: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]]
) -> None:
    by_seed: dict[int, dict[float, Mapping[str, Any]]] = {}
    for manifest in manifests:
        validate_manifest(manifest, cohort)
        by_seed.setdefault(int(manifest["seed"]), {})[float(manifest["labeled_ratio"])] = manifest
    expected_ratios = {0.05, 0.10, 0.20, 1.00}
    for seed, family in by_seed.items():
        if set(family) != expected_ratios:
            raise ValueError(f"Seed {seed} does not have all formal ratios")
        reference = family[0.05]["splits"]
        if any(member["splits"] != reference for member in family.values()):
            raise ValueError(f"Seed {seed} split membership differs across ratios")
        sets = [set(family[ratio]["labeled_indices"]) for ratio in sorted(expected_ratios)]
        if not (sets[0] < sets[1] < sets[2] < sets[3]):
            raise ValueError(f"Seed {seed} labeled subsets are not strictly nested")


def simple_tokenize(text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> list[str]:
    """Use the HPL-compatible NLTK tokenizer, capped at 40 tokens."""
    try:
        from nltk.tokenize import word_tokenize
    except ImportError as error:
        raise RuntimeError("nltk is required for the BiLSTM tokenizer") from error
    return word_tokenize(text, preserve_line=True)[:max_length]


class STSBTextDataset(Dataset):
    """Manifest-backed sentence-pair dataset with normalized targets."""

    def __init__(
        self,
        cohort: Mapping[str, Any],
        indices: Sequence[int],
        mean: float,
        std: float,
    ):
        self.cohort = cohort
        self.indices = list(indices)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        record = self.cohort["records"][index]
        return {
            "sentence1": record["sentence1"],
            "sentence2": record["sentence2"],
            "target": torch.tensor((float(record["score"]) - self.mean) / self.std),
            "cohort_index": index,
            "stable_id": record["stable_id"],
        }


# Legacy root-training compatibility. Formal benchmarks use STSBTextDataset and
# manifests above; these symbols preserve the existing `python train.py
# -dataset stsb ...` import contract until that historical path is retired.
class STSBDataset(Dataset):
    def __init__(self, encodings, labels=None, unlabeled=False):
        self.encodings = encodings
        self.labels = labels
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        feature = self.encodings[index]
        if self.unlabeled:
            noise = torch.randn_like(feature) * 0.1
            return feature, feature + noise
        return feature, self.labels[index]


class TextRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.feature_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, value):
        return self.backbone(value)

    def forward(self, value):
        return self.head(self.drop(self.encode(value))).squeeze(-1)


def make_data_stsb(args):
    """Historical MiniLM feature path retained for CLI compatibility."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader

    dataset = load_dataset("glue", "stsb")
    train = [row for row in dataset["train"] if row["label"] >= 0]
    test = [row for row in dataset["validation"] if row["label"] >= 0]
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(rows):
        first = encoder.encode([row["sentence1"] for row in rows], convert_to_tensor=True)
        second = encoder.encode([row["sentence2"] for row in rows], convert_to_tensor=True)
        return torch.cat((first, second, (first - second).abs(), first * second), dim=1).cpu()

    features, test_features = encode(train), encode(test)
    labels = torch.tensor([row["label"] for row in train], dtype=torch.float32)
    test_labels = torch.tensor([row["label"] for row in test], dtype=torch.float32)
    mean, std = label_scaler(labels.numpy())
    labels, test_labels = (labels - mean) / std, (test_labels - mean) / std
    indices = list(range(len(features)))
    random.shuffle(indices)
    validation_count = int(0.1 * len(indices))
    labeled_count = max(1, int(args.labeled_ratio * (len(indices) - validation_count)))
    validation = indices[:validation_count]
    labeled = indices[validation_count:validation_count + labeled_count]
    unlabeled = indices[validation_count + labeled_count:]

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True, drop_last=drop)

    labeled_ds = STSBDataset(features[labeled], labels[labeled])
    labeled_ds.feat_dim = features.shape[1]
    return (
        loader(labeled_ds),
        loader(STSBDataset(features[unlabeled], unlabeled=True)),
        loader(STSBDataset(features[validation], labels[validation]), False, False),
        loader(STSBDataset(test_features, test_labels), False, False),
        mean,
        std,
    )


__all__ = [
    "AUGMENTATION_VERSION",
    "CANONICAL_TSV_SHA256",
    "COHORT_VERSION",
    "EXPECTED_SPLIT_COUNTS",
    "MAX_SEQUENCE_LENGTH",
    "PROTOCOL_VERSION",
    "STSBTextDataset",
    "STSBDataset",
    "TextRegressor",
    "TARGET_RANGE",
    "build_canonical_cohort",
    "cohort_digest",
    "dataloader_generator",
    "dataloader_seed",
    "file_sha256",
    "generate_manifest",
    "label_scaler",
    "load_cohort",
    "load_manifest",
    "loader_metadata",
    "manifest_digest",
    "save_json",
    "seed_dataloader_worker",
    "simple_tokenize",
    "validate_cohort",
    "validate_manifest",
    "validate_manifest_family",
    "make_data_stsb",
]
