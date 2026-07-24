"""Dataset adapters exposing the shared protocol in upstream-native formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_processing.utkface_protocol import (
    build_evaluation_transform, build_labeled_transform, build_strong_transform,
    build_weak_transform, load_cohort, load_seed_manifest, validate_cohort,
)


class BenchmarkContext:
    def __init__(self, project_root, data_root, manifest_path):
        self.project_root = Path(project_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        if self.manifest_path.name.startswith("imdb_wiki_"):
            from data_processing.imdb_wiki_protocol import (
                load_cohort as load_dataset_cohort,
                load_seed_manifest as load_dataset_manifest,
                validate_cohort as validate_dataset_cohort,
            )
            self.cohort_path = self.project_root / "data_processing/splits/imdb_wiki_dir_cohort_v1.json"
        else:
            load_dataset_cohort = load_cohort
            load_dataset_manifest = load_seed_manifest
            validate_dataset_cohort = validate_cohort
            self.cohort_path = self.project_root / "data_processing/splits/utkface_cohort_v1.json"
        self.cohort = load_dataset_cohort(self.cohort_path)
        validate_dataset_cohort(self.cohort, self.data_root)
        self.manifest = load_dataset_manifest(self.manifest_path, self.cohort)
        self.mean = float(self.manifest["label_scaler"]["mean"])
        self.std = float(self.manifest["label_scaler"]["std"])

    def indices(self, split):
        if split == "labeled":
            return self.manifest["labeled_indices"]
        if split == "unlabeled":
            return self.manifest["unlabeled_indices"]
        return self.manifest["splits"][split]

    def record(self, index):
        return self.cohort["records"][index]

    def path(self, index):
        return self.data_root / self.record(index)["path"]

    def protocol_metadata(self):
        return {
            "cohort_sha256": self.cohort["cohort_sha256"],
            "manifest_path": str(self.manifest_path),
            "manifest_seed": self.manifest["seed"],
            "counts": self.manifest["counts"],
            "label_scaler": {"mean": self.mean, "std": self.std},
            "protocol_version": self.manifest["protocol_version"],
            "transform_version": self.manifest["transform_version"],
        }


class TupleDataset(Dataset):
    """HPL/UCVME tuple contract, optionally with paired weak/strong views."""
    def __init__(self, context, split, views="labeled", repeat=1):
        self.context, self.split, self.views = context, split, views
        self.indices = list(context.indices(split)) * int(repeat)
        self.labeled_transform = build_labeled_transform()
        self.weak_transform = build_weak_transform()
        self.strong_transform = build_strong_transform()
        self.eval_transform = build_evaluation_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        record = self.context.record(index)
        with Image.open(self.context.path(index)) as image:
            image = image.convert("RGB")
            if self.views == "weak_strong":
                value = (self.weak_transform(image), self.strong_transform(image))
            elif self.views == "weak":
                value = self.weak_transform(image)
            elif self.views == "evaluation":
                value = self.eval_transform(image)
            else:
                value = self.labeled_transform(image)
        return value, np.float32(record["age"])


class DictDataset(Dataset):
    """SimRegMatch-compatible dictionary contract."""
    def __init__(self, context, split, views="labeled"):
        self.context, self.views = context, views
        self.indices = list(context.indices(split))
        self.labeled_transform = build_labeled_transform()
        self.weak_transform = build_weak_transform()
        self.strong_transform = build_strong_transform()
        self.eval_transform = build_evaluation_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        cohort_index = self.indices[position]
        record = self.context.record(cohort_index)
        with Image.open(self.context.path(cohort_index)) as image:
            image = image.convert("RGB")
            label = np.asarray([record["age"]], dtype=np.float32)
            if self.views == "weak_strong":
                return {"idx": cohort_index, "weak": self.weak_transform(image),
                        "strong": self.strong_transform(image), "label": label}
            transform = self.eval_transform if self.views == "evaluation" else self.labeled_transform
            return {"idx": cohort_index, "input": transform(image), "label": label}


class RankUpDataset(Dataset):
    """RankUp dictionary keys while retaining cohort identities."""
    def __init__(self, context, split, unlabeled=False):
        self.context, self.unlabeled = context, unlabeled
        self.indices = list(context.indices(split))
        self.targets = np.asarray([context.record(index)["age"] for index in self.indices], dtype=np.float32)
        self.weak = build_weak_transform()
        self.strong = build_strong_transform()
        self.labeled = build_labeled_transform()
        self.evaluation = build_evaluation_transform()
        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        cohort_index = self.indices[position]
        with Image.open(self.context.path(cohort_index)) as image:
            image = image.convert("RGB")
            if self.unlabeled:
                return {"idx_ulb": position, "x_ulb_w": self.weak(image), "x_ulb_s": self.strong(image)}
            transform = self.labeled if self.split == "labeled" else self.evaluation
            return {"idx_lb": position, "x_lb": transform(image), "y_lb": self.targets[position]}
