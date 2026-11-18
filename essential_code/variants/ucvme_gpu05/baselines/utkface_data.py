"""Dataset adapters exposing the shared protocol in upstream-native formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data_processing.utkface_protocol import (
    build_evaluation_transform, build_labeled_transform, build_strong_transform,
    build_weak_transform, load_cohort, load_seed_manifest,
    validate_cohort_structure,
)


class BenchmarkContext:
    def __init__(self, project_root, data_root, manifest_path):
        self.project_root = Path(project_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset = "utkface"
        self.target_name = "age"
        self.target_unit = "years"
        self.image_root = self.data_root
        self.transforms = None
        if self.manifest_path.name.startswith("chexchonet_lvidd_"):
            import json
            from data_processing.chexchonet import discover_metadata, load_records
            from data_processing.chexchonet_protocol import (
                build_evaluation_transform, build_strong_transform,
                build_train_transform, build_weak_transform, validate_manifest,
            )
            self.dataset = "chexchonet_lvidd"
            self.target_name = "lvidd"
            self.target_unit = "centimeters"
            self.cohort_path = discover_metadata(self.data_root)
            self.cohort = {"records": load_records(self.cohort_path)}
            self.manifest = json.loads(self.manifest_path.read_text())
            validate_manifest(self.manifest, self.cohort["records"])
            self.image_root = self.data_root / "images"
            self.transforms = (build_train_transform, build_weak_transform,
                               build_strong_transform, build_evaluation_transform)
            self.mean = float(self.manifest["label_scaler"]["mean"])
            self.std = float(self.manifest["label_scaler"]["std"])
            return
        if self.manifest_path.name.startswith("ratio_") and "rsna_bone_age" in str(self.manifest_path):
            import json
            from data_processing.rsna_bone_age import build_transform, validate_manifest
            self.dataset = "rsna_bone_age"
            self.target_name = "bone_age_months"
            self.target_unit = "months"
            self.cohort_path = self.manifest_path.parent / "cohort.json"
            self.cohort = json.loads(self.cohort_path.read_text())
            self.manifest = json.loads(self.manifest_path.read_text())
            validate_manifest(self.manifest, self.cohort)
            self.image_root = self.data_root
            self.transforms = (lambda: build_transform("weak"), lambda: build_transform("weak"),
                               lambda: build_transform("strong"), lambda: build_transform("eval"))
            self.mean = float(self.manifest["target_scaler"]["mean"])
            self.std = float(self.manifest["target_scaler"]["std"])
            return
        if self.manifest_path.name.startswith("imdb_wiki_"):
            from data_processing.imdb_wiki_protocol import (
                load_cohort as load_dataset_cohort,
                load_seed_manifest as load_dataset_manifest,
                validate_cohort as validate_dataset_cohort,
            )
            self.cohort_path = self.project_root / "data_processing/splits/imdb_wiki_dir_cohort_v1.json"
            validate_dataset_files = True
        else:
            load_dataset_cohort = load_cohort
            load_dataset_manifest = load_seed_manifest
            self.cohort_path = self.project_root / "data_processing/splits/utkface_cohort_v1.json"
            validate_dataset_files = False
        self.cohort = load_dataset_cohort(self.cohort_path)
        if validate_dataset_files:
            validate_dataset_cohort(self.cohort, self.data_root)
        else:
            # A raw UTKFace directory can contain alternate copies that are not
            # members of the canonical cohort. Validate every formal reference
            # without treating unrelated raw files as benchmark samples.
            validate_cohort_structure(self.cohort)
            missing = [
                record["path"] for record in self.cohort["records"]
                if not (self.data_root / record["path"]).is_file()
            ]
            if missing:
                raise ValueError(f"Missing formal UTKFace cohort files: {missing[:5]}")
        self.manifest = load_dataset_manifest(self.manifest_path, self.cohort)
        self.mean = float(self.manifest["label_scaler"]["mean"])
        self.std = float(self.manifest["label_scaler"]["std"])

    def indices(self, split):
        if self.dataset == "chexchonet_lvidd":
            return self.manifest["indices"][split]
        if self.dataset == "rsna_bone_age":
            if split in ("labeled", "unlabeled"):
                return self.manifest[f"{split}_indices"]
            return self.cohort["splits"][split]
        if split == "labeled":
            return self.manifest["labeled_indices"]
        if split == "unlabeled":
            return self.manifest["unlabeled_indices"]
        return self.manifest["splits"][split]

    def record(self, index):
        return self.cohort["records"][index]

    def path(self, index):
        record = self.record(index)
        if self.dataset == "chexchonet_lvidd":
            return self.image_root / record["image_path"]
        if self.dataset == "rsna_bone_age":
            return self.image_root / record["image_path"]
        return self.data_root / record["path"]

    def target(self, index):
        record = self.record(index)
        if self.dataset == "chexchonet_lvidd":
            return float(record["targets"]["lvidd"])
        return float(record[self.target_name])

    def relative_path(self, index):
        record = self.record(index)
        return record.get("path", record.get("image_path"))

    def protocol_metadata(self):
        if self.dataset == "chexchonet_lvidd":
            return {"dataset": "CheXchoNet", "target": "LVIDd", "target_unit": self.target_unit,
                    "cohort_sha256": self.manifest["cohort_digest"], "cohort_digest": self.manifest["cohort_digest"],
                    "split_digest": self.manifest["split_digest"], "manifest_path": str(self.manifest_path),
                    "manifest_sha256": self.manifest["manifest_sha256"], "manifest_seed": self.manifest["seed"],
                    "counts": self.manifest["counts"], "label_scaler": self.manifest["label_scaler"],
                    "protocol_version": self.manifest["protocol_version"], "transform_version": self.manifest["transform_version"]}
        if self.dataset == "rsna_bone_age":
            return {"dataset": "RSNA Pediatric Bone Age Challenge 2017", "target": "bone age",
                    "target_unit": self.target_unit, "cohort_sha256": self.cohort["cohort_digest"],
                    "cohort_digest": self.cohort["cohort_digest"], "split_digest": self.cohort["split_digest"],
                    "manifest_path": str(self.manifest_path), "manifest_sha256": self.manifest["manifest_digest"],
                    "manifest_digest": self.manifest["manifest_digest"], "manifest_seed": self.manifest["seed"],
                    "counts": self.manifest["counts"], "label_scaler": self.manifest["target_scaler"],
                    "protocol_version": self.manifest["protocol_version"], "transform_version": self.manifest["transform_version"]}
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
        factories = context.transforms or (build_labeled_transform, build_weak_transform,
                                            build_strong_transform, build_evaluation_transform)
        self.labeled_transform, self.weak_transform, self.strong_transform, self.eval_transform = [f() for f in factories]

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
        return value, np.float32(self.context.target(index))


class DictDataset(Dataset):
    """SimRegMatch-compatible dictionary contract."""
    def __init__(self, context, split, views="labeled"):
        self.context, self.views = context, views
        self.indices = list(context.indices(split))
        factories = context.transforms or (build_labeled_transform, build_weak_transform,
                                            build_strong_transform, build_evaluation_transform)
        self.labeled_transform, self.weak_transform, self.strong_transform, self.eval_transform = [f() for f in factories]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        cohort_index = self.indices[position]
        record = self.context.record(cohort_index)
        with Image.open(self.context.path(cohort_index)) as image:
            image = image.convert("RGB")
            label = np.asarray([self.context.target(cohort_index)], dtype=np.float32)
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
