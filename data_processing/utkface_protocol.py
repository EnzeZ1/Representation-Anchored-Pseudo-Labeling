"""Shared deterministic UTKFace protocol for the controlled SSR benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROTOCOL_VERSION = "utkface-benchmark-v1"
TRANSFORM_VERSION = "rapl-augmentation-v1-explicit"
COHORT_VERSION = "utkface-cohort-v1"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ELIGIBLE_SUFFIXES = (".jpg", ".png", ".jpeg")
MIN_AGE = 0.0
MAX_AGE = 120.0

# Offsets are deliberately sparse so future loader roles can be added safely.
LOADER_ROLE_OFFSETS = {
    "labeled": 11_000,
    "unlabeled": 22_000,
    "validation": 33_000,
    "test": 44_000,
    "uncertainty": 55_000,
}


def parse_age(path: str | Path) -> float | None:
    """Parse and validate the age encoded before the first filename underscore."""
    try:
        age = float(Path(path).name.split("_", 1)[0])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(age) or not MIN_AGE <= age <= MAX_AGE:
        return None
    return age


def _canonical_record_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [f"{record['path']}\t{float(record['age']):.17g}\n" for record in records]
    return "".join(lines).encode("utf-8")


def cohort_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_record_bytes(records)).hexdigest()


def build_eligible_cohort(data_root: str | Path) -> dict[str, Any]:
    """Build the canonical cohort from sorted, immediate-child relative paths."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UTKFace data directory does not exist: {root}")

    records = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in ELIGIBLE_SUFFIXES:
            continue
        age = parse_age(path)
        if age is not None:
            records.append({"path": path.relative_to(root).as_posix(), "age": age})
    records.sort(key=lambda record: record["path"])
    if not records:
        raise RuntimeError(f"No eligible UTKFace images found in {root}")

    return {
        "cohort_version": COHORT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "records": records,
        "cohort_size": len(records),
        "cohort_sha256": cohort_digest(records),
        "generation": {
            "scope": "immediate files only",
            "relative_path_order": "lexicographic POSIX",
            "eligible_suffixes": list(ELIGIBLE_SUFFIXES),
            "age_source": "filename prefix before first underscore",
            "minimum_age": MIN_AGE,
            "maximum_age": MAX_AGE,
            "runtime": runtime_metadata(),
        },
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_cohort(cohort: Mapping[str, Any], path: str | Path) -> None:
    validate_cohort_structure(cohort)
    _write_json(path, cohort)


def load_cohort(path: str | Path) -> dict[str, Any]:
    cohort = json.loads(Path(path).read_text())
    validate_cohort_structure(cohort)
    return cohort


def validate_cohort_structure(cohort: Mapping[str, Any]) -> None:
    records = cohort.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Cohort must contain a non-empty records list")
    paths = [record.get("path") for record in records]
    if any(not isinstance(path, str) for path in paths):
        raise ValueError("Every cohort record must contain a relative path")
    if any(Path(path).is_absolute() or ".." in PurePosixPath(path).parts for path in paths):
        raise ValueError("Cohort paths must be safe relative paths")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Cohort paths must be unique and canonically sorted")
    for record in records:
        parsed = parse_age(record["path"])
        if parsed is None or float(record.get("age")) != parsed:
            raise ValueError(f"Age mismatch in cohort record: {record}")
    if cohort.get("cohort_size") != len(records):
        raise ValueError("Cohort size does not match its records")
    digest = cohort_digest(records)
    if cohort.get("cohort_sha256") != digest:
        raise ValueError("Cohort SHA-256 does not match its records")


def validate_cohort(cohort: Mapping[str, Any], data_root: str | Path) -> None:
    """Detect missing, extra, and age-mismatched files against a saved cohort."""
    validate_cohort_structure(cohort)
    observed = build_eligible_cohort(data_root)
    expected_by_path = {record["path"]: float(record["age"]) for record in cohort["records"]}
    observed_by_path = {record["path"]: float(record["age"]) for record in observed["records"]}
    missing = sorted(expected_by_path.keys() - observed_by_path.keys())
    extra = sorted(observed_by_path.keys() - expected_by_path.keys())
    mismatched = sorted(
        path for path in expected_by_path.keys() & observed_by_path.keys()
        if expected_by_path[path] != observed_by_path[path]
    )
    if missing or extra or mismatched:
        raise ValueError(
            "UTKFace cohort mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}, age_mismatched={mismatched[:5]}"
        )
    if observed["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("UTKFace cohort digest mismatch")


def label_scaler(ages: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(ages, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot compute a label scaler from no labels")
    return float(values.mean()), float(values.std() + 1e-6)


def normalize_age(age: Any, mean: float, std: float) -> Any:
    return (age - mean) / std


def inverse_normalize_age(value: Any, mean: float, std: float) -> Any:
    return value * std + mean


def generate_seed_manifest(
    cohort: Mapping[str, Any], seed: int, labeled_ratio: float = 0.05
) -> dict[str, Any]:
    validate_cohort_structure(cohort)
    if not 0.0 < labeled_ratio < 1.0:
        raise ValueError("labeled_ratio must be between zero and one")
    indices = list(range(cohort["cohort_size"]))
    random.Random(seed).shuffle(indices)
    n_test = int(0.1 * len(indices))
    n_validation = int(0.1 * len(indices))
    test = indices[:n_test]
    validation = indices[n_test:n_test + n_validation]
    train = indices[n_test + n_validation:]
    n_labeled = max(1, int(labeled_ratio * len(train)))
    labeled = train[:n_labeled]
    unlabeled = train[n_labeled:]
    mean, std = label_scaler([cohort["records"][idx]["age"] for idx in labeled])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "cohort_sha256": cohort["cohort_sha256"],
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "splits": {"train": train, "validation": validation, "test": test},
        "labeled_indices": labeled,
        "unlabeled_indices": unlabeled,
        "counts": {
            "cohort": len(indices),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "labeled": len(labeled),
            "unlabeled": len(unlabeled),
        },
        "label_scaler": {"mean": mean, "std": std, "source": "labeled subset"},
    }


def save_seed_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    _write_json(path, manifest)


def load_seed_manifest(path: str | Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    validate_seed_manifest(manifest, cohort)
    return manifest


def validate_seed_manifest(manifest: Mapping[str, Any], cohort: Mapping[str, Any]) -> None:
    validate_cohort_structure(cohort)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported UTKFace protocol version")
    if manifest.get("transform_version") != TRANSFORM_VERSION:
        raise ValueError("Unsupported UTKFace transform version")
    if manifest.get("cohort_sha256") != cohort["cohort_sha256"]:
        raise ValueError("Manifest references a different cohort")
    n = cohort["cohort_size"]
    train = list(manifest["splits"]["train"])
    validation = list(manifest["splits"]["validation"])
    test = list(manifest["splits"]["test"])
    labeled = list(manifest["labeled_indices"])
    unlabeled = list(manifest["unlabeled_indices"])
    all_indices = train + validation + test
    if sorted(all_indices) != list(range(n)) or len(all_indices) != len(set(all_indices)):
        raise ValueError("Train, validation, and test must partition the cohort")
    if labeled + unlabeled != train:
        raise ValueError("Labeled and unlabeled indices must preserve and partition train order")
    expected_counts = {
        "cohort": n, "train": len(train), "validation": len(validation),
        "test": len(test), "labeled": len(labeled), "unlabeled": len(unlabeled),
    }
    if manifest.get("counts") != expected_counts:
        raise ValueError("Manifest counts do not match index lists")
    expected_mean, expected_std = label_scaler(
        [cohort["records"][idx]["age"] for idx in labeled]
    )
    scaler = manifest["label_scaler"]
    if float(scaler["mean"]) != expected_mean or float(scaler["std"]) != expected_std:
        raise ValueError("Manifest label scaler does not match labeled membership")


def manifest_items(
    cohort: Mapping[str, Any], indices: Sequence[int], data_root: str | Path
) -> list[tuple[Path, float]]:
    root = Path(data_root).resolve()
    return [
        (root / cohort["records"][idx]["path"], float(cohort["records"][idx]["age"]))
        for idx in indices
    ]


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_labeled_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0),
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        _normalize(),
    ])


def build_weak_transform() -> transforms.Compose:
    return build_labeled_transform()


def build_strong_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0),
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(
            num_ops=2, magnitude=10, num_magnitude_bins=31,
            interpolation=InterpolationMode.NEAREST, fill=None,
        ),
        transforms.ToTensor(),
        _normalize(),
    ])


def build_evaluation_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            256, interpolation=InterpolationMode.BILINEAR,
            max_size=None, antialias=True,
        ),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        _normalize(),
    ])


def dataloader_seed(seed: int, role: str, rank: int = 0) -> int:
    if role not in LOADER_ROLE_OFFSETS:
        raise ValueError(f"Unknown DataLoader role: {role}")
    return int(seed) + LOADER_ROLE_OFFSETS[role] + int(rank) * 1_000_000


def dataloader_generator(seed: int, role: str, rank: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed(seed, role, rank))
    return generator


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id  # The worker-specific component is already in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def runtime_metadata() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }


def loader_metadata(
    *, seed: int, role: str, batch_size: int, num_workers: int,
    shuffle: bool, drop_last: bool, sampler: str, pin_memory: bool,
    rank: int = 0,
) -> dict[str, Any]:
    return {
        "role": role,
        "role_offset": LOADER_ROLE_OFFSETS[role],
        "effective_seed": dataloader_seed(seed, role, rank),
        "rank": rank,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "sampler": sampler,
        "worker_init_fn": "seed_dataloader_worker",
        "generator": "explicit torch.Generator",
    }
