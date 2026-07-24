"""Official IMDB-WIKI-DIR split and deterministic semi-supervised manifests."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from data_processing.utkface_protocol import (
    TRANSFORM_VERSION,
    build_evaluation_transform,
    build_labeled_transform,
    build_strong_transform,
    build_weak_transform,
    dataloader_generator,
    dataloader_seed,
    label_scaler,
    loader_metadata,
    runtime_metadata,
    seed_dataloader_worker,
)

PROTOCOL_VERSION = "imdb-wiki-dir-benchmark-v1"
COHORT_VERSION = "imdb-wiki-dir-cohort-v1"
OFFICIAL_CSV_SHA256 = "a31f1b43de6804ddbaa2316665a7364e74da3c5c497bdeafb40b910036f7f80b"


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_digest(records):
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(f"{record['path']}\t{record['split']}\t{float(record['age']):.17g}\n".encode())
    return digest.hexdigest()


def build_official_cohort(csv_path):
    csv_path = Path(csv_path).resolve()
    if file_sha256(csv_path) != OFFICIAL_CSV_SHA256:
        raise ValueError("IMDB-WIKI CSV does not match the audited official HPL metadata")
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != ["age", "path", "SPLIT"]:
        raise ValueError(f"Unexpected IMDB-WIKI schema: {list(frame.columns)}")
    records = [
        {"path": str(row.path), "age": float(row.age), "split": str(row.SPLIT)}
        for row in frame.itertuples(index=False)
    ]
    records.sort(key=lambda item: item["path"])
    cohort = {
        "cohort_version": COHORT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cohort_size": len(records),
        "cohort_sha256": cohort_digest(records),
        "official_metadata_sha256": OFFICIAL_CSV_SHA256,
        "records": records,
        "generation": {
            "source": str(csv_path),
            "ordering": "lexicographic POSIX relative image path",
            "split_source": "official SPLIT column; never randomized",
            "age_source": "official age column; no additional filtering",
            "runtime": runtime_metadata(),
        },
    }
    validate_cohort_structure(cohort)
    return cohort


def validate_cohort_structure(cohort):
    records = cohort.get("records", [])
    if len(records) != cohort.get("cohort_size") or not records:
        raise ValueError("Invalid cohort size")
    paths = [record["path"] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Cohort paths are not unique and sorted")
    if any(Path(path).is_absolute() or ".." in PurePosixPath(path).parts for path in paths):
        raise ValueError("Unsafe cohort relative path")
    if any(record["split"] not in {"train", "val", "test"} for record in records):
        raise ValueError("Invalid official split")
    if any(not np.isfinite(float(record["age"])) or float(record["age"]) < 0 for record in records):
        raise ValueError("Invalid age")
    if cohort_digest(records) != cohort.get("cohort_sha256"):
        raise ValueError("Cohort digest mismatch")


def validate_cohort(cohort, data_root):
    validate_cohort_structure(cohort)
    root = Path(data_root).resolve()
    missing = [record["path"] for record in cohort["records"] if not (root / record["path"]).is_file()]
    if missing:
        raise ValueError(f"Missing IMDB-WIKI images: {missing[:10]}")


def save_cohort(cohort, path):
    validate_cohort_structure(cohort)
    _write(path, cohort)


def load_cohort(path):
    cohort = json.loads(Path(path).read_text())
    validate_cohort_structure(cohort)
    return cohort


def generate_seed_manifest(cohort, seed, labeled_ratio):
    validate_cohort_structure(cohort)
    split_indices = {
        split: [index for index, record in enumerate(cohort["records"]) if record["split"] == split]
        for split in ("train", "val", "test")
    }
    train = split_indices["train"]
    permutation = train.copy()
    random.Random(int(seed)).shuffle(permutation)
    labeled_count = int(len(train) * float(labeled_ratio))
    labeled, unlabeled = permutation[:labeled_count], permutation[labeled_count:]
    mean, std = label_scaler([cohort["records"][index]["age"] for index in labeled])
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "transform_version": TRANSFORM_VERSION,
        "cohort_sha256": cohort["cohort_sha256"],
        "seed": int(seed),
        "labeled_ratio": float(labeled_ratio),
        "splits": {"train": train, "validation": split_indices["val"], "test": split_indices["test"]},
        "train_permutation": permutation,
        "labeled_indices": labeled,
        "unlabeled_indices": unlabeled,
        "counts": {"cohort": len(cohort["records"]), "train": len(train),
                   "validation": len(split_indices["val"]), "test": len(split_indices["test"]),
                   "labeled": len(labeled), "unlabeled": len(unlabeled)},
        "label_scaler": {"mean": mean, "std": std, "source": "labeled subset only"},
        "loader_seeds": {role: dataloader_seed(seed, role)
                         for role in ("labeled", "unlabeled", "validation", "test", "uncertainty")},
        "stable_records": [{"path": cohort["records"][index]["path"],
                            "age": cohort["records"][index]["age"]}
                           for index in permutation + split_indices["val"] + split_indices["test"]],
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    validate_seed_manifest(manifest, cohort)
    return manifest


def manifest_digest(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_seed_manifest(manifest, cohort):
    validate_cohort_structure(cohort)
    if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["transform_version"] != TRANSFORM_VERSION:
        raise ValueError("Manifest protocol/transform version mismatch")
    if manifest["cohort_sha256"] != cohort["cohort_sha256"]:
        raise ValueError("Manifest cohort mismatch")
    expected = {split: [i for i, record in enumerate(cohort["records"]) if record["split"] == split]
                for split in ("train", "val", "test")}
    if manifest["splits"]["train"] != expected["train"]:
        raise ValueError("Official train membership changed")
    if manifest["splits"]["validation"] != expected["val"] or manifest["splits"]["test"] != expected["test"]:
        raise ValueError("Official validation/test membership changed")
    labeled, unlabeled = manifest["labeled_indices"], manifest["unlabeled_indices"]
    if labeled + unlabeled != manifest["train_permutation"] or sorted(labeled + unlabeled) != expected["train"]:
        raise ValueError("Labeled/unlabeled membership is invalid")
    mean, std = label_scaler([cohort["records"][i]["age"] for i in labeled])
    if float(manifest["label_scaler"]["mean"]) != mean or float(manifest["label_scaler"]["std"]) != std:
        raise ValueError("Scaler mismatch")
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")


def save_seed_manifest(manifest, path):
    _write(path, manifest)


def load_seed_manifest(path, cohort):
    manifest = json.loads(Path(path).read_text())
    validate_seed_manifest(manifest, cohort)
    return manifest


def manifest_items(cohort, indices, data_root):
    root = Path(data_root).resolve()
    return [(root / cohort["records"][index]["path"], float(cohort["records"][index]["age"]))
            for index in indices]
