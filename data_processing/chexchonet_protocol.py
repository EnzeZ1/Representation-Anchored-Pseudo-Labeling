"""Patient-level CheXchoNet regression protocol (protected artifacts stay local)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROTOCOL_VERSION = "chexchonet-regression-v1"
TRANSFORM_VERSION = "chexchonet-anatomy-preserving-v1"
PRIMARY_TARGET = "lvidd"
FUTURE_TARGETS = ("ivsd", "lvpwd")
RATIOS = (0.05, 0.10, 0.20, 1.00)
SEEDS = tuple(range(6))


def build_train_transform():
    # No horizontal flip: cardiac laterality and anatomy are preserved.
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_evaluation_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_patient_manifest(records, *, seed, ratio, target=PRIMARY_TARGET):
    """Create deterministic nested labeled-patient subsets and patient splits."""
    if ratio not in RATIOS or seed not in SEEDS:
        raise ValueError("Unsupported formal ratio or seed")
    eligible = [i for i, row in enumerate(records) if row["targets"].get(target) is not None]
    patients = sorted({records[i]["patient_id"] for i in eligible})
    shuffled = patients.copy(); random.Random(seed).shuffle(shuffled)
    n_test, n_val = int(.1 * len(shuffled)), int(.1 * len(shuffled))
    test_patients = shuffled[:n_test]
    val_patients = shuffled[n_test:n_test + n_val]
    train_patients = shuffled[n_test + n_val:]
    labeled_count = len(train_patients) if ratio == 1.0 else max(1, int(len(train_patients) * ratio))
    labeled_patients = train_patients[:labeled_count]
    unlabeled_patients = train_patients[labeled_count:]
    patient_sets = {"labeled": set(labeled_patients), "unlabeled": set(unlabeled_patients),
                    "validation": set(val_patients), "test": set(test_patients)}
    indices = {role: [i for i in eligible if records[i]["patient_id"] in members]
               for role, members in patient_sets.items()}
    labels = np.asarray([records[i]["targets"][target] for i in indices["labeled"]], dtype=np.float64)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "transform_version": TRANSFORM_VERSION,
        "target": target, "seed": seed, "labeled_ratio": ratio,
        "patient_splits": {"train": train_patients, "validation": val_patients, "test": test_patients},
        "labeled_patients": labeled_patients, "unlabeled_patients": unlabeled_patients,
        "indices": indices,
        "counts": {"eligible_images": len(eligible), "train_patients": len(train_patients),
                   "validation_patients": len(val_patients), "test_patients": len(test_patients),
                   "labeled_patients": len(labeled_patients), "unlabeled_patients": len(unlabeled_patients),
                   **{f"{role}_images": len(values) for role, values in indices.items()}},
        "label_scaler": {"mean": float(labels.mean()), "std": float(labels.std() + 1e-6),
                         "source": "labeled patients only"},
    }
    manifest["manifest_sha256"] = _digest(manifest)
    validate_manifest(manifest, records)
    return manifest


def validate_manifest(manifest, records):
    if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["transform_version"] != TRANSFORM_VERSION:
        raise ValueError("Protocol version mismatch")
    roles = ("labeled", "unlabeled", "validation", "test")
    sets = [set(manifest["indices"][role]) for role in roles]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Image membership overlaps")
    patient_sets = [set(manifest["labeled_patients"]), set(manifest["unlabeled_patients"]),
                    set(manifest["patient_splits"]["validation"]), set(manifest["patient_splits"]["test"])]
    if any(patient_sets[i] & patient_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Patient membership overlaps")
    values = np.asarray([records[i]["targets"][manifest["target"]]
                         for i in manifest["indices"]["labeled"]], dtype=np.float64)
    scaler = manifest["label_scaler"]
    if not np.isclose(scaler["mean"], values.mean()) or not np.isclose(scaler["std"], values.std() + 1e-6):
        raise ValueError("Labeled-only scaler mismatch")
    payload = dict(manifest); checksum = payload.pop("manifest_sha256")
    if checksum != _digest(payload): raise ValueError("Manifest checksum mismatch")


def validate_nested(manifests):
    ordered = [manifests[ratio] for ratio in RATIOS]
    base = ordered[0]["patient_splits"]
    if any(item["patient_splits"] != base for item in ordered[1:]):
        raise ValueError("Ratios do not share patient partitions")
    labeled = [set(item["labeled_patients"]) for item in ordered]
    if not (labeled[0] < labeled[1] < labeled[2] < labeled[3]):
        raise ValueError("Labeled patient subsets are not strictly nested")


def save_manifest(manifest, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
