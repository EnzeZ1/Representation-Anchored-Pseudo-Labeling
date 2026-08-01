"""Secure, schema-tolerant access to an authorized CheXchoNet release.

This module deliberately never downloads data and never logs row identifiers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

TARGETS = ("lvidd", "ivsd", "lvpwd")
IMAGE_COLUMNS = ("image_path", "image", "path", "filename", "file")
PATIENT_COLUMNS = ("patient_id", "patient", "subject_id", "subject")


def _column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    lookup = {name.casefold(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise ValueError(f"Required metadata role absent; accepted columns: {tuple(candidates)}")


def discover_metadata(root: str | Path) -> Path | None:
    """Return the sole top-level CSV metadata file, or fail on ambiguity."""
    candidates = sorted(Path(root).glob("*.csv"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Expected one official metadata CSV, found {len(candidates)}")
    return candidates[0]


def load_records(metadata_path: str | Path) -> list[dict]:
    """Load protected metadata in memory without emitting identifiers."""
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Metadata has no header")
        image_col = _column(reader.fieldnames, IMAGE_COLUMNS)
        patient_col = _column(reader.fieldnames, PATIENT_COLUMNS)
        target_columns = {
            target: _column(reader.fieldnames, (target, target.upper()))
            for target in TARGETS
            if any(name.casefold() == target for name in reader.fieldnames)
        }
        if "lvidd" not in target_columns:
            raise ValueError("Primary target LVIDd is absent")
        records = []
        for row_number, row in enumerate(reader, start=2):
            relative = str(row[image_col]).strip()
            patient = str(row[patient_col]).strip()
            parts = PurePosixPath(relative).parts
            if not relative or PurePosixPath(relative).is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe image path at metadata row {row_number}")
            if not patient:
                raise ValueError(f"Missing patient identifier at metadata row {row_number}")
            values = {}
            for target, column in target_columns.items():
                raw = str(row[column]).strip()
                values[target] = float(raw) if raw else None
                if values[target] is not None and not np.isfinite(values[target]):
                    raise ValueError(f"Non-finite {target} at metadata row {row_number}")
            records.append({"image_path": relative, "patient_id": patient, "targets": values})
    return records


@dataclass(frozen=True)
class ReleaseAudit:
    metadata_present: bool
    image_root_present: bool
    metadata_rows: int
    image_files: int
    unique_patients: int
    target_counts: Mapping[str, int]
    missing_images: int
    corrupt_images: int

    @property
    def ready(self) -> bool:
        return bool(self.metadata_present and self.image_root_present and self.metadata_rows
                    and self.missing_images == 0 and self.corrupt_images == 0)


def audit_release(root: str | Path, *, decode: bool = False) -> ReleaseAudit:
    root = Path(root)
    metadata = discover_metadata(root) if root.is_dir() else None
    image_root = root / "images"
    if metadata is None:
        return ReleaseAudit(False, image_root.is_dir(), 0, 0, 0,
                            {target: 0 for target in TARGETS}, 0, 0)
    records = load_records(metadata)
    missing = corrupt = 0
    for record in records:
        path = image_root / record["image_path"]
        if not path.is_file():
            missing += 1
        elif decode:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                corrupt += 1
    image_files = sum(1 for path in image_root.rglob("*") if path.is_file()) if image_root.is_dir() else 0
    return ReleaseAudit(True, image_root.is_dir(), len(records), image_files,
                        len({record["patient_id"] for record in records}),
                        {target: sum(record["targets"].get(target) is not None for record in records)
                         for target in TARGETS}, missing, corrupt)


class CheXchoNetDataset(Dataset):
    def __init__(self, records, image_root, indices, target="lvidd", transform=None):
        self.records, self.image_root = records, Path(image_root)
        self.indices, self.target, self.transform = list(indices), target.casefold(), transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        record = self.records[index]
        value = record["targets"].get(self.target)
        if value is None:
            raise ValueError("Selected record lacks requested target")
        with Image.open(self.image_root / record["image_path"]) as source:
            image = source.convert("RGB")
        return (self.transform(image) if self.transform else image), float(value), index
