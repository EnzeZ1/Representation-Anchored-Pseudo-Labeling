"""Secure access to the official CheXchoNet 1.0.0 release.

Protected identifiers are kept in memory/local ignored artifacts and are never logged.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGETS = ("lvidd", "ivsd", "lvpwd")
OFFICIAL_IMAGE_COLUMN = "cxr_filename"
OFFICIAL_PATIENT_COLUMN = "patient_id"


def discover_metadata(root: str | Path) -> Path | None:
    candidates = sorted(Path(root).glob("*.csv"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(f"Expected one official metadata CSV, found {len(candidates)}")
    return candidates[0]


def load_records(metadata_path: str | Path) -> list[dict]:
    """Parse the official schema without exposing protected identifiers."""
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {OFFICIAL_IMAGE_COLUMN, OFFICIAL_PATIENT_COLUMN, *TARGETS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Official CheXchoNet metadata schema mismatch")
        records = []
        for row_number, row in enumerate(reader, start=2):
            relative = str(row[OFFICIAL_IMAGE_COLUMN]).strip()
            patient = str(row[OFFICIAL_PATIENT_COLUMN]).strip()
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ValueError(f"Unsafe official image mapping at metadata row {row_number}")
            if not patient:
                raise ValueError(f"Missing patient identifier at metadata row {row_number}")
            values = {}
            for target in TARGETS:
                raw = str(row[target]).strip()
                try:
                    values[target] = float(raw) if raw else None
                except ValueError:
                    values[target] = None
            records.append({"image_path": relative, "patient_id": patient, "targets": values})
    return records


def lvidd_cohort_indices(records) -> list[int]:
    return [i for i, row in enumerate(records)
            if row["targets"].get("lvidd") is not None
            and np.isfinite(row["targets"]["lvidd"])
            and row["targets"]["lvidd"] > 0]


def resolve_image(image_root: str | Path, record: Mapping) -> Path:
    """Resolve the single official JPEG mapping; no fallback heuristics."""
    return Path(image_root) / record["image_path"]


@dataclass(frozen=True)
class ReleaseAudit:
    metadata_rows: int
    image_files: int
    unique_patients: int
    finite_lvidd: int
    nonfinite_lvidd: int
    nonpositive_lvidd: int
    eligible_lvidd: int
    eligible_patients: int
    missing_images: int
    corrupt_images: int

    @property
    def ready(self):
        return self.metadata_rows > 0 and self.metadata_rows == self.image_files and not self.missing_images and not self.corrupt_images


def audit_release(root: str | Path, *, decode: bool = False) -> ReleaseAudit:
    root = Path(root); metadata = discover_metadata(root)
    if metadata is None:
        return ReleaseAudit(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    records = load_records(metadata); image_root = root / "images"
    missing = corrupt = 0
    for record in records:
        path = resolve_image(image_root, record)
        if not path.is_file(): missing += 1
        elif decode:
            try:
                with Image.open(path) as image: image.load()
            except (OSError, ValueError): corrupt += 1
    values = np.asarray([r["targets"]["lvidd"] if r["targets"]["lvidd"] is not None else np.nan for r in records])
    finite = np.isfinite(values); eligible = finite & (values > 0)
    return ReleaseAudit(
        len(records), sum(1 for p in image_root.iterdir() if p.is_file()),
        len({r["patient_id"] for r in records}), int(finite.sum()), int((~finite).sum()),
        int((finite & (values <= 0)).sum()), int(eligible.sum()),
        len({records[i]["patient_id"] for i in np.flatnonzero(eligible)}), missing, corrupt,
    )


class CheXchoNetDataset(Dataset):
    def __init__(self, records, image_root, indices, mean, std, transform=None, strong_transform=None):
        self.records, self.image_root, self.indices = records, Path(image_root), list(indices)
        self.mean, self.std = float(mean), float(std)
        self.transform, self.strong_transform = transform, strong_transform

    def __len__(self): return len(self.indices)

    def _decode(self, index):
        with Image.open(resolve_image(self.image_root, self.records[index])) as source:
            return source.convert("L").convert("RGB")

    def __getitem__(self, position):
        index = self.indices[position]; image = self._decode(index)
        if self.strong_transform is not None:
            return self.transform(image.copy()), self.strong_transform(image.copy()), index
        value = float(self.records[index]["targets"]["lvidd"])
        return self.transform(image) if self.transform else image, torch.tensor((value-self.mean)/self.std,dtype=torch.float32), index
