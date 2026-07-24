"""Small, algorithm-neutral helpers for UTKFace benchmark artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REQUIRED_RUN_FILES = (
    "best.pt", "config.json", "metadata.json", "metrics.json", "history.csv",
    "run.log", "test_predictions.npz", "analysis_snapshot.npz",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def write_history(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("History must contain at least one row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def run_complete(output_dir: str | Path) -> bool:
    root = Path(output_dir)
    if not all((root / name).is_file() for name in REQUIRED_RUN_FILES):
        return False
    try:
        metrics = json.loads((root / "metrics.json").read_text())
        metadata = json.loads((root / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    finite = all(np.isfinite(metrics.get(key, np.nan)) for key in ("test_mae", "test_r2", "validation_mae"))
    return bool(finite and metadata.get("status") == "complete" and metadata.get("checkpoint_reloaded"))


def base_metadata(method: str, seed: int) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "status": "running",
        "started_unix": time.time(),
        "test_used_for_selection": False,
        "checkpoint_reloaded": False,
    }
