#!/usr/bin/env python3
"""Run five smoke tests, validate them, then launch the resumable formal queue."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "utkface_5pct"
EXPECTED_DIGEST = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
EXPECTED_COUNTS = {"cohort": 23709, "train": 18969, "validation": 2370,
                   "test": 2370, "labeled": 948, "unlabeled": 18021}
EXPECTED_MEAN = 32.84599304199219
EXPECTED_STD = 19.87841796875
METHODS = ("rapl", "hpl")


def run_runner(*arguments):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_utkface_benchmark.py"),
                    *arguments], cwd=ROOT, check=True)


def validate_smokes():
    for method in METHODS:
        output = ARTIFACTS / "smoke" / method / "seed_0"
        metadata = json.loads((output / "metadata.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        assert metadata["cohort_sha256"] == EXPECTED_DIGEST
        assert metadata["manifest_seed"] == 0
        assert metadata["counts"] == EXPECTED_COUNTS
        assert metadata["label_scaler"]["mean"] == EXPECTED_MEAN
        assert metadata["label_scaler"]["std"] == EXPECTED_STD
        assert metadata["checkpoint_reloaded"] is True
        assert metadata["test_evaluations"] == 1
        assert metadata["test_used_for_selection"] is False
        assert "ImageNet-pretrained" in metadata["model"]
        assert all(value == value and abs(value) != float("inf")
                   for value in (metrics["validation_mae"], metrics["test_mae"], metrics["test_r2"]))


def main():
    (ARTIFACTS / "pipeline.pid").write_text(f"{__import__('os').getpid()}\n")
    run_runner("--smoke", "--methods", *METHODS, "--seeds", "0",
               "--max_parallel", "2", "--resume")
    validate_smokes()
    run_runner("--methods", *METHODS, "--max_parallel", "6", "--resume")


if __name__ == "__main__":
    main()
