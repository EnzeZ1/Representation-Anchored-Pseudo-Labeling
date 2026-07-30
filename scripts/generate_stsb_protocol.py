#!/usr/bin/env python3
"""Generate and validate deterministic STS-B-DIR formal manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_processing.stsb import (
    build_canonical_cohort,
    generate_manifest,
    save_json,
    validate_manifest_family,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/stsb/sts.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "data_processing/splits")
    args = parser.parse_args()
    cohort = build_canonical_cohort(args.source)
    cohort_path = args.output / "stsb_dir_cohort_v1.json"
    save_json(cohort_path, cohort)
    manifests = []
    for seed in range(6):
        for ratio in (0.05, 0.10, 0.20, 1.00):
            manifest = generate_manifest(cohort, seed, ratio)
            manifests.append(manifest)
            save_json(args.output / f"stsb_ratio_{ratio:.2f}_seed_{seed}.json", manifest)
    validate_manifest_family(cohort, manifests)
    print(json.dumps({
        "status": "pass",
        "cohort_path": str(cohort_path),
        "cohort_sha256": cohort["cohort_sha256"],
        "counts": {
            f"{ratio:.2f}": next(
                manifest["counts"] for manifest in manifests
                if manifest["seed"] == 0 and manifest["labeled_ratio"] == ratio
            )
            for ratio in (0.05, 0.10, 0.20, 1.00)
        },
        "manifests": len(manifests),
        "nested": True,
    }, indent=2))


if __name__ == "__main__":
    main()
