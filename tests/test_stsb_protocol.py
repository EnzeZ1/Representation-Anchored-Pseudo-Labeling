import copy

import pytest

from data_processing.stsb import generate_manifest, validate_manifest_family


def cohort():
    records = []
    for split, count in (("train", 5249), ("validation", 1000), ("test", 1000)):
        for index in range(count):
            records.append({
                "stable_id": f"{split}:{index:05d}",
                "source_index": str(index),
                "split": split,
                "sentence1": f"first sentence {index}",
                "sentence2": f"second sentence {index}",
                "score": float(index % 6),
            })
    from data_processing.stsb import (
        AUGMENTATION_VERSION, COHORT_VERSION, PROTOCOL_VERSION, cohort_digest
    )
    return {
        "cohort_version": COHORT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "augmentation_version": AUGMENTATION_VERSION,
        "cohort_size": len(records),
        "cohort_sha256": cohort_digest(records),
        "target_range": [0.0, 5.0],
        "records": records,
    }


def test_manifest_counts_nesting_and_determinism():
    source = cohort()
    manifests = [generate_manifest(source, seed, ratio)
                 for seed in range(6) for ratio in (0.05, 0.10, 0.20, 1.00)]
    validate_manifest_family(source, manifests)
    expected = {0.05: 262, 0.10: 524, 0.20: 1049, 1.00: 5249}
    for manifest in manifests:
        assert manifest["counts"]["labeled"] == expected[manifest["labeled_ratio"]]
        assert manifest["counts"]["unlabeled"] == 5249 - expected[manifest["labeled_ratio"]]
        assert manifest == generate_manifest(source, manifest["seed"], manifest["labeled_ratio"])


def test_manifest_detects_overlap():
    source = cohort()
    manifest = generate_manifest(source, 0, 0.05)
    broken = copy.deepcopy(manifest)
    broken["unlabeled_indices"][0] = broken["labeled_indices"][0]
    with pytest.raises(ValueError):
        from data_processing.stsb import validate_manifest
        validate_manifest(broken, source)
