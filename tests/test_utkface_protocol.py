import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data_processing.utkface_protocol import (
    COHORT_VERSION,
    IMAGE_SIZE,
    LOADER_ROLE_OFFSETS,
    PROTOCOL_VERSION,
    TRANSFORM_VERSION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eligible_cohort,
    build_evaluation_transform,
    build_labeled_transform,
    build_strong_transform,
    build_weak_transform,
    dataloader_generator,
    generate_seed_manifest,
    inverse_normalize_age,
    load_cohort,
    load_seed_manifest,
    normalize_age,
    seed_dataloader_worker,
    validate_cohort,
    validate_cohort_structure,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "utkface_all"
SPLITS = ROOT / "data_processing" / "splits"
EXPECTED_SIZE = 23_709
EXPECTED_DIGEST = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
EXPECTED_COUNTS = {
    "cohort": 23_709,
    "train": 18_969,
    "validation": 2_370,
    "test": 2_370,
    "labeled": 948,
    "unlabeled": 18_021,
}


class _RandomDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index, random.random(), np.random.random(), torch.rand(())


class UTKFaceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = load_cohort(SPLITS / "utkface_cohort_v1.json")

    def test_expected_cohort_and_digest(self):
        self.assertEqual(self.cohort["cohort_version"], COHORT_VERSION)
        self.assertEqual(self.cohort["cohort_size"], EXPECTED_SIZE)
        self.assertEqual(self.cohort["cohort_sha256"], EXPECTED_DIGEST)
        validate_cohort(self.cohort, DATA_ROOT)

    def test_seed_manifests_partition_and_counts(self):
        for seed in range(6):
            path = SPLITS / f"utkface_ratio_0.05_seed_{seed}.json"
            first = load_seed_manifest(path, self.cohort)
            second = load_seed_manifest(path, self.cohort)
            self.assertEqual(first, second)
            self.assertEqual(first["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual(first["transform_version"], TRANSFORM_VERSION)
            self.assertEqual(first["counts"], EXPECTED_COUNTS)
            train = first["splits"]["train"]
            validation = first["splits"]["validation"]
            test = first["splits"]["test"]
            self.assertTrue(set(train).isdisjoint(validation))
            self.assertTrue(set(train).isdisjoint(test))
            self.assertTrue(set(validation).isdisjoint(test))
            self.assertEqual(first["labeled_indices"] + first["unlabeled_indices"], train)
            self.assertTrue(set(first["labeled_indices"]).isdisjoint(first["unlabeled_indices"]))
            self.assertNotIn("records", first)
            self.assertEqual(first["cohort_sha256"], EXPECTED_DIGEST)

    def _assert_training_transform(self, transform, strong=False):
        operations = transform.transforms
        crop = operations[0]
        self.assertIsInstance(crop, transforms.RandomResizedCrop)
        self.assertEqual(crop.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(crop.scale, (0.8, 1.0))
        self.assertEqual(crop.ratio, (0.75, 4.0 / 3.0))
        self.assertEqual(crop.interpolation, InterpolationMode.BILINEAR)
        self.assertIs(crop.antialias, True)
        self.assertIsInstance(operations[1], transforms.RandomHorizontalFlip)
        self.assertEqual(operations[1].p, 0.5)
        if strong:
            augment = operations[2]
            self.assertIsInstance(augment, transforms.RandAugment)
            self.assertEqual(augment.num_ops, 2)
            self.assertEqual(augment.magnitude, 10)
            self.assertEqual(augment.num_magnitude_bins, 31)
            self.assertEqual(augment.interpolation, InterpolationMode.NEAREST)
            self.assertIsNone(augment.fill)
        normalization = operations[-1]
        self.assertIsInstance(normalization, transforms.Normalize)
        self.assertEqual(tuple(normalization.mean), IMAGENET_MEAN)
        self.assertEqual(tuple(normalization.std), IMAGENET_STD)

    def test_transform_definitions_are_explicit(self):
        self._assert_training_transform(build_labeled_transform())
        self._assert_training_transform(build_weak_transform())
        self._assert_training_transform(build_strong_transform(), strong=True)
        evaluation = build_evaluation_transform().transforms
        self.assertIsInstance(evaluation[0], transforms.Resize)
        self.assertEqual(evaluation[0].size, 256)
        self.assertEqual(evaluation[0].interpolation, InterpolationMode.BILINEAR)
        self.assertIs(evaluation[0].antialias, True)
        self.assertIsInstance(evaluation[1], transforms.CenterCrop)
        self.assertEqual(evaluation[1].size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tuple(evaluation[-1].mean), IMAGENET_MEAN)
        self.assertEqual(tuple(evaluation[-1].std), IMAGENET_STD)

    def test_label_normalization_round_trip(self):
        manifest = load_seed_manifest(SPLITS / "utkface_ratio_0.05_seed_3.json", self.cohort)
        mean = manifest["label_scaler"]["mean"]
        std = manifest["label_scaler"]["std"]
        ages = torch.tensor([0.0, 37.0, 120.0])
        restored = inverse_normalize_age(normalize_age(ages, mean, std), mean, std)
        torch.testing.assert_close(restored, ages)

    @staticmethod
    def _deterministic_batches():
        loader = DataLoader(
            _RandomDataset(), batch_size=4, shuffle=True, num_workers=0,
            generator=dataloader_generator(3, "labeled"),
        )
        return [batch[0].tolist() for batch in loader]

    def test_dataloader_seed_determinism(self):
        self.assertEqual(self._deterministic_batches(), self._deterministic_batches())
        self.assertEqual(len(set(LOADER_ROLE_OFFSETS.values())), len(LOADER_ROLE_OFFSETS))

    def test_worker_seed_determinism(self):
        with mock.patch("torch.initial_seed", return_value=123456789):
            seed_dataloader_worker(0)
            first = (random.random(), np.random.random())
            seed_dataloader_worker(0)
            second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_manifests_contain_no_absolute_paths(self):
        for path in SPLITS.glob("*.json"):
            text = path.read_text()
            payload = json.loads(text)
            for record in payload.get("records", []):
                self.assertFalse(Path(record["path"]).is_absolute())
            self.assertNotIn(str(ROOT), text)

    @staticmethod
    def _touch_image(path):
        path.write_bytes(b"not opened by cohort validation")

    def test_missing_and_extra_files_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._touch_image(root / "10_a.jpg")
            self._touch_image(root / "20_b.png")
            cohort = build_eligible_cohort(root)
            (root / "20_b.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_cohort(cohort, root)
            self._touch_image(root / "20_b.png")
            self._touch_image(root / "30_extra.jpeg")
            with self.assertRaisesRegex(ValueError, "extra"):
                validate_cohort(cohort, root)

    def test_age_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._touch_image(root / "10_a.jpg")
            cohort = build_eligible_cohort(root)
            corrupted = copy.deepcopy(cohort)
            corrupted["records"][0]["age"] = 11.0
            with self.assertRaisesRegex(ValueError, "Age mismatch"):
                validate_cohort_structure(corrupted)

    def test_generation_is_deterministic(self):
        self.assertEqual(
            generate_seed_manifest(self.cohort, 4, 0.05),
            generate_seed_manifest(self.cohort, 4, 0.05),
        )

    def test_ratio_manifests_share_partitions_and_are_nested(self):
        expected = {0.05: (948, 18_021), 0.10: (1_896, 17_073), 0.20: (3_793, 15_176)}
        for seed in range(6):
            manifests = {
                ratio: load_seed_manifest(
                    SPLITS / f"utkface_ratio_{ratio:.2f}_seed_{seed}.json", self.cohort
                )
                for ratio in expected
            }
            partitions = [item["splits"] for item in manifests.values()]
            self.assertTrue(all(partition == partitions[0] for partition in partitions[1:]))
            labeled = [set(manifests[ratio]["labeled_indices"]) for ratio in expected]
            self.assertLess(labeled[0], labeled[1])
            self.assertLess(labeled[1], labeled[2])
            for ratio, (n_labeled, n_unlabeled) in expected.items():
                self.assertEqual(manifests[ratio]["counts"]["labeled"], n_labeled)
                self.assertEqual(manifests[ratio]["counts"]["unlabeled"], n_unlabeled)
                regenerated = generate_seed_manifest(self.cohort, seed, ratio)
                self.assertEqual(manifests[ratio], regenerated)


if __name__ == "__main__":
    unittest.main()
