#!/usr/bin/env python3
"""Lightweight data/model preflight for the ResNet-50 UTKFace sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.utkface_data import BenchmarkContext, TupleDataset
from data_processing.utkface_protocol import (
    dataloader_generator, inverse_normalize_age, normalize_age, seed_dataloader_worker,
)

WEIGHT_ID = "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"


def loader(dataset, seed, role, shuffle, drop_last):
    return DataLoader(dataset, batch_size=32, shuffle=shuffle, drop_last=drop_last,
                      num_workers=4, worker_init_fn=seed_dataloader_worker,
                      generator=dataloader_generator(seed, role))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("rapl", "hpl"), required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.current_device() == 0
    manifest = ROOT / "data_processing" / "splits" / f"utkface_ratio_{args.ratio:.2f}_seed_{args.seed}.json"
    context = BenchmarkContext(ROOT, ROOT / "data" / "utkface_all", manifest)
    expected = {0.10: (1896, 17073), 0.20: (3793, 15176)}[args.ratio]
    assert context.cohort["cohort_sha256"] == "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
    assert (context.manifest["counts"]["labeled"], context.manifest["counts"]["unlabeled"]) == expected
    datasets = {
        "labeled": TupleDataset(context, "labeled", "labeled", repeat=2 if args.method == "hpl" else 1),
        "unlabeled": TupleDataset(context, "unlabeled", "weak_strong"),
        "validation": TupleDataset(context, "validation", "evaluation"),
        "test": TupleDataset(context, "test", "evaluation"),
    }
    batches = {role: next(iter(loader(ds, args.seed, role, role in ("labeled", "unlabeled"),
                                            role in ("labeled", "unlabeled"))))
               for role, ds in datasets.items()}
    age = torch.tensor([0.0, 37.0, 120.0])
    restored = inverse_normalize_age(normalize_age(age, context.mean, context.std), context.mean, context.std)
    assert torch.allclose(age, restored, atol=1e-5)

    if args.method == "rapl":
        from models.backbone import ResNet50Regressor
        target = ResNet50Regressor(pretrained=True).cuda()
        frozen = ResNet50Regressor(pretrained=True).cuda()
        for parameter in frozen.backbone.parameters():
            parameter.requires_grad_(False)
        assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())
        with torch.no_grad():
            assert target(batches["labeled"][0].cuda()).shape[0] == 32
            assert frozen.backbone(batches["labeled"][0].cuda()).shape == (32, 2048)
        model = "RAPL target and separately instantiated frozen probe ResNet-50"
    else:
        official = ROOT / "third_party" / "Heteroscedastic-Pseudo-Labels" / "utkface"
        sys.path.insert(0, str(official))
        from models import UncertaintyLearner, resnet50_unc
        target = resnet50_unc(pretrained=True, drp_p=0.05).cuda()
        uncertainty = UncertaintyLearner(input_dim=2, output_dim=1).cuda()
        with torch.no_grad():
            prediction, features = target(batches["labeled"][0].cuda())
            assert prediction.shape[0] == 32 and features.shape == (32, 2048)
            assert uncertainty(torch.zeros(32, 2, device="cuda")).shape[0] == 32
        model = "official HPL ResNet-50 regression model and uncertainty learner"
    assert all(math.isfinite(float(value)) for value in (context.mean, context.std))
    print(json.dumps({"status": "passed", "method": args.method, "ratio": args.ratio,
                      "seed": args.seed, "manifest": str(manifest), "counts": context.manifest["counts"],
                      "scaler": {"mean": context.mean, "std": context.std},
                      "weight_identifier": WEIGHT_ID, "model": model,
                      "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                      "process_local_device": "cuda:0",
                      "batch_shapes": {key: [list(x.shape) if hasattr(x, "shape") else None for x in value]
                                       for key, value in batches.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
