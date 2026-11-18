#!/usr/bin/env python3
"""Formal canonical Uniform benchmark for protected CheXchoNet LVIDd."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from data_processing.chexchonet import CheXchoNetDataset, discover_metadata, load_records
from data_processing.chexchonet_protocol import (
    build_evaluation_transform,
    build_strong_transform,
    build_weak_transform,
    validate_manifest,
)
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor

METHOD = "canonical_uniform"
RESNET_ID = "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"
DINO_ID = "dinov2_vits14"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def construct(backbone, device):
    model = ResNet50Regressor(pretrained=True) if backbone == "resnet50" else DINOv2Regressor(size="small")
    return model.to(device)


def loader(dataset, batch_size, seed, role, workers, shuffle):
    generator = torch.Generator()
    generator.manual_seed(seed * 1009 + {"labeled": 11, "unlabeled": 23, "validation": 37, "test": 53}[role])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=False, generator=generator,
                      worker_init_fn=lambda worker_id: np.random.seed((torch.initial_seed() + worker_id) % 2**32))


def load_data(args):
    records = load_records(discover_metadata(args.data_root))
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, records)
    if manifest["seed"] != args.seed or not math.isclose(manifest["labeled_ratio"], args.ratio):
        raise RuntimeError("Manifest assignment mismatch")
    mean, std = manifest["label_scaler"]["mean"], manifest["label_scaler"]["std"]
    image_root = args.data_root / "images"
    weak, strong, evaluation = build_weak_transform(), build_strong_transform(), build_evaluation_transform()
    datasets = {
        "labeled": CheXchoNetDataset(records, image_root, manifest["indices"]["labeled"], mean, std, weak),
        "unlabeled": CheXchoNetDataset(records, image_root, manifest["indices"]["unlabeled"], mean, std, weak, strong),
        "validation": CheXchoNetDataset(records, image_root, manifest["indices"]["validation"], mean, std, evaluation),
        "test": CheXchoNetDataset(records, image_root, manifest["indices"]["test"], mean, std, evaluation),
    }
    loaders = {role: loader(dataset, args.batch_size, args.seed, role, args.num_workers,
                            role in ("labeled", "unlabeled")) for role, dataset in datasets.items()}
    return records, manifest, datasets, loaders, float(mean), float(std)


def optimize(model, backbone, epochs):
    if backbone == "dinov2_vits14":
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        config = {"optimizer": "AdamW", "backbone_lr": 1e-5, "head_lr": 1e-4,
                  "weight_decay": 0.05, "scheduler": "CosineAnnealingLR", "T_max": epochs}
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        config = {"optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-3,
                  "scheduler": "StepLR", "step_size": 10, "gamma": 0.1}
    return optimizer, scheduler, config


@torch.no_grad()
def evaluate(model, data_loader, mean, std, device, predictions=False):
    model.eval()
    predicted, targets, indices = [], [], []
    for images, target, index in data_loader:
        predicted.append((model(images.to(device)).float().cpu() * std + mean).numpy())
        targets.append((target.float() * std + mean).numpy())
        indices.append(index.numpy())
    predicted = np.concatenate(predicted)
    targets = np.concatenate(targets)
    indices = np.concatenate(indices)
    error = predicted - targets
    metrics = {
        "mae_cm": float(np.mean(np.abs(error))),
        "mae_mm": float(np.mean(np.abs(error)) * 10),
        "mse_cm2": float(np.mean(error**2)),
        "rmse_cm": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / (np.sum((targets - targets.mean())**2) + 1e-12)),
        "pearson": float(pearsonr(predicted, targets).statistic),
        "spearman": float(spearmanr(predicted, targets).statistic),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("Non-finite metric")
    return (metrics, predicted, targets, indices) if predictions else metrics


def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one CUDA device as cuda:0")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() or (out / "final_test_started").exists():
        raise FileExistsError("Refusing to reuse an evaluated attempt")

    seed_all(args.seed)
    device = torch.device("cuda:0")
    records, manifest, datasets, loaders, mean, std = load_data(args)
    model = construct(args.backbone, device)
    initialization_hash = state_hash(model)
    optimizer, scheduler, optimization = optimize(model, args.backbone, args.epochs)

    config = {
        "dataset": "CheXchoNet",
        "target": "LVIDd",
        "target_unit": "centimeters",
        "method": METHOD,
        "uniform_definition": {
            "pseudo_target": "stopgrad(f_theta(weak(x_u)))",
            "unlabeled_loss": "mean((f_theta(strong(x_u)) - pseudo_target)^2)",
            "r": 1.0,
            "objective": "L_sup + 1.0 * L_u",
            "supervised_recycling": "cycle(labeled_loader) paired with one traversal of unlabeled_loader per epoch",
        },
        "backbone": args.backbone,
        "pretrained_identifier": RESNET_ID if args.backbone == "resnet50" else DINO_ID,
        "seed": args.seed,
        "labeled_ratio": args.ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": "float32",
        "objective_space": "standardized-target MSE",
        "selection_metric": "lowest validation MAE in original centimeters",
        "optimization": optimization,
        "lambda_u": 1.0,
        "protocol_version": manifest["protocol_version"],
        "transform_version": manifest["transform_version"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "cohort_digest": manifest["cohort_digest"],
        "split_digest": manifest["split_digest"],
        "counts": manifest["counts"],
        "label_scaler": manifest["label_scaler"],
        "target_initialization_hash": initialization_hash,
    }
    write_json(out / "config.json", config)

    started = time.time()
    history = []
    best = math.inf
    best_epoch = None
    optimizer_steps = 0
    labeled_exposure = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses, supervised_losses, unlabeled_losses = [], [], []
        epoch_steps = 0
        for labeled, unlabeled in zip(cycle(loaders["labeled"]), loaders["unlabeled"]):
            xl, yl, _ = labeled
            weak, strong, _ = unlabeled
            xl, yl = xl.to(device), yl.to(device)
            weak, strong = weak.to(device), strong.to(device)
            labeled_exposure += len(yl)

            optimizer.zero_grad(set_to_none=True)
            supervised = F.mse_loss(model(xl), yl)
            with torch.no_grad():
                pseudo_target = model(weak).detach()
            strong_prediction = model(strong)
            unlabeled_loss = (strong_prediction - pseudo_target).pow(2).mean()
            loss = supervised + unlabeled_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            optimizer_steps += 1
            epoch_steps += 1
            total_losses.append(float(loss.detach()))
            supervised_losses.append(float(supervised.detach()))
            unlabeled_losses.append(float(unlabeled_loss.detach()))
        if epoch_steps != len(loaders["unlabeled"]):
            raise RuntimeError("Canonical unlabeled traversal length changed")

        scheduler.step()
        validation = evaluate(model, loaders["validation"], mean, std, device)
        improved = validation["mae_cm"] < best
        if improved:
            best = validation["mae_cm"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "validation_mae_cm": best,
                "manifest_sha256": manifest["manifest_sha256"],
                "scaler": manifest["label_scaler"],
                "config": config,
            }, out / "best.pt")
        history.append({
            "epoch": epoch,
            "train_total_mse": float(np.mean(total_losses)),
            "train_sup_mse": float(np.mean(supervised_losses)),
            "train_unlabeled_mse": float(np.mean(unlabeled_losses)),
            "validation_mae_cm": validation["mae_cm"],
            "validation_mse_cm2": validation["mse_cm2"],
            "validation_r2": validation["r2"],
            "best_so_far": int(improved),
            "optimizer_steps": optimizer_steps,
            "labeled_example_exposure": labeled_exposure,
            "elapsed_seconds": time.time() - started,
        })
        print(f"epoch={epoch}/{args.epochs} val_mae_cm={validation['mae_cm']:.6f} best_epoch={best_epoch}", flush=True)

    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    torch.save({"model_state": model.state_dict(), "epoch": args.epochs,
                "manifest_sha256": manifest["manifest_sha256"], "config": config}, out / "last.pt")

    restored = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(restored["model_state"])
    if restored["epoch"] != best_epoch or not math.isclose(restored["validation_mae_cm"], best, abs_tol=1e-12):
        raise RuntimeError("Validation-selected checkpoint identity mismatch")

    write_json(out / "final_test_started", {"started_unix": time.time(), "test_evaluations": 0})
    test, predictions, targets, indices = evaluate(model, loaders["test"], mean, std, device, True)
    aligned = indices.tolist() == manifest["indices"]["test"]
    if not aligned:
        raise RuntimeError("Test prediction order differs from frozen manifest")
    np.savez_compressed(out / "test_predictions.npz", cohort_indices=indices,
                        predictions_cm=predictions, targets_cm=targets)
    # Dataset-appropriate formal analysis snapshot; reuses the exactly-once test outputs.
    np.savez_compressed(out / "analysis_snapshot.npz", cohort_indices=indices,
                        predictions_cm=predictions, targets_cm=targets,
                        residuals_cm=predictions - targets)
    metrics = {
        "validation_mae": best,
        "best_epoch": best_epoch,
        "best_validation_mae_cm": best,
        "test_mae": test["mae_cm"],
        "test_r2": test["r2"],
        **{f"test_{key}": value for key, value in test.items()},
        "checkpoint_reloaded": True,
        "test_used_for_selection": False,
        "test_evaluations": 1,
        "test_time_calibration": False,
    }
    if not all(math.isfinite(float(metrics[key])) for key in ("validation_mae", "test_mae", "test_r2")):
        raise RuntimeError("Non-finite formal metric")
    write_json(out / "metrics.json", metrics)
    metadata = {
        "status": "complete",
        "dataset": "CheXchoNet",
        "target": "LVIDd",
        "target_unit": "centimeters",
        "method": METHOD,
        "backbone": args.backbone,
        "seed": args.seed,
        "labeled_ratio": args.ratio,
        "protocol_version": manifest["protocol_version"],
        "transform_version": manifest["transform_version"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "cohort_digest": manifest["cohort_digest"],
        "split_digest": manifest["split_digest"],
        "counts": manifest["counts"],
        "label_scaler": manifest["label_scaler"],
        "checkpoint_reloaded": True,
        "test_used_for_selection": False,
        "test_evaluations": 1,
        "test_model_inference_count": 1,
        "test_time_calibration": False,
        "restored_epoch_verified": True,
        "restored_validation_metric_verified": True,
        "ordered_prediction_alignment": True,
        "target_initialization_hash": initialization_hash,
        "optimizer_steps": optimizer_steps,
        "labeled_example_exposure_count": labeled_exposure,
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_local_device": "cuda:0",
    }
    write_json(out / "metadata.json", metadata)
    write_json(out / "final_test_complete", {"completed_unix": time.time(), "test_evaluations": 1})
    print(json.dumps(metrics, sort_keys=True), flush=True)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("resnet50", "dinov2_vits14"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
