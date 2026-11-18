#!/usr/bin/env python3
"""UTKFace 5% supervision-quantity controls for canonical Uniform SSR.

This campaign-local entrypoint changes only which canonical supervised and
unlabeled losses are present at each optimizer-update position.  It deliberately
keeps the frozen-probe initialization path used by the paper's Uniform control,
even though the probe does not affect either supervised-only objective.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from training.probe_filter import train_probe
from training.train import eval_mae, make_data_utkface, seed_all


EPOCHS = 30
RATIO = 0.05
BATCH_SIZE = 32
WORKERS = 4
MODES = ("sup_only", "sup_only_recycled")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_args(a):
    a.dataset = "utkface"
    a.method = "probe"
    a.data_dir = "/nobackup/enzez/Representation-Anchored-Pseudo-Labeling/data/utkface_all"
    a.labeled_ratio = RATIO
    a.img_size = 224
    a.batch_size = BATCH_SIZE
    a.workers = WORKERS
    a.pretrained = True
    a.probe_backbone = a.backbone
    a.dino = "s"
    return a


def build_runtime(backbone: str, device: torch.device):
    if backbone == "resnet50":
        model = ResNet50Regressor(pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif backbone == "dinov2":
        model = DINOv2Regressor(size="small").to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        raise ValueError(backbone)
    return model, optimizer, scheduler


def initialize_like_canonical(backbone: str, labeled_loader, device: torch.device):
    # This order exactly follows the recovered paper +Reorient source:
    # frozen backbone -> full labeled probe pass -> probe construction -> trainable model.
    if backbone == "resnet50":
        frozen = ResNet50Regressor(pretrained=True).to(device)
    else:
        frozen = DINOv2Regressor(size="small").to(device)
    frozen.eval()
    for parameter in frozen.backbone.parameters():
        parameter.requires_grad_(False)
    probe = train_probe(frozen.backbone, labeled_loader, device)
    probe = probe.to(device)
    model, optimizer, scheduler = build_runtime(backbone, device)
    return frozen, probe, model, optimizer, scheduler


def cycle_forever(loader):
    while True:
        for batch in loader:
            yield batch


def schedule_counts(mode: str, n_labeled: int, n_unlabeled: int) -> dict:
    if mode == "sup_only":
        return {"labeled_batches": n_labeled, "unlabeled_batches": 0,
                "optimizer_steps": n_labeled, "recycled_anchors": 0}
    if mode == "sup_only_recycled":
        # Canonical k=1 has U total joint positions, not L native + U joint positions.
        return {"labeled_batches": n_unlabeled, "unlabeled_batches": 0,
                "optimizer_steps": n_unlabeled,
                "recycled_anchors": max(0, n_unlabeled - n_labeled)}
    raise ValueError(mode)


def dry_count(a) -> None:
    seed_all(a.seed)
    a = data_args(a)
    labeled, unlabeled, *_ = make_data_utkface(a)
    result = {mode: schedule_counts(mode, len(labeled), len(unlabeled)) for mode in MODES}
    result.update({"N_L": len(labeled), "N_U": len(unlabeled), "pass": len(labeled) == 29 and len(unlabeled) == 563})
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise RuntimeError(result)


def train(a) -> None:
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(a.seed)
    device = torch.device("cuda")
    a = data_args(a)
    labeled, unlabeled, validation, test, mean, std = make_data_utkface(a)
    n_labeled, n_unlabeled = len(labeled), len(unlabeled)
    if (n_labeled, n_unlabeled) != (29, 563):
        raise RuntimeError(f"unexpected loader lengths {(n_labeled, n_unlabeled)}")
    counts = schedule_counts(a.training_mode, n_labeled, n_unlabeled)

    frozen, probe, model, optimizer, scheduler = initialize_like_canonical(a.backbone, labeled, device)
    del frozen, probe
    initial_hash = hashlib.sha256(b"".join(
        tensor.detach().cpu().contiguous().numpy().tobytes() for tensor in model.state_dict().values()
    )).hexdigest()

    config = {
        "dataset": "UTKFace", "labeled_ratio": RATIO, "backbone": a.backbone,
        "seed": a.seed, "condition": a.training_mode, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "workers": WORKERS, "lambda_u": 1.0,
        "unlabeled_loss_present": False, "reorient_active": False,
        "manifest": str(Path(a.benchmark_manifest).resolve()),
        "manifest_sha256": file_sha256(Path(a.benchmark_manifest)),
        "cohort_sha256": a.benchmark_metadata["cohort_sha256"],
        "target_scaler": {"mean": mean, "std": std},
        "loader_lengths": {"labeled": n_labeled, "unlabeled": n_unlabeled},
        "per_epoch_contract": counts,
        "labeled_iterator_policy": ("one fresh finite pass per epoch" if a.training_mode == "sup_only"
                                    else "persistent canonical cycle across epochs"),
        "canonical_schedule_correction": "paper k=1 uses U=563 total positions; no separate L=29 pre-pass",
        "model_initialization_sha256": initial_hash,
        "optimizer": ("Adam(lr=1e-4,weight_decay=1e-3)" if a.backbone == "resnet50"
                      else "AdamW(backbone_lr=1e-5,head_lr=1e-4,weight_decay=0.05)"),
        "scheduler": ("StepLR(step_size=10,gamma=0.1)" if a.backbone == "resnet50"
                      else "CosineAnnealingLR(T_max=30)"),
        "gradient_clipping_max_norm": 5.0,
        "selection": "minimum validation MAE only", "final_test_evaluations": 1,
        "resume": False,
    }
    atomic_json(out / "config.json", config)
    atomic_json(out / "status.json", {"training": "running", "final_test": "pending"})

    best = float("inf")
    best_epoch = None
    history = []
    recycled_iterator = cycle_forever(labeled) if a.training_mode == "sup_only_recycled" else None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        iterator = iter(labeled) if a.training_mode == "sup_only" else recycled_iterator
        steps = counts["optimizer_steps"]
        total_loss = 0.0
        consumed = 0
        for _ in range(steps):
            x, y = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = F.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            consumed += 1
        if consumed != counts["labeled_batches"]:
            raise AssertionError((consumed, counts))
        scheduler.step()
        val_mae, val_r2 = eval_mae(model, validation, mean, std, device)
        improved = val_mae < best
        if improved:
            best, best_epoch = val_mae, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation_mae": val_mae, "config": config}, out / "best.pt")
        history.append({
            "epoch": epoch, "loss_supervised": total_loss / steps,
            "validation_mae": val_mae, "validation_r2": val_r2,
            "labeled_batches_consumed": consumed, "unlabeled_batches_consumed": 0,
            "optimizer_steps": steps, "recycled_anchor_count": counts["recycled_anchors"],
            "learning_rates": json.dumps([group["lr"] for group in optimizer.param_groups]),
        })
        with (out / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
        print(json.dumps({"epoch": epoch, "validation_mae": val_mae,
                          "best_epoch": best_epoch, "finite": math.isfinite(total_loss)}), flush=True)

    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    marker = out / "final_test_started.json"
    if marker.exists() or (out / "final_test_complete.json").exists():
        raise RuntimeError("refusing duplicate final test")
    atomic_json(marker, {"started_unix": time.time(), "test_evaluations": 0})
    test_mae, test_r2, predictions, targets = eval_mae(model, test, mean, std, device, return_predictions=True)
    np.savez_compressed(out / "test_predictions.npz", predictions_years=predictions, targets_years=targets)
    metrics = {"best_epoch": best_epoch, "best_validation_mae": best,
               "test_mae": test_mae, "test_r2": test_r2,
               "checkpoint_reloaded": True, "test_used_for_selection": False,
               "test_evaluations": 1}
    atomic_json(out / "metrics.json", metrics)
    atomic_json(out / "metadata.json", {
        "status": "complete", "method": "plain_supervision_quantity_control",
        **config, **metrics, "completed_unix": time.time(),
    })
    atomic_json(out / "final_test_complete.json", {"completed_unix": time.time(), "test_evaluations": 1})
    atomic_json(out / "status.json", {"training": "complete", "final_test": "complete"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-mode", choices=MODES, default="sup_only")
    parser.add_argument("--backbone", choices=("resnet50", "dinov2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-count", action="store_true")
    args = parser.parse_args()
    if args.dry_count:
        dry_count(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
