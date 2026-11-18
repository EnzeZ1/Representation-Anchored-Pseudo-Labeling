"""Pure supervised UTKFace benchmark using the shared formal protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_processing.utkface_protocol import (
    build_evaluation_transform,
    build_labeled_transform,
    dataloader_generator,
    inverse_normalize_age,
    load_cohort,
    load_seed_manifest,
    loader_metadata,
    manifest_items,
    runtime_metadata,
    seed_dataloader_worker,
)
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor


ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data_processing/splits/utkface_cohort_v1.json"
COHORT_SHA256 = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
IMDB_COHORT_PATH = ROOT / "data_processing/splits/imdb_wiki_dir_cohort_v1.json"
IMDB_COHORT_SHA256 = "919fe3e1b959e1fe75e08e83310a84c1c3a9d53a16812a1bb5f1e0117ba97f43"
DINO_CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


class LabeledImages(Dataset):
    def __init__(self, items, transform, mean: float, std: float, indices):
        self.items = items
        self.transform = transform
        self.mean = mean
        self.std = std
        self.indices = list(indices)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, age = self.items[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        target = torch.tensor((age - self.mean) / self.std, dtype=torch.float32)
        return image, target, self.indices[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("utkface", "imdb_wiki"), default="utkface")
    parser.add_argument("--backbone", choices=("resnet50", "dinov2_vits14"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/utkface_all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def load_protocol(args):
    if args.dataset == "imdb_wiki":
        from data_processing.imdb_wiki_protocol import (
            build_evaluation_transform as evaluation_transform,
            build_labeled_transform as labeled_transform,
            load_cohort as protocol_load_cohort,
            load_seed_manifest as protocol_load_manifest,
            manifest_items as protocol_manifest_items,
        )
        cohort_path, expected_digest = IMDB_COHORT_PATH, IMDB_COHORT_SHA256
    else:
        evaluation_transform = build_evaluation_transform
        labeled_transform = build_labeled_transform
        protocol_load_cohort = load_cohort
        protocol_load_manifest = load_seed_manifest
        protocol_manifest_items = manifest_items
        cohort_path, expected_digest = COHORT_PATH, COHORT_SHA256
    cohort = protocol_load_cohort(cohort_path)
    if cohort["cohort_sha256"] != expected_digest:
        raise ValueError("Unexpected canonical cohort digest")
    manifest = protocol_load_manifest(args.manifest, cohort)
    if int(manifest["seed"]) != args.seed:
        raise ValueError("Manifest seed does not match --seed")
    mean = float(manifest["label_scaler"]["mean"])
    std = float(manifest["label_scaler"]["std"])
    labeled_indices = manifest["labeled_indices"]
    datasets = {
        "labeled": LabeledImages(
            protocol_manifest_items(cohort, labeled_indices, args.data_root),
            labeled_transform(), mean, std, labeled_indices,
        ),
        "validation": LabeledImages(
            protocol_manifest_items(cohort, manifest["splits"]["validation"], args.data_root),
            evaluation_transform(), mean, std, manifest["splits"]["validation"],
        ),
        "test": LabeledImages(
            protocol_manifest_items(cohort, manifest["splits"]["test"], args.data_root),
            evaluation_transform(), mean, std, manifest["splits"]["test"],
        ),
    }
    loaders, loader_info = {}, {}
    for role, dataset in datasets.items():
        shuffle = role == "labeled"
        drop_last = role == "labeled"
        loaders[role] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=drop_last,
            worker_init_fn=seed_dataloader_worker,
            generator=dataloader_generator(args.seed, role),
        )
        loader_info[role] = loader_metadata(
            seed=args.seed, role=role, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=shuffle, drop_last=drop_last,
            sampler="RandomSampler" if shuffle else "SequentialSampler",
            pin_memory=True,
        )
    protocol = {
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort_path": str(cohort_path),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "protocol_version": manifest["protocol_version"],
        "transform_version": manifest["transform_version"],
        "seed": args.seed,
        "labeled_ratio": float(manifest["labeled_ratio"]),
        "counts": manifest["counts"],
        "label_scaler": manifest["label_scaler"],
        "dataloaders": loader_info,
    }
    return cohort, manifest, loaders, mean, std, protocol


def construct_model(backbone: str, device):
    if backbone == "resnet50":
        model = ResNet50Regressor(pretrained=True).to(device)
        identity = {
            "model": "torchvision ResNet-50",
            "feature_dimension": 2048,
            "pretrained_identifier": "torchvision.ResNet50_Weights.IMAGENET1K_V1",
            "pretrained_url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            "checkpoint_identifier": "filename-sha256-prefix:0676ba61",
        }
    else:
        model = DINOv2Regressor(size="small").to(device)
        identity = {
            "model": model.model_name,
            "feature_dimension": model.feature_dim,
            "pretrained_identifier": model.weight_identifier,
            "pretrained_url": model.weight_url,
            "checkpoint_sha256": DINO_CHECKPOINT_SHA256,
        }
    return model, identity


def optimizer_and_scheduler(model, backbone: str, epochs: int):
    if backbone == "resnet50":
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {"params": model.head.parameters(), "lr": 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        config = {
            "optimizer": "Adam", "backbone_lr": 1e-4, "head_lr": 1e-3,
            "weight_decay": 0.0, "scheduler": "StepLR(step_size=10,gamma=0.1)",
        }
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        config = {
            "optimizer": "AdamW", "backbone_lr": 1e-5, "head_lr": 1e-4,
            "weight_decay": 0.05, "scheduler": f"CosineAnnealingLR(T_max={epochs})",
        }
    return optimizer, scheduler, config


@torch.no_grad()
def evaluate(model, loader, mean, std, device, predictions=False):
    model.eval()
    pred_values, targets, indices = [], [], []
    for images, target, cohort_index in loader:
        prediction = model(images.to(device, non_blocking=True)).float().cpu()
        pred_values.append(inverse_normalize_age(prediction, mean, std))
        targets.append(inverse_normalize_age(target.float(), mean, std))
        indices.append(cohort_index.long())
    pred = torch.cat(pred_values).numpy()
    truth = torch.cat(targets).numpy()
    cohort_indices = torch.cat(indices).numpy()
    mae = float(np.mean(np.abs(pred - truth)))
    residual = float(np.sum((pred - truth) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1.0 - residual / (total + 1e-12))
    if not np.isfinite([mae, r2]).all():
        raise RuntimeError("Non-finite evaluation metric")
    return (mae, r2, pred, truth, cohort_indices) if predictions else (mae, r2)


def run_preflight(args):
    seed_everything(args.seed)
    _, _, loaders, mean, std, protocol = load_protocol(args)
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight process must see exactly one CUDA device")
    model, identity = construct_model(args.backbone, device)
    optimizer, _, optimizer_config = optimizer_and_scheduler(model, args.backbone, args.epochs)
    model.train()
    images, target, _ = next(iter(loaders["labeled"]))
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(images.to(device)), target.to(device))
    loss.backward()
    optimizer.step()
    model.eval()
    images, target, _ = next(iter(loaders["validation"]))
    with torch.no_grad():
        pred = inverse_normalize_age(model(images.to(device)).cpu(), mean, std)
    truth = inverse_normalize_age(target, mean, std)
    mae = float((pred - truth).abs().mean())
    residual = float(((pred - truth) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    print(json.dumps({
        "status": "pass", "backbone": args.backbone, "model": identity,
        "optimizer": optimizer_config, "train_sup_mse": float(loss),
        "validation_batch_mae_years": mae,
        "validation_batch_r2": 1.0 - residual / (total + 1e-12),
        "protocol": protocol, "test_inference_count": 0,
    }, sort_keys=True))


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.preflight:
        run_preflight(args)
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required for a formal run")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("A formal worker must see exactly one local CUDA device")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")

    started = time.time()
    seed_everything(args.seed)
    cohort, manifest, loaders, mean, std, protocol = load_protocol(args)
    device = torch.device("cuda:0")
    model, model_identity = construct_model(args.backbone, device)
    # CUDA 12.4/PyTorch 2.6 requires a context before resetting peak counters.
    torch.cuda.reset_peak_memory_stats(device)
    optimizer, scheduler, optimization = optimizer_and_scheduler(model, args.backbone, args.epochs)
    configuration = {
        "experiment": "pure_supervised_reference",
        "dataset": "IMDB-WIKI-DIR" if args.dataset == "imdb_wiki" else "UTKFace",
        "backbone": args.backbone,
        "objective": "MSE(prediction, normalized_true_label)",
        "unlabeled_objective": None, "epochs": args.epochs,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "precision": "float32", "seed": args.seed,
        "labeled_ratio": float(manifest["labeled_ratio"]),
        "model": model_identity, "optimization": optimization, "protocol": protocol,
    }
    json_write(output / "config.json", configuration)

    best_mae, best_epoch = float("inf"), None
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, target, _ in loaders["labeled"]:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(images), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            example_count += len(target)
        validation_mae, validation_r2 = evaluate(model, loaders["validation"], mean, std, device)
        improved = validation_mae < best_mae
        if improved:
            best_mae, best_epoch = validation_mae, epoch
            torch.save({
                "checkpoint_version": "utkface-supervised-v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "validation_mae_years": validation_mae,
                "validation_r2_years": validation_r2,
                "configuration": configuration,
                "label_scaler": {"mean": mean, "std": std},
                "manifest_sha256": protocol["manifest_sha256"],
                "cohort_sha256": protocol["cohort_sha256"],
            }, output / "best.pt")
        history.append({
            "epoch": epoch, "train_sup_mse": loss_sum / example_count,
            "validation_mae_years": validation_mae,
            "validation_r2_years": validation_r2,
            "backbone_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
            "best_so_far": int(improved),
            "elapsed_seconds": time.time() - epoch_started,
        })
        print(
            f"epoch={epoch}/{args.epochs} train_sup_mse={loss_sum/example_count:.8f} "
            f"validation_mae_years={validation_mae:.6f} validation_r2={validation_r2:.6f} "
            f"best_epoch={best_epoch}", flush=True,
        )
        scheduler.step()

    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if checkpoint["epoch"] != best_epoch or not np.isclose(
        checkpoint["validation_mae_years"], best_mae, rtol=0.0, atol=1e-10
    ):
        raise RuntimeError("Restored checkpoint metadata does not match recorded best")
    restored_mae, _ = evaluate(model, loaders["validation"], mean, std, device)
    if not np.isclose(restored_mae, best_mae, rtol=0.0, atol=1e-5):
        raise RuntimeError("Restored model validation MAE does not match recorded best")

    # This is the sole test-model inference in a formal run.
    test_mae, test_r2, prediction, target, cohort_indices = evaluate(
        model, loaders["test"], mean, std, device, predictions=True
    )
    relative_paths = np.asarray(
        [cohort["records"][int(index)]["path"] for index in cohort_indices]
    )
    np.savez_compressed(
        output / "test_predictions.npz", cohort_indices=cohort_indices,
        relative_image_identifiers=relative_paths,
        predictions_years=prediction, targets_years=target,
    )
    runtime_seconds = time.time() - started
    metrics = {
        "best_epoch": best_epoch, "best_validation_mae_years": best_mae,
        "test_mae_years": test_mae, "test_r2": test_r2,
        "checkpoint_reloaded": True, "test_used_for_selection": False,
        "test_model_inference_count": 1,
    }
    metadata = {
        **protocol,
        "dataset": "IMDB-WIKI-DIR" if args.dataset == "imdb_wiki" else "UTKFace",
        "method": "supervised",
        "backbone": args.backbone, "model": model_identity,
        "objective": "supervised MSE only", "ssl_components": [],
        "checkpoint_path": str((output / "best.pt").resolve()),
        "checkpoint_size_bytes": (output / "best.pt").stat().st_size,
        "checkpoint_selection": "lowest validation MAE after inverse normalization to years",
        "checkpoint_reloaded": True, "test_used_for_selection": False,
        "test_model_inference_count": 1, "restored_epoch_verified": True,
        "restored_validation_mae_verified": True, "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "physical_gpu": os.environ.get("SUPERVISED_PHYSICAL_GPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_local_device": "cuda:0", "runtime": runtime_metadata(),
    }
    json_write(output / "metrics.json", metrics)
    json_write(output / "metadata.json", metadata)
    print(json.dumps({"status": "complete", **metrics}), flush=True)


if __name__ == "__main__":
    main()
