"""Pure supervised STS-B-DIR benchmark using the shared formal protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_processing.stsb import (
    STSBTextDataset,
    dataloader_generator,
    file_sha256,
    load_cohort,
    load_manifest,
    loader_metadata,
    seed_dataloader_worker,
    simple_tokenize,
)
from models.roberta_backbone import ROBERTA_BASE_IDENTIFIER, RobertaPairRegressor
from models.stsb_bilstm import BiLSTMPairRegressor

ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / "data_processing/splits/stsb_dir_cohort_v1.json"
GLOVE_CACHE = ROOT / "data/glove/stsb_glove_840b_300d.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_metadata() -> dict:
    import platform
    import transformers
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
    }


class BiLSTMCollator:
    def __init__(self, vocabulary: list[str], padding_index: int, unknown_index: int):
        self.indices = {token: index for index, token in enumerate(vocabulary)}
        self.padding_index = padding_index
        self.unknown_index = unknown_index

    def encode(self, sentence: str) -> torch.Tensor:
        return torch.tensor(
            [self.indices.get(token, self.unknown_index) for token in simple_tokenize(sentence)],
            dtype=torch.long,
        )

    def __call__(self, examples):
        first = [self.encode(example["sentence1"]) for example in examples]
        second = [self.encode(example["sentence2"]) for example in examples]
        first = pad_sequence(first, batch_first=True, padding_value=self.padding_index)
        second = pad_sequence(second, batch_first=True, padding_value=self.padding_index)
        return {
            "sentence1": first,
            "sentence2": second,
            "mask1": first.ne(self.padding_index),
            "mask2": second.ne(self.padding_index),
            "target": torch.stack([example["target"] for example in examples]),
            "cohort_index": torch.tensor([example["cohort_index"] for example in examples]),
            "stable_id": [example["stable_id"] for example in examples],
        }


class RobertaCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        encoded = self.tokenizer(
            [example["sentence1"] for example in examples],
            [example["sentence2"] for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            **encoded,
            "target": torch.stack([example["target"] for example in examples]),
            "cohort_index": torch.tensor([example["cohort_index"] for example in examples]),
            "stable_id": [example["stable_id"] for example in examples],
        }


def model_forward(model, batch, backbone):
    if backbone == "bilstm_glove":
        return model(batch["sentence1"], batch["sentence2"], batch["mask1"], batch["mask2"])
    return model(batch["input_ids"], batch["attention_mask"])


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def construct(backbone: str, device):
    if backbone == "bilstm_glove":
        cache = torch.load(GLOVE_CACHE, map_location="cpu", weights_only=True)
        collator = BiLSTMCollator(
            cache["vocabulary"], cache["padding_index"], cache["unknown_index"]
        )
        model = BiLSTMPairRegressor(
            cache["embeddings"], cache["padding_index"],
            hidden_size=1024, num_layers=2, dropout=0.2, train_embeddings=False,
        ).to(device)
        identity = {
            "model": "HPL-compatible two-layer BiLSTM",
            "glove_identifier": cache["glove_identifier"],
            "embedding_dimension": 300,
            "hidden_dimension": 1024,
            "representation_dimension": 8192,
            "max_sequence_length": 40,
            "word_embeddings_trainable": False,
        }
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE_IDENTIFIER)
        collator = RobertaCollator(tokenizer, max_length=128)
        model = RobertaPairRegressor(ROBERTA_BASE_IDENTIFIER).to(device)
        identity = {
            "model": "RoBERTa-base pair regressor",
            "pretrained_identifier": ROBERTA_BASE_IDENTIFIER,
            "representation_dimension": model.feature_dim,
            "max_sequence_length": 128,
        }
    return model, collator, identity


def optimization(model, backbone: str, epochs: int, steps_per_epoch: int):
    if backbone == "bilstm_glove":
        encoder = [parameter for name, parameter in model.named_parameters()
                   if not name.startswith("head.") and parameter.requires_grad]
        optimizer = torch.optim.Adam([
            {"params": encoder, "lr": 1e-4},
            {"params": model.head.parameters(), "lr": 1e-3},
        ], weight_decay=1e-5)
        scheduler = None
        config = {
            "optimizer": "Adam", "encoder_lr": 1e-4, "head_lr": 1e-3,
            "weight_decay": 1e-5, "scheduler": None,
        }
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ], weight_decay=0.01)
        from transformers import get_linear_schedule_with_warmup
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(total_steps * 0.10)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        config = {
            "optimizer": "AdamW", "encoder_lr": 2e-5, "head_lr": 1e-4,
            "weight_decay": 0.01, "scheduler": "linear", "warmup_ratio": 0.10,
            "gradient_clipping": 1.0,
        }
    return optimizer, scheduler, config


def load_protocol(args, collator):
    cohort = load_cohort(COHORT_PATH)
    manifest = load_manifest(args.manifest, cohort)
    if int(manifest["seed"]) != args.seed:
        raise ValueError("Manifest seed differs from --seed")
    mean = float(manifest["label_scaler"]["mean"])
    std = float(manifest["label_scaler"]["std"])
    datasets = {
        "labeled": STSBTextDataset(cohort, manifest["labeled_indices"], mean, std),
        "validation": STSBTextDataset(cohort, manifest["splits"]["validation"], mean, std),
        "test": STSBTextDataset(cohort, manifest["splits"]["test"], mean, std),
    }
    loaders, loader_info = {}, {}
    for role, dataset in datasets.items():
        shuffle = role == "labeled"
        loaders[role] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
            pin_memory=True, drop_last=False, collate_fn=collator,
            worker_init_fn=seed_dataloader_worker,
            generator=dataloader_generator(args.seed, role),
        )
        loader_info[role] = loader_metadata(
            seed=args.seed, role=role, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=shuffle, drop_last=False,
            sampler="RandomSampler" if shuffle else "SequentialSampler", pin_memory=True,
        )
    protocol = {
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort_path": str(COHORT_PATH),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_payload_sha256": manifest["manifest_sha256"],
        "protocol_version": manifest["protocol_version"],
        "augmentation_version": manifest["augmentation_version"],
        "counts": manifest["counts"],
        "label_scaler": manifest["label_scaler"],
        "dataloaders": loader_info,
    }
    return cohort, manifest, loaders, mean, std, protocol


@torch.no_grad()
def evaluate(model, loader, mean, std, device, backbone, predictions=False):
    model.eval()
    values, targets, indices, identifiers = [], [], [], []
    for batch in loader:
        identifiers.extend(batch["stable_id"])
        batch = to_device(batch, device)
        prediction = model_forward(model, batch, backbone).float().cpu() * std + mean
        target = batch["target"].float().cpu() * std + mean
        values.append(prediction)
        targets.append(target)
        indices.append(batch["cohort_index"].cpu())
    prediction = torch.cat(values).numpy()
    target = torch.cat(targets).numpy()
    cohort_indices = torch.cat(indices).numpy()
    mse = float(np.mean((prediction - target) ** 2))
    mae = float(np.mean(np.abs(prediction - target)))
    r2 = float(1.0 - np.sum((prediction - target) ** 2)
               / (np.sum((target - target.mean()) ** 2) + 1e-12))
    if not np.isfinite([mse, mae, r2]).all():
        raise RuntimeError("Non-finite STS-B metric")
    result = (mse, mae, r2)
    return result + (prediction, target, cohort_indices, np.asarray(identifiers)) if predictions else result


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("bilstm_glove", "roberta_base"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 200 if args.backbone == "bilstm_glove" else 10
    if args.batch_size is None:
        args.batch_size = 32 if args.backbone == "bilstm_glove" else 16
    return args


def run_preflight(args):
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model, collator, identity = construct(args.backbone, device)
    _, _, loaders, mean, std, protocol = load_protocol(args, collator)
    optimizer, scheduler, config = optimization(
        model, args.backbone, args.epochs, len(loaders["labeled"])
    )
    batch = to_device(next(iter(loaders["labeled"])), device)
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model_forward(model, batch, args.backbone), batch["target"])
    loss.backward()
    if args.backbone == "roberta_base":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler:
        scheduler.step()
    validation = to_device(next(iter(loaders["validation"])), device)
    with torch.no_grad():
        original_prediction = model_forward(model, validation, args.backbone).cpu() * std + mean
    original_target = validation["target"].cpu() * std + mean
    mse = float(F.mse_loss(original_prediction, original_target))
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "preflight.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 0, "validation_mse": mse}, checkpoint)
        restored = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(restored["model_state"])
    print(json.dumps({
        "status": "pass", "backbone": args.backbone, "identity": identity,
        "optimization": config, "train_sup_mse": float(loss), "validation_batch_mse": mse,
        "protocol": protocol, "test_model_inference_count": 0,
    }, sort_keys=True))


def main():
    args = arguments()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("STS-B worker must see exactly one CUDA device as cuda:0")
    if args.preflight:
        run_preflight(args)
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = time.time()
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model, collator, identity = construct(args.backbone, device)
    cohort, manifest, loaders, mean, std, protocol = load_protocol(args, collator)
    optimizer, scheduler, optimizer_config = optimization(
        model, args.backbone, args.epochs, len(loaders["labeled"])
    )
    config = {
        "experiment": "pure_supervised_stsb_reference",
        "dataset": "STS-B-DIR",
        "backbone": args.backbone,
        "objective": "MSE(prediction, normalized_true_score)",
        "unlabeled_objective": None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": "float32",
        "selection_metric": "lowest validation MSE in original STS-B score units",
        "model": identity,
        "optimization": optimizer_config,
        "protocol": protocol,
    }
    write_json(output / "config.json", config)
    history = []
    best_mse = math.inf
    best_epoch = None
    checkpoint = output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["labeled"]:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model_forward(model, batch, args.backbone), batch["target"])
            loss.backward()
            if args.backbone == "roberta_base":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(float(loss.detach()))
        validation_mse, validation_mae, validation_r2 = evaluate(
            model, loaders["validation"], mean, std, device, args.backbone
        )
        improved = validation_mse < best_mse
        if improved:
            best_mse, best_epoch = validation_mse, epoch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "validation_mse": validation_mse,
                "validation_mae": validation_mae,
                "manifest_sha256": protocol["manifest_sha256"],
                "scaler": {"mean": mean, "std": std},
                "config": config,
            }, checkpoint)
        history.append({
            "epoch": epoch,
            "train_sup_mse": float(np.mean(losses)),
            "validation_mse": validation_mse,
            "validation_mae": validation_mae,
            "validation_r2": validation_r2,
            "learning_rates": json.dumps([group["lr"] for group in optimizer.param_groups]),
            "best_so_far": improved,
            "elapsed_seconds": time.time() - started,
        })
        print(
            f"epoch={epoch}/{args.epochs} train_sup_mse={np.mean(losses):.8f} "
            f"validation_mse={validation_mse:.6f} validation_r2={validation_r2:.6f} "
            f"best_epoch={best_epoch}", flush=True,
        )
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(restored["model_state"])
    if restored["epoch"] != best_epoch or not math.isclose(
        float(restored["validation_mse"]), best_mse, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError("Restored STS-B checkpoint metadata mismatch")
    test = evaluate(model, loaders["test"], mean, std, device, args.backbone, predictions=True)
    test_mse, test_mae, test_r2, prediction, target, indices, identifiers = test
    np.savez_compressed(
        output / "test_predictions.npz",
        cohort_indices=indices,
        stable_identifiers=identifiers,
        predictions_score_units=prediction,
        targets_score_units=target,
    )
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_mse_score_units": best_mse,
        "test_mse_score_units": test_mse,
        "test_mae_score_units": test_mae,
        "test_r2": test_r2,
        "checkpoint_reloaded": True,
        "test_used_for_selection": False,
        "test_model_inference_count": 1,
    }
    write_json(output / "metrics.json", metrics)
    metadata = {
        "status": "complete",
        "method": "supervised",
        "dataset": "STS-B-DIR",
        "backbone": args.backbone,
        "seed": args.seed,
        "labeled_ratio": manifest["labeled_ratio"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_selection": "lowest validation MSE in original STS-B score units",
        "checkpoint_reloaded": True,
        "restored_epoch_verified": True,
        "restored_validation_metric_verified": True,
        "test_used_for_selection": False,
        "test_model_inference_count": 1,
        "cohort_sha256": cohort["cohort_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "counts": manifest["counts"],
        "label_scaler": manifest["label_scaler"],
        "runtime_seconds": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_local_device": "cuda:0",
        "runtime": runtime_metadata(),
    }
    write_json(output / "metadata.json", metadata)
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
