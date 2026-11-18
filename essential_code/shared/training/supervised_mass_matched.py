"""UTKFace 5% supervision-mass-matched Uniform SSR experiment.

This isolated entrypoint inherits the canonical Uniform/+Interaction recipe.
The only experimental factor is whether 29 units of supervised coefficient
mass are placed at 29 evenly spaced steps or spread across all 563 steps.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from training.canonical_uniform_frequency import (
    DIAGNOSTIC_EPOCHS,
    EPOCHS,
    args_for_data,
    atomic_json,
    build_runtime,
    clip_and_step,
    evaluate,
    next_labeled,
    runtime_description,
    seed_all,
    uniform_losses,
)
from training.supervised_spacing_decoupling import anchor_steps
from training.train import make_data_utkface


CONDITIONS = ("even_29", "dense_mass_matched")
EXPECTED_U = 563
EXPECTED_L = 29
ALPHA = EXPECTED_L / EXPECTED_U


def data_loaders(args):
    args = args_for_data(args)
    return args, make_data_utkface(args)


def schedule_contract(u: int, l: int) -> dict:
    if (u, l) != (EXPECTED_U, EXPECTED_L):
        raise AssertionError(f"expected U/L={(EXPECTED_U, EXPECTED_L)}, got {(u, l)}")
    even = anchor_steps("evenly_interleaved", u, l, seed=0, epoch=1)
    result = {
        "U": u,
        "L": l,
        "alpha": l / u,
        "dense_supervised_mass": u * (l / u),
        "even_supervised_mass": float(l),
        "even_anchor_indices": even,
        "conditions": {
            "even_29": {
                "optimizer_steps": u,
                "unlabeled_batches": u,
                "supervised_containing_updates": l,
                "labeled_batches": l,
                "supervised_coefficient_per_active_step": 1.0,
                "labeled_iterator": "fresh each epoch; exactly one pass; no recycling",
            },
            "dense_mass_matched": {
                "optimizer_steps": u,
                "unlabeled_batches": u,
                "supervised_containing_updates": u,
                "labeled_batches": u,
                "supervised_coefficient_per_active_step": l / u,
                "labeled_iterator": "persistent canonical NextRecycle iterator",
            },
        },
    }
    assert len(even) == l and len(set(even)) == l
    assert abs(result["dense_supervised_mass"] - result["even_supervised_mass"]) < 1e-12
    return result


def validate(args):
    args, (labeled, unlabeled, _val, _test, _mean, _std) = data_loaders(args)
    report = schedule_contract(len(unlabeled), len(labeled))
    report.update({
        "pass": True,
        "manifest": args.benchmark_manifest,
        "condition_only_differences": [
            "supervised temporal placement/frequency",
            "supervised coefficient 1 versus L/U",
            "labeled iterator exposure implied by those placements",
        ],
        "shared": {
            "optimizer_steps": EXPECTED_U,
            "unlabeled_batches": EXPECTED_U,
            "lambda_u": 1.0,
            "unlabeled_objective": "MSE(strong_prediction, stopgrad(weak_prediction))",
            "single_combined_optimizer_step": True,
        },
    })
    atomic_json(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = {"training_status": "running", "final_evaluation_status": "pending"}
    atomic_json(out / "status.json", status)
    seed_all(args.seed)
    device = torch.device("cuda")
    args, (labeled, unlabeled, val, test, mean, std) = data_loaders(args)
    u, l = len(unlabeled), len(labeled)
    contract = schedule_contract(u, l)
    even_indices = set(contract["even_anchor_indices"])
    alpha = l / u if args.condition == "dense_mass_matched" else 1.0

    model, optimizer, scheduler = build_runtime(args.backbone, device)
    runtime = runtime_description(model, optimizer, scheduler)
    dense_state = {"iterator": iter(labeled), "consumed": 0, "traversals": 1}
    best = float("inf")
    best_epoch = None
    history = []

    config = {
        "experiment": "supervised_mass_matched_utkface_5pct",
        "dataset": "UTKFace",
        "labeled_ratio": 0.05,
        "method": "uniform_weak_to_strong",
        "condition": args.condition,
        "backbone": args.backbone,
        "seed": args.seed,
        "epochs": EPOCHS,
        "lambda_u": 1.0,
        "supervised_coefficient": alpha,
        "supervised_coefficient_mass_per_epoch": float(l),
        "supervised_containing_updates_per_epoch": l if args.condition == "even_29" else u,
        "total_optimizer_updates_per_epoch": u,
        "unlabeled_batches_per_epoch": u,
        "labeled_batches_consumed_per_epoch": l if args.condition == "even_29" else u,
        "even_anchor_indices": contract["even_anchor_indices"] if args.condition == "even_29" else None,
        "labeled_iterator_policy": (
            "fresh iterator each epoch; one complete pass; no recycling"
            if args.condition == "even_29"
            else "persistent canonical NextRecycle iterator; recycled at exhaustion"
        ),
        "manifest_path": args.benchmark_manifest,
        "cohort_digest": args.benchmark_metadata["cohort_sha256"],
        "scaler_mean": mean,
        "scaler_std": std,
        "attempt_number": args.attempt,
        "config_hash": args.config_hash,
        "source_trainer_sha256": args.source_sha256,
        "validation_selection": "minimum validation MAE only",
        "final_test_evaluations": 1,
        "resume": False,
        **runtime,
    }
    atomic_json(out / "config.json", config)
    metadata = {**config, "training_status": "running", "final_evaluation_status": "pending"}
    atomic_json(out / "metadata.json", metadata)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"total": 0.0, "supervised": 0.0, "unlabeled": 0.0}
        supervised_updates = labeled_consumed = optimizer_steps = 0
        epoch_label_iter = iter(labeled) if args.condition == "even_29" else None
        dense_start = dense_state["consumed"]

        for step, (x_weak, x_strong) in enumerate(unlabeled):
            x_weak = x_weak.to(device, non_blocking=True)
            x_strong = x_strong.to(device, non_blocking=True)
            active = args.condition == "dense_mass_matched" or step in even_indices
            if active:
                if args.condition == "dense_mass_matched":
                    x_labeled, y_labeled = next_labeled(dense_state, labeled)
                else:
                    try:
                        x_labeled, y_labeled = next(epoch_label_iter)
                    except StopIteration as exc:
                        raise AssertionError("EVEN-29 labeled iterator exhausted early") from exc
                x_labeled = x_labeled.to(device, non_blocking=True)
                y_labeled = y_labeled.to(device, non_blocking=True)
                losses = uniform_losses(model, x_weak, x_strong, x_labeled, y_labeled)
                total = losses["unlabeled"] + alpha * losses["supervised"]
                supervised_updates += 1
                labeled_consumed += 1
                sums["supervised"] += float(losses["supervised"].detach())
            else:
                losses = uniform_losses(model, x_weak, x_strong)
                total = losses["unlabeled"]
            assert torch.all(losses["r"] == 1)
            clip_and_step(model, optimizer, total)
            optimizer_steps += 1
            sums["total"] += float(total.detach())
            sums["unlabeled"] += float(losses["unlabeled"].detach())

        expected_supervised = l if args.condition == "even_29" else u
        assert optimizer_steps == u and supervised_updates == expected_supervised
        assert labeled_consumed == expected_supervised
        if args.condition == "even_29":
            try:
                next(epoch_label_iter)
            except StopIteration:
                pass
            else:
                raise AssertionError("EVEN-29 did not consume exactly one labeled pass")
        else:
            assert dense_state["consumed"] - dense_start == u

        val_mae, val_r2 = evaluate(model, val, mean, std, device)
        if val_mae < best:
            best = val_mae
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mae": best}, out / "best.pt")
        if epoch in DIAGNOSTIC_EPOCHS:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / f"epoch_{epoch}.pt")
        scheduler.step()
        history.append({
            "epoch": epoch,
            "loss_total": sums["total"] / u,
            "loss_supervised_active_mean": sums["supervised"] / supervised_updates,
            "loss_unlabeled": sums["unlabeled"] / u,
            "validation_mae": val_mae,
            "validation_r2": val_r2,
            "supervised_coefficient": alpha,
            "supervised_containing_updates": supervised_updates,
            "labeled_batches_consumed": labeled_consumed,
            "unlabeled_batches_consumed": u,
            "optimizer_steps": optimizer_steps,
            "lr": [group["lr"] for group in optimizer.param_groups],
        })
        with (out / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0])
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({"epoch": epoch, "condition": args.condition, "backbone": args.backbone,
                          "seed": args.seed, "validation_mae": val_mae,
                          "supervised_updates": supervised_updates, "optimizer_steps": optimizer_steps}), flush=True)

    metadata.update({
        "training_status": "complete",
        "best_validation_epoch": best_epoch,
        "best_validation_mae": best,
        "runtime_seconds_before_test": time.time() - started,
    })
    status["training_status"] = "complete"
    atomic_json(out / "metadata.json", metadata)
    atomic_json(out / "status.json", status)

    final_started = out / "final_test_started"
    final_complete = out / "final_test_complete"
    assert not final_started.exists() and not final_complete.exists()
    tmp = out / "final_test_started.tmp"
    tmp.write_text(json.dumps({"started_unix": time.time(), "test_evaluations": 0}) + "\n")
    os.replace(tmp, final_started)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_mae, test_r2, predictions, targets = evaluate(model, test, mean, std, device, predictions=True)
    np.savez_compressed(out / "test_predictions.npz", predictions_years=predictions, targets_years=targets)
    metrics = {
        "best_validation_epoch": best_epoch,
        "best_validation_mae": best,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_evaluations": 1,
    }
    atomic_json(out / "metrics.json", metrics)
    tmp = out / "final_test_complete.tmp"
    tmp.write_text(json.dumps({"completed_unix": time.time(), "test_evaluations": 1}) + "\n")
    os.replace(tmp, final_complete)
    status["final_evaluation_status"] = "complete"
    metadata.update({
        "final_evaluation_status": "complete",
        "checkpoint_reloaded": True,
        "test_used_for_selection": False,
        "test_evaluations": 1,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "runtime_seconds": time.time() - started,
    })
    atomic_json(out / "metadata.json", metadata)
    atomic_json(out / "status.json", status)
    print(json.dumps({"status": "complete", **metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--backbone", choices=("resnet50", "dinov2"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="even_29")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--config-hash", default="validation")
    parser.add_argument("--source-sha256", default="validation")
    parser.add_argument("--dino", default="s")
    parsed = parser.parse_args()
    validate(parsed) if parsed.validate else train(parsed)
