"""Matched-budget supervised-anchor spacing experiment for canonical Uniform SSR."""
from __future__ import annotations

import argparse, csv, json, os, time
from pathlib import Path

import numpy as np
import torch

from training.canonical_uniform_frequency import (
    DIAGNOSTIC_EPOCHS, EPOCHS, args_for_data, atomic_json, build_runtime,
    clip_and_step, evaluate, runtime_description, seed_all, uniform_losses,
)
from training.train import make_data_utkface

SCHEDULES = ("evenly_interleaved", "front_clustered", "random_interleaved", "back_loaded", "back_clustered")
RANDOM_STREAM_TAG = 0x53504143


def anchor_steps(schedule: str, u: int, l: int, seed: int, epoch: int) -> list[int]:
    if not 0 < l <= u:
        raise ValueError(f"requires 0 < L <= U, got L={l}, U={u}")
    if schedule == "front_clustered":
        result = list(range(l))
    elif schedule == "back_loaded":
        # Preserve the initial supervised contact used by the canonical
        # trajectory, then defer the remaining L-1 anchors to the end.
        result = [0] + list(range(u - (l - 1), u))
    elif schedule == "back_clustered":
        # Pure back-loading: all L supervised anchors occupy the final L
        # optimizer positions. There is no initial or warm-up anchor.
        result = list(range(u - l, u))
    elif schedule == "evenly_interleaved":
        result = [int(np.floor((j + .5) * u / l)) for j in range(l)]
    elif schedule == "random_interleaved":
        # An isolated stream: schedule sampling cannot advance Python, NumPy global,
        # PyTorch, model, augmentation, or DataLoader RNG state.
        rng = np.random.default_rng(np.random.SeedSequence([RANDOM_STREAM_TAG, seed, epoch]))
        result = sorted(int(x) for x in rng.choice(u, size=l, replace=False))
    else:
        raise ValueError(schedule)
    if len(result) != l or len(set(result)) != l or min(result) < 0 or max(result) >= u:
        raise AssertionError((schedule, u, l, result))
    return result


def validate_schedules(u: int, l: int, seed: int, epochs: int = EPOCHS) -> dict:
    report = {"U": u, "L": l, "seed": seed, "epochs": epochs, "schedules": {}}
    for schedule in SCHEDULES:
        per_epoch = {}
        for epoch in range(1, epochs + 1):
            indices = anchor_steps(schedule, u, l, seed, epoch)
            # Instrument the loop contract without loading images or training.
            labeled_advances = sum(t in set(indices) for t in range(u))
            per_epoch[str(epoch)] = indices
            assert labeled_advances == l
        report["schedules"][schedule] = {
            "anchor_indices_by_epoch": per_epoch,
            "unlabeled_batches_per_epoch": u,
            "labeled_batches_per_epoch": l,
            "optimizer_steps_per_epoch": u,
            "no_labeled_cycling": True,
        }
    assert report["schedules"]["front_clustered"]["anchor_indices_by_epoch"]["1"] == list(range(l))
    assert report["schedules"]["back_loaded"]["anchor_indices_by_epoch"]["1"] == [0] + list(range(u - (l - 1), u))
    assert report["schedules"]["back_clustered"]["anchor_indices_by_epoch"]["1"] == list(range(u - l, u))
    even = report["schedules"]["evenly_interleaved"]["anchor_indices_by_epoch"]["1"]
    assert even[0] < u // l and even[-1] >= u - (u // l) - 1
    report["pass"] = True
    return report


def loaders(a):
    a = args_for_data(a)
    return a, make_data_utkface(a)


def validate_from_loaders(a):
    a, (lab, unlab, _val, _test, _mean, _std) = loaders(a)
    report = validate_schedules(len(unlab), len(lab), a.seed)
    atomic_json(Path(a.output), report)
    print(json.dumps({"pass": True, "U": len(unlab), "L": len(lab)}))


def train(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    status = {"training_status": "running", "final_evaluation_status": "pending", "analysis_status": "pending"}
    atomic_json(out / "status.json", status)
    seed_all(a.seed); device = torch.device("cuda")
    a, (lab, unlab, val, test, mean, std) = loaders(a)
    u, l = len(unlab), len(lab)
    validation = validate_schedules(u, l, a.seed)
    atomic_json(out / "schedule_validation.json", validation)
    schedules_by_epoch = {str(e): anchor_steps(a.schedule, u, l, a.seed, e) for e in range(1, EPOCHS + 1)}
    atomic_json(out / "anchor_indices.json", {
        "schedule": a.schedule, "seed": a.seed, "U": u, "L": l,
        "random_rng": "numpy.default_rng(SeedSequence([0x53504143, seed, epoch]))" if a.schedule == "random_interleaved" else None,
        "anchor_indices_by_epoch": schedules_by_epoch,
    })
    model, opt, sched = build_runtime(a.backbone, device)
    runtime = runtime_description(model, opt, sched); best = float("inf"); best_epoch = None; rows = []
    metadata = {
        "method": "uniform", "experiment": "supervised_spacing_decoupling_utkface_5pct",
        "backbone": a.backbone, "seed": a.seed, "schedule": a.schedule,
        "attempt_number": a.attempt, "config_hash": a.config_hash,
        "manifest_path": a.benchmark_manifest, "cohort_digest": a.benchmark_metadata["cohort_sha256"],
        "scaler_mean": mean, "scaler_std": std, **runtime, "lambda_u": 1.0, "r_mode": "constant_one",
        "epochs": EPOCHS, "U": u, "L": l, "steps_per_epoch": u,
        "supervised_updates_per_epoch": l, "unlabeled_updates_per_epoch": u,
        "total_optimizer_steps": EPOCHS * u, "total_supervised_updates": EPOCHS * l,
        "labeled_iterator_policy": "fresh iterator each epoch; exactly one pass; never cycled",
        "anchor_indices_file": "anchor_indices.json", "training_status": "running",
        "final_evaluation_status": "pending", "analysis_status": "pending",
    }
    atomic_json(out / "metadata.json", metadata)
    for epoch in range(1, EPOCHS + 1):
        model.train(); anchors = unlabeled = 0; label_iter = iter(lab)
        indices = set(schedules_by_epoch[str(epoch)])
        sums = {"total": 0., "supervised": 0., "unlabeled": 0.}
        for t, (xuw, xus) in enumerate(unlab):
            xuw = xuw.to(device, non_blocking=True); xus = xus.to(device, non_blocking=True)
            if t in indices:
                try: xl, yl = next(label_iter)
                except StopIteration as exc: raise AssertionError("labeled loader ended before L anchors") from exc
                xl = xl.to(device, non_blocking=True); yl = yl.to(device, non_blocking=True)
                losses = uniform_losses(model, xuw, xus, xl, yl)
                anchors += 1; sums["supervised"] += float(losses["supervised"].detach())
            else:
                losses = uniform_losses(model, xuw, xus)
            assert torch.all(losses["r"] == 1)
            clip_and_step(model, opt, losses["total"]); unlabeled += 1
            sums["total"] += float(losses["total"].detach()); sums["unlabeled"] += float(losses["unlabeled"].detach())
        assert unlabeled == u and anchors == l
        try: next(label_iter)
        except StopIteration: pass
        else: raise AssertionError("labeled loader had unconsumed batches after L anchors")
        val_mae, val_r2 = evaluate(model, val, mean, std, device)
        if val_mae < best:
            best = val_mae; best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mae": best}, out / "best.pt")
        if epoch in DIAGNOSTIC_EPOCHS:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / f"epoch_{epoch}.pt")
        sched.step()
        rows.append({"epoch": epoch, "loss_total": sums["total"] / u,
                     "loss_supervised_anchor_mean": sums["supervised"] / l,
                     "loss_unlabeled": sums["unlabeled"] / u, "validation_mae": val_mae,
                     "validation_r2": val_r2, "direct_supervised_anchors": anchors,
                     "labeled_batches_consumed": anchors, "labeled_loader_passes": 1,
                     "unlabeled_batches_consumed": unlabeled, "optimizer_steps": unlabeled,
                     "lr": [g["lr"] for g in opt.param_groups]})
        with (out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    metadata.update(training_status="complete", best_validation_epoch=best_epoch, best_validation_mae=best,
                    labeled_batches_actually_consumed=EPOCHS*l, unlabeled_batches_actually_consumed=EPOCHS*u)
    status["training_status"] = "complete"
    atomic_json(out / "metadata.json", metadata); atomic_json(out / "status.json", status)
    marker, started = out / "final_test_complete", out / "final_test_started"
    assert not marker.exists() and not started.exists()
    tmp = out / "final_test_started.tmp"; tmp.write_text(json.dumps({"started_unix": time.time()}) + "\n"); os.replace(tmp, started)
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    mae, r2, p, y = evaluate(model, test, mean, std, device, True)
    np.savez_compressed(out / "test_predictions.npz", predictions_years=p, targets_years=y)
    atomic_json(out / "metrics.json", {"best_validation_epoch": best_epoch, "best_validation_mae": best, "test_mae": mae, "test_r2": r2})
    tmp = out / "final_test_complete.tmp"; tmp.write_text(json.dumps({"completed_unix": time.time(), "test_evaluations": 1}) + "\n"); os.replace(tmp, marker)
    status["final_evaluation_status"] = "complete"; metadata.update(final_evaluation_status="complete", final_test_mae=mae, final_test_r2=r2)
    atomic_json(out / "metadata.json", metadata); atomic_json(out / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate-schedules", action="store_true")
    p.add_argument("--backbone", choices=("resnet50", "dinov2"), required=True)
    p.add_argument("--schedule", choices=SCHEDULES, default="evenly_interleaved")
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attempt", type=int, default=1); p.add_argument("--config-hash", default="validation")
    p.add_argument("--dino", default="s")
    a = p.parse_args(); validate_from_loaders(a) if a.validate_schedules else train(a)
