#!/usr/bin/env python3
"""Detached eight-GPU queue for the UTKFace supervision-mass experiment."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/supervised_mass_matched_utkface_5pct_20260924"
QUEUE = ART / "queue"
STATE = QUEUE / "queue_state.json"
LOCK = QUEUE / "queue.lock"
PYTHON = Path("/nobackup/enzez/.venv/bin/python")
TRAINER = ROOT / "training/supervised_mass_matched.py"
BACKBONES = ("resnet50", "dinov2")
CONDITIONS = ("even_29", "dense_mass_matched")
SEEDS = tuple(range(6))
MAX_ATTEMPTS = 3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def with_lock(fn):
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text())
        result = fn(state)
        state["updated_unix"] = time.time()
        atomic_json(STATE, state)
        return result


def valid(run: Path) -> bool:
    try:
        status = json.loads((run / "status.json").read_text())
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        history = list(csv.DictReader((run / "history.csv").open()))
        marker = json.loads((run / "final_test_complete").read_text())
        predictions = np.load(run / "test_predictions.npz")
        checkpoint = run / "best.pt"
    except Exception:
        return False
    expected_supervised = 29 if config.get("condition") == "even_29" else 563
    return (
        status.get("training_status") == "complete"
        and status.get("final_evaluation_status") == "complete"
        and config.get("epochs") == 30
        and config.get("total_optimizer_updates_per_epoch") == 563
        and config.get("supervised_containing_updates_per_epoch") == expected_supervised
        and len(history) == 30
        and all(int(row["optimizer_steps"]) == 563 for row in history)
        and all(int(row["supervised_containing_updates"]) == expected_supervised for row in history)
        and checkpoint.is_file() and checkpoint.stat().st_size > 0
        and marker.get("test_evaluations") == 1
        and metrics.get("test_evaluations") == 1
        and math.isfinite(float(metrics["test_mae"]))
        and math.isfinite(float(metrics["test_r2"]))
        and len(predictions["predictions_years"]) > 0
        and np.isfinite(predictions["predictions_years"]).all()
    )


def claim(gpu: int):
    def mutate(state):
        for job in state["jobs"]:
            if job["status"] != "pending":
                continue
            job["attempts"] += 1
            attempt = QUEUE / "attempts" / job["key"] / f"attempt_{job['attempts']}"
            attempt.mkdir(parents=True, exist_ok=False)
            job.update({
                "status": "running",
                "worker_pid": os.getpid(),
                "physical_gpu": gpu,
                "started_unix": time.time(),
                "attempt_directory": str(attempt),
            })
            state["workers"][str(gpu)] = {"pid": os.getpid(), "status": "running", "key": job["key"]}
            return dict(job)
        state["workers"][str(gpu)] = {"pid": os.getpid(), "status": "exhausted", "key": None}
        return None
    return with_lock(mutate)


def finish(job, exit_code: int):
    attempt = Path(job["attempt_directory"])
    run = attempt / "run"
    is_valid = valid(run)
    def mutate(state):
        entry = next(item for item in state["jobs"] if item["key"] == job["key"])
        record = {
            "attempt": entry["attempts"], "directory": str(attempt), "exit_code": exit_code,
            "valid": is_valid, "ended_unix": time.time(),
        }
        entry["attempt_history"].append(record)
        if is_valid:
            entry.update(status="complete_valid", artifact_directory=str(run), exit_code=exit_code)
        elif entry["attempts"] < MAX_ATTEMPTS:
            entry.update(status="pending", exit_code=exit_code, retry_reason="invalid_or_failed_attempt; fresh epoch-0 retry")
        else:
            entry.update(status="failed", exit_code=exit_code)
        state["workers"][str(job["physical_gpu"])] = {"pid": os.getpid(), "status": "idle", "key": None}
    with_lock(mutate)


def summarize_if_complete():
    state = json.loads(STATE.read_text())
    if not all(job["status"] == "complete_valid" for job in state["jobs"]):
        return
    rows = []
    for job in state["jobs"]:
        run = Path(job["artifact_directory"])
        metrics = json.loads((run / "metrics.json").read_text())
        config = json.loads((run / "config.json").read_text())
        rows.append({
            "condition": job["condition"], "backbone": job["backbone"], "seed": job["seed"],
            "attempt": job["attempts"], "best_validation_epoch": metrics["best_validation_epoch"],
            "best_validation_mae": metrics["best_validation_mae"], "test_mae": metrics["test_mae"],
            "test_r2": metrics["test_r2"], "supervised_coefficient": config["supervised_coefficient"],
            "supervised_containing_updates_per_epoch": config["supervised_containing_updates_per_epoch"],
            "total_optimizer_updates_per_epoch": config["total_optimizer_updates_per_epoch"],
            "artifact_path": str(run),
        })
    seed_csv = ART / "seed_level_results.csv"
    with seed_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = []
    for condition in CONDITIONS:
        for backbone in BACKBONES:
            group = [row for row in rows if row["condition"] == condition and row["backbone"] == backbone]
            maes = np.asarray([row["test_mae"] for row in group], float)
            r2s = np.asarray([row["test_r2"] for row in group], float)
            summary.append({"condition": condition, "backbone": backbone, "n": len(group),
                            "test_mae_mean": maes.mean(), "test_mae_population_sd": maes.std(ddof=0),
                            "test_r2_mean": r2s.mean(), "test_r2_population_sd": r2s.std(ddof=0)})
    with (ART / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0]); writer.writeheader(); writer.writerows(summary)
    paired = []
    for backbone in BACKBONES:
        for seed in SEEDS:
            even = next(row for row in rows if row["backbone"] == backbone and row["seed"] == seed and row["condition"] == "even_29")
            dense = next(row for row in rows if row["backbone"] == backbone and row["seed"] == seed and row["condition"] == "dense_mass_matched")
            paired.append({"backbone": backbone, "seed": seed, "even_29_test_mae": even["test_mae"],
                           "dense_mass_matched_test_mae": dense["test_mae"],
                           "dense_minus_even_mae": dense["test_mae"] - even["test_mae"]})
    for backbone in BACKBONES:
        vals = np.asarray([row["dense_minus_even_mae"] for row in paired if row["backbone"] == backbone])
        paired.append({"backbone": backbone, "seed": "aggregate", "even_29_test_mae": "",
                       "dense_mass_matched_test_mae": "", "dense_minus_even_mae": vals.mean(),
                       "delta_population_sd": vals.std(ddof=0)})
    fields = sorted({key for row in paired for key in row})
    with (ART / "paired_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    (ART / "audit_report.md").write_text(
        "# Supervision-mass-matched audit\n\n"
        "All 24 runs passed the formal validator. Both conditions use 563 optimizer positions and "
        "563 unlabeled batches per epoch, the same canonical Uniform weak-to-strong objective, model, "
        "optimizer, scheduler, transforms, manifests, seeds, validation-only selection, and exactly-once "
        "final test. EVEN-29 places coefficient-1 supervision at 29 evenly spaced positions. "
        "DENSE-MASS-MATCHED places coefficient 29/563 supervision at every position, for equal nominal "
        "supervised coefficient mass per epoch.\n"
    )
    atomic_json(ART / "complete.json", {"status": "complete", "completed_unix": time.time(), "runs": 24})


def worker(gpu: int):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"})
    while True:
        job = claim(gpu)
        if job is None:
            summarize_if_complete()
            return
        attempt = Path(job["attempt_directory"])
        run = attempt / "run"
        command = [str(PYTHON), str(TRAINER), "--backbone", job["backbone"], "--condition", job["condition"],
                   "--seed", str(job["seed"]), "--output", str(run), "--attempt", str(job["attempts"]),
                   "--config-hash", job["config_hash"], "--source-sha256", job["trainer_sha256"]]
        atomic_json(attempt / "command.json", {"argv": command, "cwd": str(ROOT), "physical_gpu": gpu,
                                                "started_unix": time.time(), "resume": False})
        code = 255
        try:
            with (attempt / "run.log").open("a") as log:
                code = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
        finally:
            finish(job, code)


def prepare():
    if ART.exists():
        raise RuntimeError(f"refusing to overwrite {ART}")
    import torch
    import torchvision
    trainer_sha = sha(TRAINER)
    inherited = [ROOT / "training/canonical_uniform_frequency.py", ROOT / "training/supervised_spacing_decoupling.py",
                 ROOT / "training/train.py", ROOT / "models/backbone.py", ROOT / "models/dinov2_backbone.py"]
    manifests = {}
    for seed in SEEDS:
        path = ROOT / f"data_processing/splits/utkface_ratio_0.05_seed_{seed}.json"
        doc = json.loads(path.read_text())
        manifests[str(seed)] = {"path": str(path), "sha256": sha(path), "label_scaler": doc["label_scaler"]}
    jobs = []
    # Adjacent paired conditions preserve transparent queue provenance; no artifact reuse.
    for backbone in BACKBONES:
        for seed in SEEDS:
            for condition in CONDITIONS:
                key = f"{condition}__{backbone}__seed_{seed}"
                jobs.append({"key": key, "condition": condition, "backbone": backbone, "seed": seed,
                             "status": "pending", "attempts": 0, "attempt_history": [], "trainer_sha256": trainer_sha})
    scientific = {
        "dataset": "UTKFace", "ratio": 0.05, "backbones": list(BACKBONES), "seeds": list(SEEDS),
        "conditions": list(CONDITIONS), "epochs": 30, "U": 563, "L": 29, "alpha": 29/563,
        "lambda_u": 1.0, "optimizer_steps_per_epoch": 563, "unlabeled_batches_per_epoch": 563,
        "even_schedule": "floor((j+0.5)*563/29), j=0..28",
        "dense_labeled_iterator": "canonical persistent NextRecycle",
        "canonical_configuration_source": "training/canonical_uniform_frequency.py",
        "placement_source": "training/supervised_spacing_decoupling.py",
        "manifests": manifests,
    }
    config_hash = hashlib.sha256(json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for job in jobs:
        job["config_hash"] = config_hash
    ART.mkdir(parents=True)
    snapshots = ART / "source_snapshot"
    snapshots.mkdir()
    source_hashes = {}
    for path in [TRAINER, Path(__file__).resolve(), *inherited]:
        target = snapshots / path.name
        shutil.copy2(path, target)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout
    manifest = {
        "experiment": "supervised_mass_matched_utkface_5pct_20260924",
        "created_unix": time.time(), "host": platform.node(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(), "source_hashes": source_hashes,
        "runtime": {"python": sys.version, "pytorch": torch.__version__, "torchvision": torchvision.__version__,
                    "cuda": torch.version.cuda, "platform": platform.platform()},
        "scientific_configuration": scientific, "config_hash": config_hash,
        "only_intended_difference": "supervision frequency/placement versus per-step coefficient at equal nominal epoch mass",
    }
    atomic_json(ART / "frozen_manifest.json", manifest)
    atomic_json(STATE, {"experiment": manifest["experiment"], "jobs": jobs, "workers": {}, "created_unix": time.time()})
    print(json.dumps({"artifact_root": str(ART), "config_hash": config_hash, "trainer_sha256": trainer_sha}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        worker(args.worker)
    else:
        parser.error("choose --prepare or --worker GPU")
