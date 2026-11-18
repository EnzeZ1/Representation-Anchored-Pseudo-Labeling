#!/usr/bin/env python3
"""Formal queue for the fresh FRONT / EVEN / pure BACK placement experiment."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts/supervised_spacing_front_even_back_utkface_5pct_20260929"
Q, STATE, LOCK = ART / "queue", ART / "queue/queue_state.json", ART / "queue/queue.lock"
PY = Path("/nobackup/enzez/.venv/bin/python")
TRAIN = ROOT / "training/supervised_spacing_decoupling.py"
REFERENCE = ROOT / "artifacts/supervised_spacing_decoupling_utkface_5pct/frozen_manifest.json"
BACKBONES, SEEDS = ("resnet50", "dinov2"), (0, 1, 2)
SCHEDULES = ("front_clustered", "evenly_interleaved", "back_clustered")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(run: Path) -> bool:
    try:
        status = json.loads((run / "status.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        marker = json.loads((run / "final_test_complete").read_text())
        rows = (run / "history.csv").read_text().splitlines()
        anchors = json.loads((run / "anchor_indices.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (status.get("training_status") == "complete"
            and status.get("final_evaluation_status") == "complete"
            and len(rows) == 31 and marker.get("test_evaluations") == 1
            and all(k in metrics for k in ("test_mae", "test_r2", "best_validation_epoch"))
            and (run / "best.pt").stat().st_size > 0
            and (run / "test_predictions.npz").stat().st_size > 0
            and anchors.get("U") == 563 and anchors.get("L") == 29)


def locked(fn):
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text())
        result = fn(state)
        state["updated_unix"] = time.time()
        atomic_json(STATE, state)
        return result


def claim(gpu: int):
    def mutate(state):
        for job in state["jobs"]:
            if job["status"] != "pending":
                continue
            job.update(status="running", attempts=job["attempts"] + 1,
                       worker_pid=os.getpid(), physical_gpu=gpu, start_time=time.time())
            attempt = Q / "attempts" / job["id"] / f"attempt_{job['attempts']}"
            attempt.mkdir(parents=True, exist_ok=False)
            job["attempt_directory"] = str(attempt)
            state["workers"][str(gpu)] = {"pid": os.getpid(), "job_id": job["id"], "status": "running"}
            return dict(job)
        state["workers"][str(gpu)] = {"pid": os.getpid(), "job_id": None, "status": "complete"}
        return None
    return locked(mutate)


def finish(job, code: int):
    run = Path(job["attempt_directory"]) / "run"
    def mutate(state):
        entry = next(x for x in state["jobs"] if x["id"] == job["id"])
        entry.update(exit_code=code, end_time=time.time())
        if complete(run):
            entry.update(status="complete_valid", artifact_directory=str(run))
        else:
            entry.update(status="failed")
        state["workers"][str(job["physical_gpu"])] = {"pid": os.getpid(), "job_id": None, "status": "idle"}
    locked(mutate)


def worker(gpu: int):
    manifest = json.loads((ART / "frozen_manifest.json").read_text())
    frozen_hash = manifest["source_sha256"]["training/supervised_spacing_decoupling.py"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    while True:
        if sha256(TRAIN) != frozen_hash:
            raise RuntimeError("trainer source changed after queue freeze")
        job = claim(gpu)
        if job is None:
            return
        attempt = Path(job["attempt_directory"])
        command = [str(PY), str(TRAIN), "--backbone", job["backbone"], "--seed", str(job["seed"]),
                   "--schedule", job["schedule"], "--attempt", str(job["attempts"]),
                   "--config-hash", job["config_hash"], "--output", str(attempt / "run")]
        atomic_json(attempt / "command.json", {"argv": command, "cwd": str(ROOT), "physical_gpu": gpu,
                                                "no_resume": True, "source_sha256": frozen_hash})
        code = 255
        try:
            with (attempt / "run.log").open("a") as log:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT).returncode
        finally:
            finish(job, code)


def prepare():
    if ART.exists():
        raise RuntimeError(f"refusing to overwrite {ART}")
    import torch
    import torchvision
    from training.supervised_spacing_decoupling import anchor_steps
    reference = json.loads(REFERENCE.read_text())
    base = reference["scientific_configuration"]
    assert (base["U"], base["L"], base["epochs"]) == (563, 29, 30)
    assert base["seeds"] == list(SEEDS) and base["backbones"] == list(BACKBONES)
    anchors = {
        "front_clustered": list(range(29)),
        "evenly_interleaved": [int(x) for x in anchor_steps("evenly_interleaved", 563, 29, 0, 1)],
        "back_clustered": list(range(534, 563)),
    }
    for schedule, expected in anchors.items():
        for seed in SEEDS:
            for epoch in range(1, 31):
                assert anchor_steps(schedule, 563, 29, seed, epoch) == expected
        assert len(expected) == len(set(expected)) == 29
    scientific = {
        **{k: base[k] for k in ("method", "dataset", "ratio", "cohort_digest", "seed_manifests",
                                "backbones", "seeds", "U", "L", "epochs", "lambda_u", "r",
                                "unlabeled_updates_per_epoch", "supervised_updates_per_epoch",
                                "optimizer_steps_per_epoch", "labeled_loader_passes_per_epoch",
                                "canonical_configuration_source", "diagnostic_checkpoint_epochs")},
        "schedules": list(SCHEDULES), "anchor_indices_every_epoch": anchors,
        "schedule_definitions": {
            "front_clustered": "0..28",
            "evenly_interleaved": "floor((j+0.5)*563/29), j=0..28",
            "back_clustered": "534..562 (pure back-loading; no initial/warm-up anchor)",
        },
        "loss_at_anchor": "L_u + L_s", "loss_at_non_anchor": "L_u",
        "supervised_coefficient": 1.0, "unlabeled_loss_present_at_all_steps": True,
        "no_extra_optimizer_steps": True, "no_warmup_pass": True,
        "reference_experiment": str(REFERENCE.parent), "reference_config_hash": reference["config_hash"],
        "only_intended_difference_among_conditions": "temporal indices of 29 supervised-containing positions",
    }
    blob = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    config_hash = hashlib.sha256(blob).hexdigest()
    jobs = [{"id": f"uniform__{b}__seed_{s}__{schedule}", "backbone": b, "seed": s,
             "schedule": schedule, "status": "pending", "attempts": 0, "config_hash": config_hash}
            for b in BACKBONES for s in SEEDS for schedule in SCHEDULES]
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT,
                          stdout=subprocess.PIPE, check=True).stdout
    manifest = {
        "experiment": ART.name, "created_unix": time.time(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "uncommitted_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {"training/supervised_spacing_decoupling.py": sha256(TRAIN),
                          "scripts/run_supervised_spacing_front_even_back.py": sha256(Path(__file__))},
        "runtime": {"python": sys.version, "pytorch": torch.__version__, "torchvision": torchvision.__version__,
                    "cuda_build": torch.version.cuda, "platform": platform.platform()},
        "scientific_configuration": scientific, "config_hash": config_hash,
    }
    atomic_json(ART / "frozen_manifest.json", manifest)
    atomic_json(ART / "schedule_validation.json", {"pass": True, "anchors": anchors,
                                                    "optimizer_steps": 563, "unlabeled_steps": 563,
                                                    "supervised_steps": 29})
    atomic_json(STATE, {"experiment": ART.name, "jobs": jobs, "workers": {}, "created_unix": time.time()})
    print(json.dumps({"root": str(ART), "jobs": len(jobs), "config_hash": config_hash,
                      "trainer_sha256": sha256(TRAIN), "anchors": anchors}, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--prepare", action="store_true"); p.add_argument("--worker", type=int)
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.worker is not None: worker(a.worker)
    else: p.error("mode required")


if __name__ == "__main__": main()
