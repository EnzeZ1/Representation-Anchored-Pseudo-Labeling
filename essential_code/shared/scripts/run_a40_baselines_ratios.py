#!/usr/bin/env python3
"""Sequential resumable A40 queue for UCVME/SimRegMatch at 10% and 20%."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from baselines.benchmark_io import write_json


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/nobackup/enzez/reasoning-chains/.venv/bin/python")
DATA = ROOT / "data/utkface_all"
SPLITS = ROOT / "data_processing/splits"
OUTPUT = ROOT / "artifacts/a40_baselines/utkface"
QUEUE = ROOT / "artifacts/benchmark_queues/a40_baselines_10_20"
STATE = QUEUE / "queue_state.json"
STATUS = QUEUE / "run_status.json"
CONFIG = QUEUE / "queue_config.json"
LOG = QUEUE / "launcher.log"
RUNNER_PID = QUEUE / "runner.pid"
COHORT_DIGEST = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
GPU_UUID = "GPU-77fceb5f-454d-b01b-da74-34263153b40e"
METHODS = ("ucvme", "simregmatch")
RATIOS = (0.10, 0.20)
SEEDS = tuple(range(6))
REVISIONS = {
    "ucvme": "9f27a579cc8cb8806a56b52c3f888b647d22a074",
    "simregmatch": "0a57b897caeb5e7beebbe8379f999cb2f409bece",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(ratio: float, seed: int) -> Path:
    return SPLITS / f"utkface_ratio_{ratio:.2f}_seed_{seed}.json"


def canonical(method: str, ratio: float, seed: int) -> Path:
    return OUTPUT / method / "resnet50" / f"ratio_{ratio:.2f}" / f"seed_{seed}"


def attempt_path(method: str, ratio: float, seed: int, attempt: int) -> Path:
    return (
        OUTPUT / method / "resnet50" / f"ratio_{ratio:.2f}"
        / "_attempts" / f"seed_{seed}" / f"attempt_{attempt}"
    )


def command(method: str, ratio: float, seed: int, destination: Path) -> tuple[Path, list[str]]:
    common = [
        "--benchmark_manifest", str(manifest(ratio, seed)),
        "--benchmark_output_dir", str(destination),
        "--data_dir", str(DATA),
        "--seed", str(seed),
    ]
    if method == "ucvme":
        return ROOT / "third_party/UCVME", [str(PYTHON), "ucvme_age.py", *common]
    return ROOT / "third_party/SimRegMatch", [
        str(PYTHON), "main.py", *common, "--labeled-ratio", str(ratio),
    ]


def integrity(path: Path, method: str, ratio: float, seed: int) -> tuple[bool, str]:
    required = (
        "best.pt", "config.json", "metadata.json", "metrics.json",
        "history.csv", "run.log", "test_predictions.npz",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return False, f"missing {missing}"
    try:
        metrics = json.loads((path / "metrics.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        expected_manifest = manifest(ratio, seed).resolve()
        if metadata.get("cohort_sha256") != COHORT_DIGEST:
            return False, "cohort digest mismatch"
        if Path(metadata.get("manifest_path", "")).resolve() != expected_manifest:
            return False, "manifest path mismatch"
        if metadata.get("manifest_sha256") != sha256(expected_manifest):
            return False, "manifest checksum mismatch"
        if not math.isclose(float(metadata.get("labeled_ratio")), ratio, abs_tol=1e-12):
            return False, "labeled ratio mismatch"
        if metadata.get("checkpoint_reloaded") is not True:
            return False, "checkpoint was not reloaded"
        if metadata.get("test_used_for_selection") is not False:
            return False, "test was used for selection"
        if metadata.get("checkpoint_selection") != "lowest_validation_mae_years":
            return False, "checkpoint selection mismatch"
        if int(metadata.get("test_evaluations", -1)) != 1:
            return False, "test evaluation count mismatch"
        if int(metadata.get("test_model_inference_count", -1)) != 1:
            return False, "test model inference count mismatch"
        for key in ("validation_mae", "test_mae", "test_r2"):
            if not math.isfinite(float(metrics[key])):
                return False, f"non-finite {key}"
        payload = json.loads(expected_manifest.read_text())
        cohort_indices = np.load(path / "test_predictions.npz", allow_pickle=False)["cohort_indices"]
        if not np.array_equal(cohort_indices, np.asarray(payload["splits"]["test"], dtype=np.int64)):
            return False, "prediction ordering mismatch"
        if len(cohort_indices) != 2370:
            return False, "test prediction count mismatch"
        if (path / "best.pt").stat().st_size <= 0:
            return False, "empty checkpoint"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, "formal artifacts validated"


def enrich_metadata(path: Path, method: str, ratio: float, seed: int, attempt: int) -> None:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "dataset": "UTKFace",
        "labeled_ratio": ratio,
        "rng_seed": seed,
        "target_backbone": "ImageNet-pretrained ResNet-50",
        "pretrained_backbone_identifier": "torchvision.ResNet50_Weights.IMAGENET1K_V1",
        "method_revision": REVISIONS[method],
        "manifest_sha256": sha256(manifest(ratio, seed)),
        "checkpoint_selection": "lowest_validation_mae_years",
        "test_model_inference_count": 1,
        "test_evaluated_after_checkpoint_restoration": True,
        "physical_gpu_index": 0,
        "physical_gpu_uuid": GPU_UUID,
        "process_local_device": "cuda:0",
        "formal_attempt": attempt,
    })
    write_json(metadata_path, metadata)


def gpu_idle() -> tuple[bool, dict]:
    rows = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpu = None
    for row in rows.splitlines():
        index, uuid, name, total, free, util = [part.strip() for part in row.split(",")]
        if int(index) == 0:
            gpu = {
                "index": 0, "uuid": uuid, "name": name,
                "memory_total_mib": int(total), "memory_free_mib": int(free),
                "utilization_percent": int(util),
            }
            break
    if gpu is None or gpu["uuid"] != GPU_UUID:
        return False, gpu or {"error": "GPU 0 unavailable"}
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpu["compute_processes"] = [
        line for line in processes.splitlines() if line.startswith(GPU_UUID)
    ]
    return (
        not gpu["compute_processes"]
        and gpu["memory_free_mib"] >= 40_000
        and gpu["utilization_percent"] <= 10,
        gpu,
    )


def message(text: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def persist(jobs: list[dict], running: dict | None = None) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runner_pid": os.getpid(), "total_jobs": 24, "max_parallel": 1,
        "physical_gpu": 0, "cohort_sha256": COHORT_DIGEST,
        "jobs": jobs, "running": running,
    }
    write_json(STATE, payload)
    write_json(STATUS, payload)


def jobs_for_resume(resume: bool) -> list[dict]:
    previous = {}
    if resume and STATE.is_file():
        previous = {
            job["id"]: job for job in json.loads(STATE.read_text()).get("jobs", [])
        }
    jobs = []
    for ratio in RATIOS:
        for method in METHODS:
            for seed in SEEDS:
                path = canonical(method, ratio, seed)
                valid, detail = integrity(path, method, ratio, seed) if path.exists() else (False, "absent")
                job_id = f"utkface-resnet50-ratio_{ratio:.2f}-{method}-seed_{seed}"
                old = previous.get(job_id, {})
                jobs.append({
                    "id": job_id, "dataset": "UTKFace", "method": method,
                    "labeled_ratio": ratio, "rng_seed": seed,
                    "target_backbone": "resnet50", "physical_gpu": 0,
                    "manifest_path": str(manifest(ratio, seed)),
                    "manifest_sha256": sha256(manifest(ratio, seed)),
                    "artifact_directory": str(path),
                    "status": "complete" if valid else "pending",
                    "attempts": int(old.get("attempts", 0)) if not valid else int(old.get("attempts", 0)),
                    "attempt_history": list(old.get("attempt_history", [])),
                    "integrity_detail": detail,
                })
    assert len(jobs) == len({job["id"] for job in jobs}) == 24
    for method in METHODS:
        for ratio in RATIOS:
            assert {
                job["rng_seed"] for job in jobs
                if job["method"] == method and job["labeled_ratio"] == ratio
            } == set(SEEDS)
    return jobs


def update_summary(jobs: list[dict]) -> None:
    rows = []
    for ratio in (0.05, 0.10, 0.20):
        for method in METHODS:
            for seed in SEEDS:
                path = canonical(method, ratio, seed)
                metrics_path = path / "metrics.json"
                if not metrics_path.is_file():
                    continue
                metrics = json.loads(metrics_path.read_text())
                rows.append({
                    "method": method, "ratio": ratio, "seed": seed,
                    "test_mae": metrics["test_mae"], "test_r2": metrics["test_r2"],
                    "validation_mae": metrics["validation_mae"],
                    "best_epoch": metrics["best_epoch"], "artifact_directory": str(path),
                })
    destination = ROOT / "artifacts/a40_baselines/utkface_results_long.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# A40 UTKFace baseline results", ""]
    for ratio in (0.05, 0.10, 0.20):
        lines += [
            f"## {ratio:.0%} labeled", "",
            "| Method | Seeds | Test MAE $\\downarrow$ | Test $R^2$ $\\uparrow$ |",
            "|---|---|---:|---:|",
        ]
        for method in METHODS:
            subset = [row for row in rows if row["ratio"] == ratio and row["method"] == method]
            if len(subset) == 6:
                maes = [float(row["test_mae"]) for row in subset]
                r2s = [float(row["test_r2"]) for row in subset]
                lines.append(
                    f"| {method.upper()} | 0–5 | {statistics.mean(maes):.4f} ± "
                    f"{statistics.stdev(maes):.4f} | {statistics.mean(r2s):.4f} ± "
                    f"{statistics.stdev(r2s):.4f} |"
                )
            else:
                seeds = ",".join(str(row["seed"]) for row in subset) or "—"
                lines.append(f"| {method.upper()} | {seeds} | incomplete | incomplete |")
        lines.append("")
    (ROOT / "artifacts/a40_baselines/utkface_results_summary.md").write_text(
        "\n".join(lines) + "\n"
    )


def finalize_simregmatch(job: dict, attempt: int) -> tuple[bool, str]:
    result = subprocess.run([
        str(PYTHON), str(ROOT / "scripts/repair_a40_adapter_failures.py"),
        "--simregmatch-seed", str(job["rng_seed"]),
        "--source-attempt", str(attempt),
        "--labeled-ratio", str(job["labeled_ratio"]),
    ], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        return False, (result.stderr or result.stdout).strip()
    return integrity(
        Path(job["artifact_directory"]), job["method"],
        job["labeled_ratio"], job["rng_seed"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    RUNNER_PID.write_text(f"{os.getpid()}\n")
    jobs = jobs_for_resume(args.resume)
    write_json(CONFIG, {
        "total_jobs": 24, "methods": list(METHODS), "ratios": list(RATIOS),
        "seeds": list(SEEDS), "physical_gpu": 0, "max_parallel": 1,
        "jobs": [{key: job[key] for key in (
            "id", "method", "labeled_ratio", "rng_seed", "manifest_path",
            "manifest_sha256", "artifact_directory",
        )} for job in jobs],
    })
    persist(jobs)
    update_summary(jobs)
    if args.dry_run:
        return 0
    for job in jobs:
        if job["status"] == "complete":
            message(f"Skipping validated {job['id']}")
            continue
        succeeded = False
        for attempt in range(job["attempts"] + 1, 3):
            while True:
                idle, gpu = gpu_idle()
                if idle:
                    break
                job["status"] = "waiting_for_gpu"
                job["gpu_state"] = gpu
                persist(jobs)
                time.sleep(args.poll_seconds)
            destination = attempt_path(
                job["method"], job["labeled_ratio"], job["rng_seed"], attempt
            )
            if destination.exists() and any(destination.iterdir()):
                job["status"] = "failed"
                job["last_failure"] = f"attempt collision: {destination}"
                persist(jobs)
                break
            destination.mkdir(parents=True, exist_ok=True)
            cwd, cmd = command(
                job["method"], job["labeled_ratio"], job["rng_seed"], destination
            )
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
                "UTKFACE_BENCHMARK_ROOT": str(ROOT),
                "PYTHONPATH": os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]),
                "MPLCONFIGDIR": str(QUEUE / "matplotlib"),
            })
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            job.update({
                "status": "running", "attempts": attempt,
                "attempt_directory": str(destination), "command": cmd,
                "started_unix": time.time(),
            })
            with (destination / "run.log").open("a") as handle:
                process = subprocess.Popen(
                    cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job["pid"] = process.pid
                persist(jobs, {
                    "job_id": job["id"], "pid": process.pid, "attempt": attempt,
                    "physical_gpu": 0, "gpu_uuid": GPU_UUID,
                    "process_local_device": "cuda:0",
                })
                message(f"Started {job['id']} attempt={attempt} pid={process.pid}")
                returncode = process.wait()
            elapsed = time.time() - job["started_unix"]
            if returncode == 0 and (destination / "metadata.json").is_file():
                enrich_metadata(
                    destination, job["method"], job["labeled_ratio"],
                    job["rng_seed"], attempt,
                )
            valid, detail = integrity(
                destination, job["method"], job["labeled_ratio"], job["rng_seed"]
            )
            salvaged = False
            if (
                not valid and job["method"] == "simregmatch"
                and (destination / "test_predictions.npz").is_file()
                and (destination / "best.pt").is_file()
            ):
                salvaged, detail = finalize_simregmatch(job, attempt)
                valid = salvaged
            job["attempt_history"].append({
                "attempt": attempt, "directory": str(destination),
                "pid": process.pid, "exit_code": returncode,
                "runtime_seconds": elapsed, "integrity_valid": valid,
                "integrity_detail": detail,
                "failure_class": "postprocessing_adapter_failure" if salvaged else (
                    None if valid else "runtime_or_adapter_failure"
                ),
                "artifact_salvaged": salvaged,
            })
            if valid:
                if not salvaged:
                    os.replace(destination, Path(job["artifact_directory"]))
                job["status"] = "complete"
                succeeded = True
                persist(jobs)
                update_summary(jobs)
                message(f"Completed {job['id']} attempt={attempt}")
                break
            job["status"] = "retrying" if attempt == 1 else "failed"
            job["last_failure"] = f"exit={returncode}; {detail}"
            persist(jobs)
            message(f"{job['id']} attempt={attempt} failed: {job['last_failure']}")
        if not succeeded and job["status"] != "failed":
            job["status"] = "failed"
            persist(jobs)
    persist(jobs)
    update_summary(jobs)
    message("Queue reached terminal status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
