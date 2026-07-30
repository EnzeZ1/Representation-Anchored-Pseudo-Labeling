#!/usr/bin/env python3
"""Eight fixed-GPU resumable queues for the supervised STS-B-DIR benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/nobackup/enzez/.venv/bin/python")
QUEUE_ROOT = ROOT / "artifacts/benchmark_queues/supervised_stsb_8gpu"
RESULTS = ROOT / "artifacts/supervised_baselines/stsb"
STATE = QUEUE_ROOT / "queue_state.json"
STATUS = QUEUE_ROOT / "run_status.json"
LOG = QUEUE_ROOT / "launcher.log"
RUNNER_PID = QUEUE_ROOT / "runner.pid"
IMDB_REPAIR = (
    ROOT / "artifacts/supervised_baselines/imdb_wiki/dinov2_vits14/"
    "ratio_1.00/seed_5"
)
COHORT_SHA256 = None
GROUPS = {
    0: ("bilstm_glove", "0.05"),
    1: ("bilstm_glove", "0.10"),
    2: ("bilstm_glove", "0.20"),
    3: ("bilstm_glove", "1.00"),
    4: ("roberta_base", "0.05"),
    5: ("roberta_base", "0.10"),
    6: ("roberta_base", "0.20"),
    7: ("roberta_base", "1.00"),
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def imdb_repair_valid():
    required = ("best.pt", "metadata.json", "metrics.json", "history.csv", "test_predictions.npz")
    if not all((IMDB_REPAIR / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((IMDB_REPAIR / "metadata.json").read_text())
        metrics = json.loads((IMDB_REPAIR / "metrics.json").read_text())
        return (
            metadata["checkpoint_reloaded"] is True
            and metadata["test_used_for_selection"] is False
            and metadata["test_model_inference_count"] == 1
            and metrics["checkpoint_reloaded"] is True
            and metrics["test_used_for_selection"] is False
            and metrics["test_model_inference_count"] == 1
        )
    except Exception:
        return False


def result_valid(path):
    required = ("best.pt", "config.json", "metadata.json", "metrics.json",
                "history.csv", "run.log", "test_predictions.npz")
    if not all((path / name).is_file() for name in required):
        return False
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        return (
            metadata["status"] == "complete"
            and metadata["checkpoint_reloaded"] is True
            and metadata["test_used_for_selection"] is False
            and metadata["test_model_inference_count"] == 1
            and metrics["checkpoint_reloaded"] is True
            and metrics["test_used_for_selection"] is False
            and metrics["test_model_inference_count"] == 1
            and all(math.isfinite(float(metrics[key]))
                    for key in ("best_validation_mse_score_units",
                                "test_mse_score_units", "test_r2"))
        )
    except Exception:
        return False


def jobs():
    records = []
    for gpu, (backbone, ratio) in GROUPS.items():
        for seed in range(6):
            output = RESULTS / backbone / f"ratio_{ratio}" / f"seed_{seed}"
            attempts = [
                int(path.name.removeprefix("attempt_"))
                for path in output.glob("attempt_*")
                if path.name.removeprefix("attempt_").isdigit()
            ]
            records.append({
                "id": f"{backbone}:ratio_{ratio}:seed_{seed}",
                "gpu_owner": gpu,
                "backbone": backbone,
                "labeled_ratio": float(ratio),
                "seed": seed,
                "manifest": str(
                    ROOT / f"data_processing/splits/stsb_ratio_{ratio}_seed_{seed}.json"
                ),
                "output_dir": str(output),
                "attempts": max(attempts, default=0),
                "formal_failures": 0,
                "status": "complete" if result_valid(output) else "pending",
            })
    assert len(records) == 48 and len({job["id"] for job in records}) == 48
    return records


def command(job, attempt):
    attempt_dir = Path(job["output_dir"]) / f"attempt_{attempt}"
    epochs = "200" if job["backbone"] == "bilstm_glove" else "10"
    batch = "32" if job["backbone"] == "bilstm_glove" else "16"
    return [
        str(PYTHON), "-m", "training.supervised_stsb",
        "--backbone", job["backbone"],
        "--manifest", job["manifest"],
        "--output-dir", str(attempt_dir),
        "--seed", str(job["seed"]),
        "--epochs", epochs,
        "--batch-size", batch,
        "--num-workers", "2",
    ], attempt_dir


def promote(attempt, canonical):
    canonical.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "config.json", "metadata.json", "metrics.json",
                 "history.csv", "run.log", "test_predictions.npz"):
        source = attempt / name
        if not source.is_file():
            raise FileNotFoundError(source)
        os.replace(source, canonical / name)
    write_json(canonical / "successful_attempt.json", {
        "source_attempt": attempt.name, "promoted_unix": time.time(),
    })


def gpu_free(gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    free, utilization = (int(value.strip()) for value in output.split(","))
    processes = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return not processes and free >= 9000 and utilization <= 10


def persist(records, running, stopping=False):
    payload = {
        "updated_unix": time.time(),
        "runner_pid": os.getpid(),
        "total_jobs": len(records),
        "stopping": stopping,
        "gpu7_gate": {
            "condition": "IMDB-WIKI DINOv2 100% seed 5 formally valid",
            "satisfied": imdb_repair_valid(),
        },
        "jobs": records,
        "running": {
            job_id: {
                "pid": item["process"].pid,
                "physical_gpu": item["gpu"],
                "attempt": item["attempt"],
            }
            for job_id, item in running.items()
        },
    }
    write_json(STATE, payload)
    write_json(STATUS, payload)


def update_summary(records):
    rows = []
    for job in records:
        path = Path(job["output_dir"])
        if not result_valid(path):
            continue
        metrics = json.loads((path / "metrics.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        rows.append({
            "backbone": job["backbone"],
            "labeled_ratio": job["labeled_ratio"],
            "seed": job["seed"],
            "test_mse": metrics["test_mse_score_units"],
            "test_mae": metrics["test_mae_score_units"],
            "test_r2": metrics["test_r2"],
            "best_epoch": metrics["best_epoch"],
            "best_validation_mse": metrics["best_validation_mse_score_units"],
            "runtime_seconds": metadata["runtime_seconds"],
            "checkpoint_path": str(path / "best.pt"),
        })
    long_path = ROOT / "artifacts/supervised_baselines/stsb_results_long.csv"
    summary_path = ROOT / "artifacts/supervised_baselines/stsb_results_summary.md"
    long_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with long_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# STS-B-DIR supervised reference results", "",
        "| Backbone | Labeled Ratio | Seeds | Test MSE ↓ | Test R² ↑ |",
        "|---|---:|---|---:|---:|",
    ]
    for backbone, ratio in GROUPS.values():
        group = [row for row in rows if row["backbone"] == backbone
                 and math.isclose(row["labeled_ratio"], float(ratio))]
        if len(group) != 6:
            continue
        mse = [float(row["test_mse"]) for row in group]
        r2 = [float(row["test_r2"]) for row in group]
        lines.append(
            f"| {backbone} | {float(ratio):.2f} | 0–5 | "
            f"{mean(mse):.4f} ± {sample_sd(mse):.4f} | "
            f"{mean(r2):.4f} ± {sample_sd(r2):.4f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    RUNNER_PID.write_text(str(os.getpid()) + "\n")
    records = jobs()
    write_json(QUEUE_ROOT / "queue_config.json", {
        "dataset": "STS-B-DIR",
        "method": "supervised",
        "total_jobs": 48,
        "gpu_groups": GROUPS,
        "seeds": list(range(6)),
        "selection_metric": "validation MSE in original STS-B score units",
        "one_process_per_physical_gpu": True,
        "gpu7_dependency": "IMDB-WIKI DINOv2 100% seed 5",
    })
    running = {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        log("Stop requested; active children are left to finish.")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    persist(records, running)
    while True:
        for job_id, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["handle"].close()
            job = item["job"]
            if return_code == 0 and result_valid(item["attempt_dir"]):
                promote(item["attempt_dir"], Path(job["output_dir"]))
                job["status"] = "complete"
                log(f"Completed {job_id} on GPU {item['gpu']}")
            elif job["formal_failures"] < 1:
                job["formal_failures"] += 1
                job["status"] = "pending"
                job["last_failure"] = f"attempt {item['attempt']} exit {return_code}"
                log(f"Retry scheduled for {job_id}: exit {return_code}")
            else:
                job["status"] = "failed"
                job["last_failure"] = f"attempt {item['attempt']} exit {return_code}; exhausted"
                log(f"Failed {job_id}; retry exhausted")
            del running[job_id]
            update_summary(records)
            persist(records, running, stopping)
        if not running and (
            stopping or not any(job["status"] == "pending" for job in records)
        ):
            break
        if not stopping:
            used = {item["gpu"] for item in running.values()}
            for gpu in range(8):
                if gpu in used or (gpu == 7 and not imdb_repair_valid()) or not gpu_free(gpu):
                    continue
                job = next((candidate for candidate in records
                            if candidate["gpu_owner"] == gpu and candidate["status"] == "pending"), None)
                if job is None:
                    continue
                job["attempts"] += 1
                job["status"] = "running"
                command_line, attempt_dir = command(job, job["attempts"])
                attempt_dir.mkdir(parents=True, exist_ok=False)
                handle = (attempt_dir / "run.log").open("a")
                environment = os.environ.copy()
                environment.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "SUPERVISED_PHYSICAL_GPU": str(gpu),
                    "PYTHONPATH": str(ROOT),
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                })
                process = subprocess.Popen(
                    command_line, cwd=ROOT, env=environment, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                running[job["id"]] = {
                    "process": process, "handle": handle, "gpu": gpu,
                    "job": job, "attempt": job["attempts"], "attempt_dir": attempt_dir,
                }
                log(f"Started {job['id']} pid={process.pid} physical GPU={gpu}")
                persist(records, running, stopping)
        time.sleep(args.poll_seconds)
    update_summary(records)
    persist(records, running, stopping)
    failed = [job for job in records if job["status"] == "failed"]
    log(f"Queue terminal: complete={len(records)-len(failed)} failed={len(failed)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
