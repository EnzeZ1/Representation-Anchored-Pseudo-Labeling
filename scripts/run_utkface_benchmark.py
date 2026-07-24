#!/usr/bin/env python3
"""Persistent, resumable GPU queue for the UTKFace five-method benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.benchmark_io import run_complete, write_json
from data_processing.utkface_protocol import runtime_metadata


ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "utkface_5pct"
PYTHON_DEPS = ARTIFACT_ROOT / "python_deps"
QUEUE_PATH = ARTIFACT_ROOT / "queue_state.json"
STATUS_PATH = ARTIFACT_ROOT / "run_status.json"
LAUNCHER_LOG = ARTIFACT_ROOT / "launcher.log"
RUNNER_PID = ARTIFACT_ROOT / "runner.pid"
SOURCE_STATE = ARTIFACT_ROOT / "source_state.json"
COHORT_DIGEST = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
METHODS = ("rapl", "hpl", "ucvme", "rankup", "simregmatch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--available_gpus", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--poll_seconds", type=int, default=20)
    return parser.parse_args()


def manifest(seed):
    return PROJECT_ROOT / "data_processing" / "splits" / f"utkface_ratio_0.05_seed_{seed}.json"


def output_dir(method, seed, smoke):
    if smoke:
        return ARTIFACT_ROOT / "smoke" / method / f"seed_{seed}"
    return ARTIFACT_ROOT / method / f"seed_{seed}"


def command_for(method, seed, smoke):
    output = output_dir(method, seed, smoke)
    epochs = 1 if smoke else None
    common = ["--benchmark_manifest", str(manifest(seed)), "--benchmark_output_dir", str(output),
              "--data_dir", str(PROJECT_ROOT / "data" / "utkface_all"), "--seed", str(seed)]
    if method == "rapl":
        command = [sys.executable, str(PROJECT_ROOT / "train.py"), "-dataset", "utkface", "--data_dir",
                   str(PROJECT_ROOT / "data" / "utkface_all"), "--method", "probe", "--labeled_ratio", "0.05",
                   "--backbone", "resnet50", "--save", str(output / "best.pt"), *common]
        if epochs is not None:
            command += ["--epochs", str(epochs)]
        return PROJECT_ROOT, command
    upstream = {
        "hpl": PROJECT_ROOT / "third_party" / "Heteroscedastic-Pseudo-Labels" / "utkface",
        "ucvme": PROJECT_ROOT / "third_party" / "UCVME",
        "rankup": PROJECT_ROOT / "third_party" / "Semi-Supervised-Regression",
        "simregmatch": PROJECT_ROOT / "third_party" / "SimRegMatch",
    }[method]
    script = {"hpl": "main_ours.py", "ucvme": "ucvme_age.py", "rankup": "train.py", "simregmatch": "main.py"}[method]
    command = [sys.executable, script, *common]
    if method == 'rankup':
        command += ['--c', 'config/classic_cv/rankup/rankup_utkface_lb250_s0.yaml', '--gpu', '0']
    if epochs is not None:
        command += ["--benchmark_epochs", str(epochs)]
    return upstream, command


def gpu_state():
    command = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    result = {}
    for line in output.splitlines():
        index, free, utilization = (int(item.strip()) for item in line.split(","))
        result[index] = {"memory_free_mib": free, "utilization": utilization}
    return result


def available_idle_gpus(allowed, occupied):
    state = gpu_state()
    return [gpu for gpu in allowed if gpu not in occupied and gpu in state
            and state[gpu]["memory_free_mib"] >= 9000 and state[gpu]["utilization"] <= 10]


def persist(queue, running):
    payload = {
        "updated_unix": time.time(),
        "runner_pid": os.getpid(),
        "cohort_sha256": COHORT_DIGEST,
        "jobs": queue,
        "running": {key: {"pid": value["process"].pid, "gpu": value["gpu"]} for key, value in running.items()},
    }
    write_json(QUEUE_PATH, payload)
    write_json(STATUS_PATH, payload)


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LAUNCHER_LOG.open("a") as handle:
        handle.write(line + "\n")


def record_source_state():
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=PROJECT_ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=PROJECT_ROOT, text=True
    ).splitlines()
    dependency_inventory = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--path", str(PYTHON_DEPS), "--format=freeze"], text=True
    )
    (ARTIFACT_ROOT / "environment.txt").write_text(dependency_inventory)
    write_json(SOURCE_STATE, {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_benchmark_sources": [p for p in untracked if p.startswith(("baselines/", "scripts/", "tests/", "data_processing/"))],
        "python_executable": sys.executable,
        "python_dependencies_path": str(PYTHON_DEPS),
        "runtime": runtime_metadata(),
    })


def aggregate(jobs):
    rows = []
    for job in jobs:
        if job["status"] != "complete":
            continue
        root = Path(job["output_dir"])
        metrics = json.loads((root / "metrics.json").read_text())
        metadata = json.loads((root / "metadata.json").read_text())
        rows.append({"method": job["method"], "seed": job["seed"], **metrics,
                     "wall_clock_seconds": metadata.get("wall_clock_seconds"),
                     "peak_allocated_cuda_bytes": metadata.get("peak_allocated_cuda_bytes")})
    if not rows:
        return
    with (ARTIFACT_ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# UTKFace 5% benchmark summary", "", "Sample standard deviations use `ddof=1`.", "",
             "| Method | Successful | Failed | Test MAE mean ± SD | Test R² mean ± SD | Mean runtime (s) | Peak GPU memory (bytes) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        failures = sum(job["method"] == method and job["status"] == "failed" for job in jobs)
        if subset:
            maes = [float(row["test_mae"]) for row in subset]; r2s = [float(row["test_r2"]) for row in subset]
            sd_mae = statistics.stdev(maes) if len(maes) > 1 else float("nan")
            sd_r2 = statistics.stdev(r2s) if len(r2s) > 1 else float("nan")
            runtimes = [float(row["wall_clock_seconds"]) for row in subset if row["wall_clock_seconds"] is not None]
            peaks = [int(row["peak_allocated_cuda_bytes"]) for row in subset if row["peak_allocated_cuda_bytes"] is not None]
            lines.append(f"| {method} | {len(subset)} | {failures} | {statistics.mean(maes):.6f} ± {sd_mae:.6f} | {statistics.mean(r2s):.6f} ± {sd_r2:.6f} | {statistics.mean(runtimes) if runtimes else float('nan'):.1f} | {max(peaks) if peaks else 0} |")
    (ARTIFACT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNNER_PID.write_text(f"{os.getpid()}\n")
    record_source_state()
    jobs = []
    for method in args.methods:
        for seed in args.seeds:
            out = output_dir(method, seed, args.smoke)
            cwd, command = command_for(method, seed, args.smoke)
            jobs.append({"id": f"{method}:seed_{seed}", "method": method, "seed": seed,
                         "output_dir": str(out), "cwd": str(cwd), "command": command,
                         "status": "complete" if args.resume and run_complete(out) else "pending", "attempts": 0})
    if args.dry_run:
        for job in jobs:
            print(f"cd {shlex.quote(job['cwd'])} && CUDA_VISIBLE_DEVICES=<gpu> " + shlex.join(job["command"]))
        persist(jobs, {})
        return

    running = {}
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        del signum, frame
        stopping = True
        log("Stop requested; no new jobs will launch. Running jobs are left intact until completion.")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    persist(jobs, running)
    while True:
        for job_id, record in list(running.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["log_handle"].close()
            job = record["job"]
            if returncode == 0 and run_complete(job["output_dir"]):
                job["status"] = "complete"
                log(f"Completed {job_id} on GPU {record['gpu']}")
            elif job["attempts"] < 2:
                job["status"] = "pending"
                job["last_failure"] = f"exit code {returncode}; retry scheduled"
                log(f"Failed {job_id} with exit {returncode}; scheduling one retry")
            else:
                job["status"] = "failed"
                job["last_failure"] = f"exit code {returncode}; retries exhausted"
                log(f"Failed {job_id}; retries exhausted")
            del running[job_id]
            persist(jobs, running)

        pending = [job for job in jobs if job["status"] == "pending"]
        if not running and not pending:
            break
        if not stopping and pending and len(running) < args.max_parallel:
            idle = available_idle_gpus(args.available_gpus, {item["gpu"] for item in running.values()})
            for gpu, job in zip(idle[:args.max_parallel - len(running)], pending):
                output = Path(job["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                log_handle = (output / "run.log").open("a")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["UTKFACE_BENCHMARK_ROOT"] = str(PROJECT_ROOT)
                env["MPLCONFIGDIR"] = str(ARTIFACT_ROOT / "matplotlib")
                env["NUMBA_CACHE_DIR"] = str(ARTIFACT_ROOT / "numba_cache")
                Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
                Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(PYTHON_DEPS), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
                )
                job["status"] = "running"
                job["attempts"] += 1
                job["gpu"] = gpu
                process = subprocess.Popen(job["command"], cwd=job["cwd"], env=env,
                                           stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
                running[job["id"]] = {"process": process, "gpu": gpu, "job": job, "log_handle": log_handle}
                log(f"Started {job['id']} pid={process.pid} GPU={gpu}: {shlex.join(job['command'])}")
                persist(jobs, running)
        time.sleep(args.poll_seconds)
    persist(jobs, running)
    log("Queue finished")
    if not args.smoke:
        aggregate(jobs)
    if any(job["status"] == "failed" for job in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
