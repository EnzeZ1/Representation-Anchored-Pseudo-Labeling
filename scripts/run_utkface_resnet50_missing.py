#!/usr/bin/env python3
"""Run only the five audited missing ResNet-50 UTKFace experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.benchmark_io import REQUIRED_RUN_FILES, write_json

QUEUE = ROOT / "artifacts/benchmark_queues/utkface_resnet50_missing_runs"
BENCH = ROOT / "artifacts/benchmarks/utkface/resnet50"
OLD_5PCT = ROOT / "artifacts/utkface_5pct"
DEPS = OLD_5PCT / "python_deps"
STATE = QUEUE / "queue_state.json"
STATUS = QUEUE / "run_status.json"
CONFIG = QUEUE / "queue_config.json"
LOG = QUEUE / "launcher.log"
DIGEST = "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56"
JOBS = (("rapl", .05, 5), ("hpl", .05, 5), ("rapl", .20, 1), ("hpl", .20, 0), ("hpl", .20, 2))
FAILED_PREVIOUS = {
    ("rapl", 1): {"pid": 2762668, "gpu": 3, "failure_timestamp": "2026-07-22 12:11:37"},
    ("hpl", 0): {"pid": 3817726, "gpu": 2, "failure_timestamp": "2026-07-22 14:39:41"},
    ("hpl", 2): {"pid": 3917516, "gpu": 4, "failure_timestamp": "2026-07-22 14:53:48"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--available_gpus", nargs="+", type=int, default=list(range(6)))
    parser.add_argument("--poll_seconds", type=int, default=20)
    return parser.parse_args()


def manifest(ratio, seed):
    return ROOT / f"data_processing/splits/utkface_ratio_{ratio:.2f}_seed_{seed}.json"


def canonical_output(method, ratio, seed):
    if ratio == .05:
        return OLD_5PCT / method / f"seed_{seed}"
    return BENCH / f"ratio_{ratio:.2f}" / method / f"seed_{seed}"


def strict_complete(path):
    path = Path(path)
    if not all((path / name).is_file() for name in REQUIRED_RUN_FILES):
        return False
    try:
        metrics = json.loads((path / "metrics.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        return bool(
            metadata["status"] == "complete"
            and metadata["checkpoint_reloaded"] is True
            and metadata["test_used_for_selection"] is False
            and metadata["test_evaluations"] == 1
            and all(math.isfinite(float(metrics[key])) for key in ("validation_mae", "test_mae", "test_r2"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def command(method, ratio, seed, output):
    common = [
        "--benchmark_manifest", str(manifest(ratio, seed)),
        "--benchmark_output_dir", str(output),
        "--data_dir", str(ROOT / "data/utkface_all"),
        "--seed", str(seed),
    ]
    if method == "rapl":
        return ROOT, [
            sys.executable, str(ROOT / "train.py"), "-dataset", "utkface",
            "--method", "probe", "--labeled_ratio", str(ratio),
            "--backbone", "resnet50", "--probe_backbone", "resnet50",
            "--save", str(output / "best.pt"), *common,
        ]
    cwd = ROOT / "third_party/Heteroscedastic-Pseudo-Labels/utkface"
    return cwd, [sys.executable, "main_ours.py", "--benchmark_backbone", "resnet50", *common]


def log(message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def gpu_inventory():
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    )
    states = []
    for line in query.splitlines():
        index, uuid, free, utilization = (item.strip() for item in line.split(","))
        states.append({"index": int(index), "uuid": uuid, "free_mib": int(free), "utilization": int(utilization)})
    try:
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True
        )
    except subprocess.CalledProcessError:
        processes = ""
    busy = {}
    for line in processes.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = (item.strip() for item in line.split(",", 3))
        busy.setdefault(uuid, []).append({"pid": int(pid), "process_name": name, "used_memory_mib": memory})
    for state in states:
        state["compute_processes"] = busy.get(state["uuid"], [])
    return states


def idle_gpus(allowed, reserved):
    return [
        state for state in gpu_inventory()
        if state["index"] in allowed
        and state["index"] not in reserved
        and not state["compute_processes"]
        and state["free_mib"] >= 10000
        and state["utilization"] <= 10
    ]


def archive_previous_failures():
    candidates = [(method, .20, seed) for method, seed in FAILED_PREVIOUS]
    existing = [(method, ratio, seed, canonical_output(method, ratio, seed))
                for method, ratio, seed in candidates
                if canonical_output(method, ratio, seed).exists()
                and not strict_complete(canonical_output(method, ratio, seed))]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_root = BENCH / "archive_failed_attempts" / stamp
    records = []
    for method, ratio, seed, source in existing:
        destination = archive_root / f"ratio_{ratio:.2f}" / method / f"seed_{seed}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_log = source / "run.log"
        traceback = run_log.read_text(errors="replace") if run_log.exists() else None
        shutil.move(str(source), str(destination))
        previous = FAILED_PREVIOUS[(method, seed)]
        records.append({
            "original_path": str(source),
            "archived_path": str(destination),
            "previous_pid": previous["pid"],
            "previous_gpu": previous["gpu"],
            "failure_timestamp": previous["failure_timestamp"],
            "oom_traceback": traceback,
            "retry_blocked_reason": "nonempty incomplete output directory collision",
            "status": "failed_attempt_archived",
        })
    write_json(archive_root / "archive_manifest.json", {"created_unix": time.time(), "entries": records})
    log(f"Archived {len(records)} failed partial directories under {archive_root}")
    return archive_root


def persist(jobs, running, archive_root):
    payload = {
        "updated_unix": time.time(),
        "runner_pid": os.getpid(),
        "stage": "formal",
        "job_count": len(jobs),
        "cohort_sha256": DIGEST,
        "archive_root": str(archive_root) if archive_root else None,
        "jobs": jobs,
        "running": {
            job_id: {"pid": record["process"].pid, "physical_gpu_index": record["gpu"]["index"],
                     "gpu_uuid": record["gpu"]["uuid"], "attempt": record["job"]["attempts"]}
            for job_id, record in running.items()
        },
    }
    write_json(STATE, payload)
    write_json(STATUS, payload)


def main():
    options = parse_args()
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / "runner.pid").write_text(str(os.getpid()) + "\n")
    archive_root = None
    jobs = []
    for method, ratio, seed in JOBS:
        output = canonical_output(method, ratio, seed)
        jobs.append({
            "id": f"{method}:ratio_{ratio:.2f}:seed_{seed}",
            "experiment_id": f"utkface-r{ratio:.2f}-resnet50-{method}-seed-{seed}",
            "dataset": "UTKFace", "method": method, "labeled_ratio": ratio, "rng_seed": seed,
            "target_backbone": "ImageNet-pretrained ResNet-50",
            "probe_backbone": "separately instantiated frozen ImageNet-pretrained ResNet-50" if method == "rapl" else None,
            "manifest_path": str(manifest(ratio, seed)), "artifact_directory": str(output),
            "status": "complete" if options.resume and strict_complete(output) else "pending",
            "attempts": 0, "attempt_history": [],
        })
    assert len(jobs) == 5 and len({job["experiment_id"] for job in jobs}) == 5
    assert {(job["method"], job["labeled_ratio"], job["rng_seed"]) for job in jobs} == set(JOBS)
    write_json(CONFIG, {
        "job_count": 5, "jobs": jobs, "cohort_sha256": DIGEST, "max_attempts": 2,
        "attempt_policy": "queue-owned attempt directories; promote only after strict integrity validation",
    })
    if options.dry_run:
        for job in jobs:
            attempt = QUEUE / "attempts" / job["id"].replace(":", "_") / "attempt_1"
            cwd, cmd = command(job["method"], job["labeled_ratio"], job["rng_seed"], attempt)
            print(f"cd {shlex.quote(str(cwd))} && CUDA_VISIBLE_DEVICES=<idle-gpu> {shlex.join(cmd)}")
        return
    archive_root = archive_previous_failures()
    running = {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        log("Stop requested; running children will continue and no new jobs will launch.")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    persist(jobs, running, archive_root)
    while True:
        for job_id, record in list(running.items()):
            rc = record["process"].poll()
            if rc is None:
                continue
            record["handle"].close()
            job = record["job"]
            attempt_dir = record["attempt_dir"]
            valid = rc == 0 and strict_complete(attempt_dir)
            job["attempt_history"][-1].update({"exit_code": rc, "ended_unix": time.time(), "integrity_valid": valid})
            if valid:
                canonical = Path(job["artifact_directory"])
                if canonical.exists():
                    raise RuntimeError(f"Refusing to overwrite canonical output: {canonical}")
                canonical.parent.mkdir(parents=True, exist_ok=True)
                os.replace(attempt_dir, canonical)
                job["status"] = "complete"
                job["promoted_from"] = str(attempt_dir)
                log(f"Completed and promoted {job_id} from attempt {job['attempts']} GPU={record['gpu']['index']}")
                subprocess.run([sys.executable, str(ROOT / "scripts/update_benchmark_registry.py")], cwd=ROOT, check=False)
            elif job["attempts"] < 2:
                job["status"] = "pending"
                job["last_failure"] = f"attempt {job['attempts']} exit {rc}; retry scheduled"
                log(f"Failed {job_id} attempt {job['attempts']} exit={rc}; scheduling one retry")
            else:
                job["status"] = "failed"
                job["last_failure"] = f"attempt {job['attempts']} exit {rc}; retries exhausted"
                log(f"Failed {job_id}; retries exhausted")
            del running[job_id]
            persist(jobs, running, archive_root)
        unfinished = [job for job in jobs if job["status"] in ("pending", "running")]
        if not running and not unfinished:
            break
        pending = [job for job in jobs if job["status"] == "pending"]
        if not stopping and pending and len(running) < options.max_parallel:
            available = idle_gpus(options.available_gpus, {record["gpu"]["index"] for record in running.values()})
            for gpu, job in zip(available, pending[:options.max_parallel - len(running)]):
                attempt_number = job["attempts"] + 1
                attempt_dir = QUEUE / "attempts" / job["id"].replace(":", "_") / f"attempt_{attempt_number}"
                if attempt_dir.exists():
                    raise RuntimeError(f"Attempt directory already exists: {attempt_dir}")
                attempt_dir.mkdir(parents=True)
                cwd, cmd = command(job["method"], job["labeled_ratio"], job["rng_seed"], attempt_dir)
                environment = os.environ.copy()
                environment.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
                    "UTKFACE_BENCHMARK_ROOT": str(ROOT),
                    "PYTHONPATH": os.pathsep.join([str(DEPS), str(ROOT), environment.get("PYTHONPATH", "")]),
                    "MPLCONFIGDIR": str(QUEUE / "matplotlib"),
                    "NUMBA_CACHE_DIR": str(QUEUE / "numba_cache"),
                    "PYTHONUNBUFFERED": "1",
                })
                Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
                Path(environment["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
                handle = (attempt_dir / "run.log").open("a")
                process = subprocess.Popen(cmd, cwd=cwd, env=environment, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                job["attempts"] = attempt_number
                job["status"] = "running"
                job["physical_gpu_index"] = gpu["index"]
                job["gpu_uuid"] = gpu["uuid"]
                job["cuda_visible_devices"] = str(gpu["index"])
                job["process_local_device"] = "cuda:0"
                job["pid"] = process.pid
                job["attempt_history"].append({
                    "attempt": attempt_number, "attempt_directory": str(attempt_dir),
                    "started_unix": time.time(), "pid": process.pid,
                    "physical_gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"],
                    "free_memory_mib_before_launch": gpu["free_mib"],
                    "utilization_before_launch": gpu["utilization"],
                    "compute_processes_before_launch": gpu["compute_processes"],
                    "cuda_visible_devices": str(gpu["index"]), "process_local_device": "cuda:0",
                    "command": cmd,
                })
                running[job["id"]] = {
                    "process": process, "handle": handle, "gpu": gpu, "job": job,
                    "attempt_dir": attempt_dir,
                }
                log(f"Started {job['id']} attempt={attempt_number} pid={process.pid} "
                    f"GPU={gpu['index']} UUID={gpu['uuid']} free={gpu['free_mib']}MiB: {shlex.join(cmd)}")
                persist(jobs, running, archive_root)
        time.sleep(options.poll_seconds)
    persist(jobs, running, archive_root)
    if all(job["status"] == "complete" for job in jobs):
        for script in ("update_benchmark_registry.py", "report_resnet50_utkface.py", "report_final_backbone_tables.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    log("Missing ResNet-50 formal queue finished")
    raise SystemExit(1 if any(job["status"] == "failed" for job in jobs) else 0)


if __name__ == "__main__":
    main()
