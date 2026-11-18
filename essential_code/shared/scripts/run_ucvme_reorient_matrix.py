#!/usr/bin/env python3
"""Detached, validating queue for the 48-cell UCVME + Reorient matrix."""
from __future__ import annotations

import csv
import signal
import sys
import traceback
import uuid
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/nobackup/enzez/reasoning-chains/.venv/bin/python")
QUEUE = ROOT / "artifacts/benchmark_queues/ucvme_reorient_5pct"
OUTPUT = ROOT / "artifacts/ucvme_reorient_5pct"
UCVME = ROOT / "third_party/UCVME"
DATA = {
    "utkface": ROOT / "data/utkface_all",
    "imdb_wiki": Path("/nobackup/enzez/Representation-Anchored-Pseudo-Labeling-a40-data/imdb_wiki"),
    "rsna": Path("/nobackup/enzez/Representation-Anchored-Pseudo-Labeling-a40-data/rsna_bone_age/2017/extracted"),
    "chexchonet_lvidd": Path("/nobackup/enzez/Representation-Anchored-Pseudo-Labeling-a40-data/chexchonet/1.0.0"),
}
BACKBONES = ("resnet50", "dinov2_vits14")
THREAD_ENV = {name: "1" for name in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
ALLOWED_GPUS = (0, 1, 2, 3, 4)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def manifest(dataset: str, seed: int) -> Path:
    if dataset in ("utkface", "imdb_wiki"):
        return ROOT / f"data_processing/splits/{dataset}_ratio_0.05_seed_{seed}.json"
    if dataset == "rsna":
        return DATA[dataset].parent / f"protocol/ratio_0.05_seed_{seed}.json"
    return ROOT / f"artifacts/protected_protocol/chexchonet_lvidd/chexchonet_lvidd_ratio_0.05_seed_{seed}.json"


def jobs() -> list[dict]:
    return [{"id": f"{dataset}:{backbone}:seed_{seed}", "dataset": dataset,
             "backbone": backbone, "seed": seed, "status": "pending"}
            for dataset in DATA for backbone in BACKBONES for seed in range(6)]


def target(job: dict) -> Path:
    return OUTPUT / job["dataset"] / job["backbone"] / f"seed_{job['seed']}"


def valid(path: Path, job: dict) -> tuple[bool, str]:
    required = ("best.pt", "config.json", "metadata.json", "metrics.json",
                "history.csv", "test_predictions.npz")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return False, f"missing {missing}"
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        config = json.loads((path / "config.json").read_text())
        spec = json.loads(manifest(job['dataset'], job['seed']).read_text())
        with (path / 'history.csv').open() as handle:
            history = list(csv.DictReader(handle))
        assert [int(row['epoch']) for row in history] == list(range(1, 31))
        selected = min(range(30), key=lambda i: float(history[i]['validation_mae']))
        assert selected == metadata['best_epoch'] == metrics['best_epoch']
        assert float(history[selected]['validation_mae']) == metrics['validation_mae'] == metadata['validation_mae']
        assert metadata['method'] == 'ucvme' and metadata['test_evaluations'] == 1
        assert spec['labeled_ratio'] == 0.05 and spec['seed'] == job['seed']
        for key in ('protocol_version', 'transform_version', 'counts'):
            assert metadata[key] == spec[key], key
        assert metadata['cohort_sha256'] == spec.get('cohort_sha256', spec.get('cohort_digest'))
        for key in ('split_digest', 'manifest_digest'):
            if key in spec:
                assert metadata[key] == spec[key]
        if job['dataset'] == 'chexchonet_lvidd':
            assert metadata['manifest_sha256'] == spec['manifest_sha256']
        argv = config['argv']
        expected = command(job, Path(argv[argv.index('--benchmark_output_dir') + 1]))[1:]
        assert argv == expected, 'noncanonical training arguments'
        assert (path / 'best.pt').stat().st_size == metadata['checkpoint_size_bytes'] > 0
        # Locally produced checkpoints contain NumPy RNG state. CPU load only, no inference.
        import torch
        checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == selected
        assert checkpoint['validation_mae_years'] == metrics['validation_mae']
        assert checkpoint['state_dict'] and checkpoint['state_dict_1']
        assert all(torch.isfinite(t).all() for key in ('state_dict', 'state_dict_1') for t in checkpoint[key].values())
        del checkpoint
        if job['dataset'] == 'rsna':
            cohort = json.loads((manifest(job['dataset'], job['seed']).parent / 'cohort.json').read_text())
            indices = cohort['splits']['test']
        else:
            indices = spec.get('indices', spec.get('splits'))['test']
        with np.load(path / 'test_predictions.npz', allow_pickle=False) as archive:
            predictions = {key: archive[key] for key in archive.files}
        assert np.array_equal(predictions['cohort_indices'], indices)
        assert len(predictions['predictions']) == len(predictions['targets']) == len(indices) > 0
        assert np.isfinite(predictions['predictions']).all() and np.isfinite(predictions['targets']).all()
        checks = (
            metadata.get("status") == "complete",
            metadata.get("plugin") == "reorient",
            metadata.get("supervised_recycling") is True,
            metadata.get("preserve_full_unlabeled_trajectory") is True,
            metadata.get("combined_same_optimizer_update") is True,
            metadata.get("additional_supervised_optimizer_step") is False,
            metadata.get("checkpoint_reloaded") is True,
            metadata.get("test_used_for_selection") is False,
            int(metadata.get("test_model_inference_count", -1)) == 1,
            int(metadata.get("manifest_seed", -1)) == job["seed"],
            Path(metadata.get("manifest_path", "")).resolve() == manifest(job["dataset"], job["seed"]).resolve(),
            metadata.get("target_backbone") == job["backbone"],
            all(math.isfinite(float(metrics[key])) for key in ("validation_mae", "test_mae", "test_r2")),
            len(predictions["predictions"]) > 0,
        )
        return (True, "validated") if all(checks) else (False, "provenance validation failed")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def initialize() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    current = jobs()
    for job in current:
        if not manifest(job["dataset"], job["seed"]).is_file():
            raise FileNotFoundError(manifest(job["dataset"], job["seed"]))
        if not DATA[job["dataset"]].is_dir():
            raise FileNotFoundError(DATA[job["dataset"]])
        ok, reason = valid(target(job), job) if target(job).exists() else (False, "absent")
        job.update(status="complete" if ok else "pending", validation=reason)
    atomic_json(QUEUE / "queue.json", {"version": "ucvme-reorient-5pct-v1", "jobs": current})
    atomic_json(QUEUE / "protocol.json", {
        "cells": len(current), "datasets": list(DATA), "backbones": list(BACKBONES),
        "seeds": list(range(6)), "labeled_ratio": 0.05,
        "change_scope": "supervised anchoring schedule only",
        "full_unlabeled_trajectory": True, "labeled_recycling": True,
        "same_optimizer_update": True, "standalone_supervised_step": False,
        "thread_limits": THREAD_ENV, "gpu_admission_used_memory_mib": 20000,
    })
    print(json.dumps({"valid": sum(j["status"] == "complete" for j in current),
                      "missing": sum(j["status"] != "complete" for j in current)}))
    return 0


def gpu_memory() -> dict[int, int]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                             "--format=csv,noheader,nounits"], text=True,
                            stdout=subprocess.PIPE, check=True)
    return {int(line.split(",")[0]): int(line.split(",")[1])
            for line in result.stdout.splitlines() if line.strip()}


def command(job: dict, attempt: Path) -> list[str]:
    args = [str(PYTHON), "ucvme_age.py", "--benchmark_manifest",
            str(manifest(job["dataset"], job["seed"])), "--benchmark_output_dir",
            str(attempt), "--data_dir", str(DATA[job["dataset"]]), "--seed",
            str(job["seed"]), "--benchmark_epochs", "30", "--batch_size",
            "16" if job["dataset"] == "rsna" else "32", "--backbone",
            job["backbone"], "--reorient"]
    return args


MAX_ATTEMPTS = 2


def alive(pid, token):
    try:
        proc = Path('/proc') / str(pid)
        return proc.joinpath('stat').read_text().split(') ', 1)[1].split()[0] != 'Z' and token in proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (OSError, ValueError):
        return False


def worker(attempt):
    # Holding this lock across launch and wait closes controller-crash duplicate races.
    with (attempt / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (attempt / 'exit.json').exists():
            return 0
        launch = json.loads((attempt / 'launch.json').read_text())
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env.update({'CUDA_VISIBLE_DEVICES': str(launch['physical_gpu']),
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'UTKFACE_BENCHMARK_ROOT': str(ROOT),
                    'PYTHONPATH': str(ROOT), 'PYTHONUNBUFFERED': '1',
                    'HF_HOME': '/nobackup/enzez/hf_cache', 'HF_HUB_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1'})
        try:
            with (attempt / 'run.log').open('x') as log:
                trainer = subprocess.Popen(command(launch, attempt), cwd=UCVME, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                atomic_json(attempt / 'process.json', {'worker_pid': os.getpid(),
                            'trainer_pid': trainer.pid, 'started_unix': time.time()})
                code = trainer.wait()
            result = {'exit_code': code, 'signal': -code if code < 0 else None}
        except Exception:
            result = {'exit_code': None, 'signal': None, 'exception': traceback.format_exc()}
        result.update(finished_unix=time.time(), physical_gpu=launch['physical_gpu'])
        atomic_json(attempt / 'exit.json', result)
    return 0


def failure(job, attempt, result, reason):
    log = attempt / 'run.log'
    tail = ''
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 16000))
            tail = handle.read().decode(errors='replace')
    record = dict(result, attempt_directory=str(attempt), physical_gpu=job['physical_gpu'],
                  started_unix=job.get('started_unix'), reason=reason, log_tail=tail)
    job.setdefault('attempt_history', []).append(record)
    job.update(status='pending' if job.get('recovery_attempts', 0) < MAX_ATTEMPTS else 'failed',
               validation=reason, exit_code=result.get('exit_code'), signal=result.get('signal'))


def controller():
    with (QUEUE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (QUEUE / 'controller.pid').write_text(str(os.getpid()) + '\n')
        state = json.loads((QUEUE / 'queue.json').read_text())
        children = {}
        while True:
            for child in list(children.values()):
                child.poll()  # Reap children; adopted workers are checked through /proc.
            for job in state['jobs']:
                if job['status'] != 'running':
                    continue
                attempt = Path(job['attempt_directory'])
                process_file = attempt / 'process.json'
                if process_file.exists():
                    job.update(json.loads(process_file.read_text()))
                if alive(job.get('worker_pid', -1), str(attempt)) or alive(job.get('trainer_pid', -1), str(attempt)):
                    continue
                # Recover the launch/PID-write crash window without creating another attempt.
                survivors = [int(p.name) for p in Path('/proc').glob('[0-9]*') if alive(int(p.name), str(attempt))]
                if survivors:
                    job['worker_pid'] = survivors[0]
                    continue
                # A persisted launch may precede worker creation. Give that transition a bounded grace.
                if time.time() - job.get('started_unix', 0) < 10:
                    continue
                result = json.loads((attempt / 'exit.json').read_text()) if (attempt / 'exit.json').exists() else {
                    'exit_code': None, 'signal': None, 'finished_unix': time.time()}
                destination = target(job)
                ok, reason = valid(destination if destination.exists() else attempt, job)
                if ok:
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.rename(attempt, destination)
                    job.update(status='complete', validation='validated', **result)
                else:
                    failure(job, attempt, result, result.get('exception', reason))
                atomic_json(QUEUE / 'queue.json', state)
            occupied = {j['physical_gpu'] for j in state['jobs'] if j['status'] == 'running'}
            for gpu in ALLOWED_GPUS:
                pending = next((j for j in state['jobs'] if j['status'] == 'pending'), None)
                if gpu in occupied or pending is None:
                    continue
                job = pending
                if target(job).exists():
                    ok, reason = valid(target(job), job)
                    job.update(status='complete' if ok else 'failed', validation=reason)
                    atomic_json(QUEUE / 'queue.json', state)
                    continue
                attempt = OUTPUT / '_attempts' / job['dataset'] / job['backbone'] / f"seed_{job['seed']}" / f"attempt_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
                attempt.mkdir(parents=True, exist_ok=False)
                job.update(status='running', physical_gpu=gpu, attempt_directory=str(attempt),
                           started_unix=time.time(), recovery_attempts=job.get('recovery_attempts', 0) + 1)
                for key in ('trainer_pid', 'worker_pid', 'exit_code', 'signal', 'finished_unix'):
                    job.pop(key, None)
                atomic_json(attempt / 'launch.json', job)
                atomic_json(QUEUE / 'queue.json', state)
                try:
                    child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker', str(attempt)],
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                    children[gpu] = child
                    job['worker_pid'] = child.pid
                except Exception:
                    failure(job, attempt, {'exit_code': None, 'signal': None, 'finished_unix': time.time()}, traceback.format_exc())
                atomic_json(QUEUE / 'queue.json', state)
            active = [j for j in state['jobs'] if j['status'] == 'running']
            done = not active and not any(j['status'] == 'pending' for j in state['jobs'])
            atomic_json(QUEUE / 'controller_state.json', {'pid': os.getpid(), 'updated_unix': time.time(),
                        'status': 'finished' if done else 'running', 'running': active})
            if done:
                return 0
            time.sleep(0.5)


def supervise():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (QUEUE / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for restart in range(10):
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--controller'], stdin=subprocess.DEVNULL)
            atomic_json(QUEUE / 'supervisor_state.json', {'pid': os.getpid(), 'controller_pid': child.pid,
                        'restart': restart, 'started_unix': time.time()})
            code = child.wait()
            if code == 0:
                return 0
            atomic_json(QUEUE / 'supervisor_exit.json', {'exit_code': code, 'restart': restart, 'time': time.time()})
            time.sleep(2)
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.supervise:
        return supervise()
    if args.controller:
        return controller()
    if args.initialize:
        if (QUEUE / 'queue.json').exists():
            raise RuntimeError('refusing to overwrite an existing campaign')
        return initialize()
    parser.error('choose a mode')


if __name__ == '__main__':
    raise SystemExit(main())
