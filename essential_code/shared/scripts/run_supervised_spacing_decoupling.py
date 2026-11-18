#!/usr/bin/env python3
"""Persistent resumable GPU queue for matched-budget supervised spacing."""
import argparse, fcntl, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/supervised_spacing_decoupling_utkface_5pct"
Q = ART / "queue"; STATE = Q / "queue_state.json"; LOCK = Q / "queue.lock"
PY = Path("/nobackup/enzez/.venv/bin/python")
TRAIN = ROOT / "training/supervised_spacing_decoupling.py"
BACKBONES = ("resnet50", "dinov2"); SEEDS = (0, 1, 2)
SCHEDULES = ("evenly_interleaved", "front_clustered", "random_interleaved")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)


def complete(run):
    try: status = json.loads((run / "status.json").read_text())
    except (OSError, json.JSONDecodeError): return False
    return (status.get("training_status") == "complete" and status.get("final_evaluation_status") == "complete"
            and (run / "final_test_complete").is_file() and (run / "metrics.json").is_file())


def locked(fn):
    LOCK.touch(exist_ok=True)
    with LOCK.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX); state = json.loads(STATE.read_text()); result = fn(state)
        state["updated_unix"] = time.time(); write(STATE, state); return result


def claim(gpu):
    def mutate(state):
        for job in state["jobs"]:
            if job["training_status"] == "pending":
                job.update(training_status="running", attempts=job["attempts"] + 1, pid=os.getpid(),
                           physical_gpu=gpu, cuda_visible_devices=str(gpu), start_time=time.time())
                attempt = Q / "attempts" / job["id"] / f"attempt_{job['attempts']}"
                attempt.mkdir(parents=True, exist_ok=False); job["attempt_directory"] = str(attempt)
                state["workers"][str(gpu)] = {"pid": os.getpid(), "job_id": job["id"], "status": "running"}
                return dict(job)
        state["workers"][str(gpu)] = {"pid": os.getpid(), "job_id": None, "status": "complete"}; return None
    return locked(mutate)


def finish(job, code):
    run = Path(job["attempt_directory"]) / "run"
    try: saved = json.loads((run / "status.json").read_text())
    except (OSError, json.JSONDecodeError): saved = {}
    def mutate(state):
        entry = next(x for x in state["jobs"] if x["id"] == job["id"])
        entry.update(training_status=saved.get("training_status", "failed"),
                     final_evaluation_status=saved.get("final_evaluation_status", "failed"),
                     analysis_status=saved.get("analysis_status", "pending"), exit_code=code, end_time=time.time())
        if complete(run): entry["artifact_directory"] = str(run)
        state["workers"][str(job["physical_gpu"])] = {"pid": os.getpid(), "job_id": None, "status": "idle"}
    locked(mutate)


def reconcile():
    def mutate(state):
        for job in state["jobs"]:
            if job["training_status"] != "running": continue
            try: os.kill(int(job.get("pid", -1)), 0); alive = True
            except (OSError, ValueError): alive = False
            if alive: continue
            run = Path(job.get("attempt_directory", "")) / "run"
            if complete(run):
                job.update(training_status="complete", final_evaluation_status="complete",
                           artifact_directory=str(run), reconciliation="preserved_valid_completion")
            else: job.update(training_status="failed", reconciliation="stale_worker_not_requeued")
    locked(mutate)


def worker(gpu):
    reconcile(); env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    while True:
        job = claim(gpu)
        if job is None: return
        attempt = Path(job["attempt_directory"])
        command = [str(PY), str(TRAIN), "--backbone", job["backbone"], "--seed", str(job["seed"]),
                   "--schedule", job["schedule"], "--attempt", str(job["attempts"]),
                   "--config-hash", job["config_hash"], "--output", str(attempt / "run")]
        code = 255
        try:
            with (attempt / "run.log").open("a") as log:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        finally: finish(job, code)


def prepare(u, l):
    if ART.exists(): raise RuntimeError(f"refusing to overwrite existing experiment root: {ART}")
    import torch, torchvision
    jobs = [{"id": f"uniform__{b}__seed_{s}__{schedule}", "method": "uniform", "backbone": b,
             "seed": s, "schedule": schedule, "attempts": 0, "training_status": "pending",
             "final_evaluation_status": "pending", "analysis_status": "pending"}
            for b in BACKBONES for s in SEEDS for schedule in SCHEDULES]
    if len({j["id"] for j in jobs}) != len(jobs): raise AssertionError("duplicate run keys")
    seeds = {}
    for seed in SEEDS:
        path = ROOT / f"data_processing/splits/utkface_ratio_0.05_seed_{seed}.json"
        manifest = json.loads(path.read_text())
        seeds[str(seed)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "scaler": manifest["label_scaler"]}
    scientific = {
        "method": "uniform", "dataset": "UTKFace", "ratio": .05,
        "cohort_digest": "61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56",
        "seed_manifests": seeds, "backbones": list(BACKBONES), "seeds": list(SEEDS), "schedules": list(SCHEDULES),
        "U": u, "L": l, "epochs": 30, "lambda_u": 1.0, "r": 1.0,
        "unlabeled_updates_per_epoch": u, "supervised_updates_per_epoch": l,
        "optimizer_steps_per_epoch": u, "labeled_loader_passes_per_epoch": 1,
        "schedule_definitions": {
            "evenly_interleaved": "sorted floor((j+0.5)*U/L), j=0..L-1",
            "front_clustered": "0..L-1",
            "random_interleaved": "uniform L-subset without replacement; isolated SeedSequence([0x53504143,seed,epoch])",
        },
        "canonical_configuration_source": "training/canonical_uniform_frequency.py",
        "diagnostic_checkpoint_epochs": [1, 5, 10, 20, 30], "jobs": [j["id"] for j in jobs],
    }
    blob = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode(); config_hash = hashlib.sha256(blob).hexdigest()
    for job in jobs: job["config_hash"] = config_hash
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout
    manifest = {"experiment": "supervised_spacing_decoupling_utkface_5pct", "created_unix": time.time(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "uncommitted_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "runtime": {"python": sys.version, "pytorch": torch.__version__, "torchvision": torchvision.__version__,
                            "cuda": torch.version.cuda, "platform": platform.platform()},
                "scientific_configuration": scientific, "config_hash": config_hash,
                "posthoc": {"reference_drift": {"epochs": [1, 5, 10, 20, 30],
                            "reference_subset": "artifacts/anchor_mechanism_utkface_5pct/diagnostics/reference_subset.json",
                            "definition": "mean_R (f_theta_t(x)-y)^2", "test_data": False}}}
    write(ART / "frozen_manifest.json", manifest)
    from training.supervised_spacing_decoupling import validate_schedules
    write(ART / "schedule_validation.json", validate_schedules(u, l, seed=0))
    write(STATE, {"experiment": manifest["experiment"], "jobs": jobs, "workers": {}, "created_unix": time.time()})
    print(config_hash)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--prepare", action="store_true"); p.add_argument("--U", type=int); p.add_argument("--L", type=int); p.add_argument("--worker", type=int)
    a = p.parse_args()
    if a.prepare:
        if a.U is None or a.L is None: p.error("--prepare requires --U and --L from loader validation")
        prepare(a.U, a.L)
    elif a.worker is not None: worker(a.worker)
    else: p.error("mode required")


if __name__ == "__main__": main()
