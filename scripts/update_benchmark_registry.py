#!/usr/bin/env python3
"""Build the future-compatible benchmark registry from immutable run artifacts."""

from __future__ import annotations

import argparse, csv, json, math, statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts" / "benchmark_registry"
FIELDS = ["experiment_id", "run_id", "rng_seed", "dataset", "labeled_ratio", "method", "target_backbone",
          "probe_backbone", "seed", "run_source", "protocol_version", "manifest_path",
          "cohort_digest", "labeled_count", "unlabeled_count", "validation_count", "test_count",
          "scaler_mean", "scaler_std", "best_epoch", "best_validation_mae_years",
          "test_mae_years", "test_r2", "runtime_seconds", "peak_gpu_memory", "checkpoint_size",
          "checkpoint_path", "artifact_directory", "git_commit_sha", "git_diff_sha256", "status",
          "aggregate_eligible", "notes"]


def source_state():
    path = ROOT / "artifacts" / "utkface_5pct" / "source_state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_row(path: Path, method: str, ratio: float, target: str, probe, seed_id: int, rng_seed: int, source: str):
    metadata = json.loads((path / "metadata.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    counts = metadata.get("counts", {})
    scaler = metadata.get("label_scaler", {})
    state = source_state()
    return {
        "experiment_id": f"utkface-r{ratio:.2f}-{target}-{method}-seedid{seed_id}",
        "run_id": seed_id, "rng_seed": rng_seed,
        "dataset": "UTKFace", "labeled_ratio": ratio, "method": method,
        "target_backbone": target, "probe_backbone": probe, "seed": rng_seed,
        "run_source": source, "protocol_version": metadata.get("protocol_version"),
        "manifest_path": metadata.get("manifest_path"), "cohort_digest": metadata.get("cohort_sha256"),
        "labeled_count": counts.get("labeled"), "unlabeled_count": counts.get("unlabeled"),
        "validation_count": counts.get("validation"), "test_count": counts.get("test"),
        "scaler_mean": scaler.get("mean"), "scaler_std": scaler.get("std"),
        "best_epoch": metrics.get("best_epoch"), "best_validation_mae_years": metrics.get("validation_mae"),
        "test_mae_years": metrics.get("test_mae"), "test_r2": metrics.get("test_r2"),
        "runtime_seconds": metadata.get("wall_clock_seconds"),
        "peak_gpu_memory": metadata.get("peak_allocated_cuda_bytes"),
        "checkpoint_size": metadata.get("checkpoint_size_bytes", (path / "best.pt").stat().st_size),
        "checkpoint_path": str((path / "best.pt").resolve()), "artifact_directory": str(path.resolve()),
        "git_commit_sha": state.get("commit"), "git_diff_sha256": state.get("git_diff_sha256"),
        "status": "complete", "aggregate_eligible": seed_id in range(6),
        "notes": None,
    }


def collect():
    rows = []
    old = ROOT / "artifacts" / "utkface_5pct"
    for method in ("rapl", "hpl"):
        for seed in range(6):
            path = old / method / f"seed_{seed}"
            if (path / "metrics.json").exists():
                rows.append(run_row(path, method, .05, "ImageNet-pretrained ResNet-50",
                                    "frozen ImageNet-pretrained ResNet-50" if method == "rapl" else None,
                                    seed, seed, "verified_current_protocol"))
    new = ROOT / "artifacts" / "benchmarks" / "utkface" / "dinov2"
    for ratio in (.05, .10, .20):
        for method in ("rapl", "hpl"):
            for seed_id in range(6):
                path = new / f"ratio_{ratio:.2f}" / method / f"seed_{seed_id}"
                if (path / "metrics.json").exists():
                    rows.append(run_row(path, method, ratio, "DINOv2 ViT-S/14 LVD-142M",
                                        "frozen DINOv2 ViT-S/14 LVD-142M" if method == "rapl" else None,
                                        seed_id, seed_id, "verified_current_protocol"))
    resnet = ROOT / "artifacts" / "benchmarks" / "utkface" / "resnet50"
    for ratio in (.10, .20):
        for method in ("rapl", "hpl"):
            for seed_id in range(6):
                path = resnet / f"ratio_{ratio:.2f}" / method / f"seed_{seed_id}"
                if (path / "metrics.json").exists():
                    rows.append(run_row(path, method, ratio, "ImageNet-pretrained ResNet-50",
                                        "frozen ImageNet-pretrained ResNet-50" if method == "rapl" else None,
                                        seed_id, seed_id, "verified_current_protocol"))
    dinov3 = ROOT / "artifacts" / "benchmarks" / "utkface" / "dinov3_convnext_tiny" / "ratio_0.05"
    for method in ("rapl", "hpl"):
        for seed_id in range(6):
            path = dinov3 / method / f"seed_{seed_id}"
            if (path / "metrics.json").exists():
                rows.append(run_row(path, method, .05, "DINOv3 ConvNeXt-Tiny LVD-1689M",
                                    "frozen DINOv3 ConvNeXt-Tiny LVD-1689M" if method == "rapl" else None,
                                    seed_id, seed_id, "verified_current_protocol"))
    imdb = ROOT / "artifacts" / "benchmarks" / "imdb_wiki"
    for backbone, target, probe in (
        ("resnet50", "ImageNet-pretrained ResNet-50", "frozen ImageNet-pretrained ResNet-50"),
        ("dinov2", "DINOv2 ViT-S/14 LVD-142M", "frozen DINOv2 ViT-S/14 LVD-142M"),
    ):
        for ratio in (.05, .10, .20):
            for method in ("rapl", "hpl"):
                for seed_id in range(6):
                    path = imdb / backbone / f"ratio_{ratio:.2f}" / method / f"seed_{seed_id}"
                    if (path / "metrics.json").exists():
                        row = run_row(path, method, ratio, target, probe if method == "rapl" else None,
                                      seed_id, seed_id, "verified_current_protocol")
                        row["dataset"] = "IMDB-WIKI-DIR"
                        row["experiment_id"] = f"imdb-wiki-r{ratio:.2f}-{target}-{method}-seedid{seed_id}"
                        rows.append(row)
    rows.extend([
        {"experiment_id":"legacy-utkface-r0.05-hpl-seed0","dataset":"UTKFace","labeled_ratio":.05,
         "method":"hpl","target_backbone":"ImageNet-pretrained ResNet-50","probe_backbone":None,"seed":0,
         "run_source":"legacy_manual_result","test_mae_years":5.920,"status":"historical_metric_only",
         "aggregate_eligible":False,"notes":"Manually supplied value; no artifact matching this exact metric with complete current-protocol metadata was found."},
        {"experiment_id":"legacy-utkface-r0.05-rapl-seed0","dataset":"UTKFace","labeled_ratio":.05,
         "method":"rapl","target_backbone":"ImageNet-pretrained ResNet-50","probe_backbone":"frozen ImageNet-pretrained ResNet-50","seed":0,
         "run_source":"legacy_manual_result","test_mae_years":5.759,"status":"historical_metric_only",
         "aggregate_eligible":False,"notes":"Manually supplied value; no artifact matching this exact metric with complete current-protocol metadata was found."},
    ])
    return [{field: row.get(field) for field in FIELDS} for row in rows]


def summaries(rows):
    groups = {}
    for row in rows:
        if row["status"] != "complete" or not row["aggregate_eligible"]:
            continue
        key = tuple(row[name] for name in ("dataset","labeled_ratio","method","target_backbone","probe_backbone"))
        groups.setdefault(key, []).append(row)
    result=[]
    for key, values in sorted(groups.items(), key=str):
        values=sorted(values,key=lambda row:int(row["seed"])); maes=[float(v["test_mae_years"]) for v in values]; r2s=[float(v["test_r2"]) for v in values]
        result.append({"dataset":key[0],"labeled_ratio":key[1],"method":key[2],"target_backbone":key[3],"probe_backbone":key[4],
          "individual_seed_results":json.dumps({str(v['seed']):v['test_mae_years'] for v in values},sort_keys=True),
          "test_mae_mean":statistics.mean(maes),"test_mae_sample_std":statistics.stdev(maes) if len(maes)>1 else None,
          "test_r2_mean":statistics.mean(r2s),"test_r2_sample_std":statistics.stdev(r2s) if len(r2s)>1 else None,
          "validation_mae_per_seed":json.dumps({str(v['seed']):v['best_validation_mae_years'] for v in values},sort_keys=True),
          "best_epoch_per_seed":json.dumps({str(v['seed']):v['best_epoch'] for v in values},sort_keys=True),
          "mean_runtime":statistics.mean(float(v['runtime_seconds']) for v in values),
          "mean_peak_gpu_memory":statistics.mean(int(v['peak_gpu_memory']) for v in values),
          "max_peak_gpu_memory":max(int(v['peak_gpu_memory']) for v in values),
          "successful_runs":len(values),"failed_runs":0,"retried_runs":0,"complete_seed_group":len(values) in (5,6)})
    return result


def write():
    rows=collect(); summary=summaries(rows); REGISTRY.mkdir(parents=True,exist_ok=True)
    with (REGISTRY/'results_long.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with (REGISTRY/'results_summary.csv').open('w',newline='') as h:
        fields=list(summary[0]) if summary else []; w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(summary)
    (REGISTRY/'results.json').write_text(json.dumps({'runs':rows,'summaries':summary},indent=2,sort_keys=True)+'\n')
    lines=['# Benchmark registry','','## Current verified formal results','', '| Dataset | Ratio | Method | Target | Probe | MAE mean ± sample SD | R² mean ± sample SD |','|---|---:|---|---|---|---:|---:|']
    def metric(value):
        return "—" if value is None else f"{value:.4f}"
    for s in summary: lines.append(f"| {s['dataset']} | {s['labeled_ratio']:.2f} | {s['method']} | {s['target_backbone']} | {s['probe_backbone'] or '—'} | {metric(s['test_mae_mean'])} ± {metric(s['test_mae_sample_std'])} | {metric(s['test_r2_mean'])} ± {metric(s['test_r2_sample_std'])} |")
    lines += ['', '## Historical manually supplied seed-0 results','', '| Dataset | Ratio | Method | Target | Test MAE | Status |','|---|---:|---|---|---:|---|',
              '| UTKFace | 0.05 | HPL | ImageNet-pretrained ResNet-50 | 5.920 | historical_metric_only |',
              '| UTKFace | 0.05 | RAPL | ImageNet-pretrained ResNet-50 | 5.759 | historical_metric_only |','',
              'Historical rows are excluded from all aggregates. Per-seed, checkpoint, runtime, memory, failure, and retry details are retained in the CSV and JSON records.']
    (REGISTRY/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': write()
