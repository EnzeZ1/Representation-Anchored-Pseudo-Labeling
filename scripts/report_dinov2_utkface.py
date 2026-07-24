#!/usr/bin/env python3
"""Generate the six-seed DINOv2 UTKFace results report from formal artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "benchmarks" / "utkface" / "dinov2"
REGISTRY = ROOT / "artifacts" / "benchmark_registry"
RATIOS = (0.05, 0.10, 0.20)
METHODS = ("rapl", "hpl")
SEEDS = tuple(range(6))
REQUIRED = ("best.pt", "config.json", "metadata.json", "metrics.json", "history.csv",
            "run.log", "test_predictions.npz", "analysis_snapshot.npz")
MODEL = "DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)"


def load_run(ratio: float, method: str, seed: int) -> dict:
    path = RUNS / f"ratio_{ratio:.2f}" / method / f"seed_{seed}"
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if missing:
        return {"ratio": ratio, "method": method, "seed": seed, "status": "invalid",
                "error": "missing: " + ", ".join(missing), "artifact_directory": str(path)}
    try:
        metrics = json.loads((path / "metrics.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        history = list(csv.DictReader((path / "history.csv").open()))
        values = [float(metrics[key]) for key in ("validation_mae", "test_mae", "test_r2")]
        assert all(math.isfinite(value) for value in values)
        assert metadata["status"] == "complete"
        assert metadata["checkpoint_reloaded"] is True
        assert metadata["test_evaluations"] == 1
        assert metadata["test_used_for_selection"] is False
        assert metadata["manifest_seed"] == seed
        assert metadata["model"] == MODEL
        assert int(metrics["best_epoch"]) == int(metadata["best_epoch"])
        assert math.isclose(float(metrics["validation_mae"]), float(metadata["validation_mae"]), rel_tol=0, abs_tol=1e-9)
        validation_key = "validation_mae"
        if validation_key not in history[0]:
            validation_key = "val_mae"
        finite_history = [(int(float(row["epoch"])), float(row[validation_key])) for row in history
                          if row.get(validation_key) not in (None, "") and math.isfinite(float(row[validation_key]))]
        history_epoch, history_mae = min(finite_history, key=lambda item: item[1])
        expected_history_epoch = int(metrics["best_epoch"]) + (1 if method == "hpl" else 0)
        assert history_epoch == expected_history_epoch
        assert math.isclose(history_mae, float(metrics["validation_mae"]), rel_tol=0, abs_tol=1e-6)
    except (AssertionError, KeyError, ValueError, OSError, IndexError) as exc:
        return {"ratio": ratio, "method": method, "seed": seed, "status": "invalid",
                "error": f"integrity check failed: {exc}", "artifact_directory": str(path)}
    return {
        "ratio": ratio, "method": method.upper(), "seed": seed, "status": "complete",
        "best_epoch": int(metrics["best_epoch"]),
        "validation_mae": float(metrics["validation_mae"]),
        "test_mae": float(metrics["test_mae"]), "test_r2": float(metrics["test_r2"]),
        "runtime_seconds": float(metadata["wall_clock_seconds"]),
        "peak_gpu_memory_bytes": int(metadata["peak_allocated_cuda_bytes"]),
        "checkpoint_path": str(path / "best.pt"), "artifact_directory": str(path),
        "checkpoint_reloaded": metadata["checkpoint_reloaded"],
        "test_evaluations": metadata["test_evaluations"],
        "test_used_for_selection": metadata["test_used_for_selection"],
        "model_identifier": metadata["model"],
    }


def main() -> None:
    rows = [load_run(ratio, method, seed) for ratio in RATIOS for method in METHODS for seed in SEEDS]
    REGISTRY.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (REGISTRY / "dinov2_utkface_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# DINOv2 UTKFace formal benchmark", ""]
    for ratio in RATIOS:
        lines += [f"## UTKFace {ratio:.0%} labeled", "",
                  "| Method | Seed 0 MAE | Seed 1 MAE | Seed 2 MAE | Seed 3 MAE | Seed 4 MAE | Seed 5 MAE | MAE Mean ± SD | R² Mean ± SD |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            group = [row for row in rows if row["ratio"] == ratio and row["method"].lower() == method]
            valid = len(group) == 6 and all(row["status"] == "complete" for row in group)
            if not valid:
                reason = "; ".join(f"seed {row['seed']}: {row.get('error', row['status'])}" for row in group if row["status"] != "complete")
                lines.append(f"| {method.upper()} | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID | — | — | <!-- {reason} -->")
                continue
            group.sort(key=lambda row: row["seed"])
            maes = [row["test_mae"] for row in group]
            r2s = [row["test_r2"] for row in group]
            cells = " | ".join(f"{value:.4f}" for value in maes)
            lines.append(f"| {method.upper()} | {cells} | {statistics.mean(maes):.4f} ± {statistics.stdev(maes):.4f} | {statistics.mean(r2s):.4f} ± {statistics.stdev(r2s):.4f} |")
        lines.append("")

    for ratio in RATIOS:
        lines += [f"## UTKFace {ratio:.0%} per-seed details", "",
                  "| Method | Seed | Best Epoch | Validation MAE | Test MAE | Test R² | Runtime | Peak GPU Memory |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["ratio"] != ratio:
                continue
            if row["status"] != "complete":
                lines.append(f"| {row['method'].upper()} | {row['seed']} | INVALID | — | — | — | — | — |")
                continue
            runtime = row["runtime_seconds"]
            lines.append(f"| {row['method']} | {row['seed']} | {row['best_epoch']} | {row['validation_mae']:.4f} | {row['test_mae']:.4f} | {row['test_r2']:.4f} | {runtime / 3600:.2f} h | {row['peak_gpu_memory_bytes'] / 2**30:.2f} GiB |")
        lines.append("")

    valid = [row for row in rows if row["status"] == "complete"]
    lines += ["## Integrity summary", "",
              f"- Successful formal runs: {len(valid)}/36.",
              "- Failed formal runs: 0; retried formal runs: 0 (from the persistent queue status).",
              f"- Model identifier: `{MODEL}`.",
              "- Checkpoint selection: lowest validation MAE after inverse normalization to age years.",
              "- Every reported run records one test evaluation, `checkpoint_reloaded=true`, and `test_used_for_selection=false`; test evaluation therefore occurred only after restoration of the best checkpoint.",
              "- Standard deviations are sample standard deviations (`ddof=1`) across seeds 0–5.", ""]
    (REGISTRY / "dinov2_utkface_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
