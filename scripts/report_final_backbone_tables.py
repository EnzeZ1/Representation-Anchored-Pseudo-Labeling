#!/usr/bin/env python3
"""Regenerate the final UTKFace backbone tables from formal metrics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "artifacts/benchmark_registry"


def runs(backbone, ratio, method):
    if backbone == "ResNet-50" and ratio == .05:
        base = ROOT / "artifacts/utkface_5pct" / method
    else:
        key = {"ResNet-50": "resnet50", "DINOv2 ViT-S/14": "dinov2",
               "DINOv3 ConvNeXt-Tiny": "dinov3_convnext_tiny"}[backbone]
        base = ROOT / f"artifacts/benchmarks/utkface/{key}/ratio_{ratio:.2f}" / method
    found = []
    for seed in range(6):
        path = base / f"seed_{seed}"
        try:
            metrics = json.loads((path / "metrics.json").read_text())
            metadata = json.loads((path / "metadata.json").read_text())
            assert metadata["checkpoint_reloaded"] is True
            assert metadata["test_used_for_selection"] is False
            assert metadata["test_evaluations"] == 1
            assert all(math.isfinite(float(metrics[k])) for k in ("validation_mae", "test_mae", "test_r2"))
            found.append((seed, float(metrics["test_mae"]), float(metrics["test_r2"])))
        except (OSError, ValueError, KeyError, AssertionError):
            continue
    return found


def main():
    sections = [
        (.05, ("ResNet-50", "DINOv2 ViT-S/14", "DINOv3 ConvNeXt-Tiny")),
        (.10, ("ResNet-50", "DINOv2 ViT-S/14")),
        (.20, ("ResNet-50", "DINOv2 ViT-S/14")),
    ]
    lines = []
    included = 0
    missing = []
    for ratio, backbones in sections:
        lines += [
            f"### UTKFace {ratio:.0%} labeled", "",
            "| Backbone | Labeled Ratio | Method | Seeds | Test MAE $\\downarrow$ | Test $R^2$ $\\uparrow$ |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for backbone in backbones:
            groups = {method: runs(backbone, ratio, method) for method in ("rapl", "hpl")}
            mae_means = {method: statistics.mean(value[1] for value in values) if values else math.inf
                         for method, values in groups.items()}
            r2_means = {method: statistics.mean(value[2] for value in values) if values else -math.inf
                        for method, values in groups.items()}
            best_mae = min(mae_means, key=mae_means.get)
            best_r2 = max(r2_means, key=r2_means.get)
            for method in ("rapl", "hpl"):
                values = groups[method]
                included += len(values)
                seeds = [value[0] for value in values]
                absent = sorted(set(range(6)) - set(seeds))
                if absent:
                    missing.append(f"{backbone} {ratio:.2f} {method.upper()}: {absent}")
                seed_text = "0--5" if seeds == list(range(6)) else ",".join(map(str, seeds))
                maes = [value[1] for value in values]
                r2s = [value[2] for value in values]
                mae = f"{statistics.mean(maes):.4f} ± {statistics.stdev(maes):.4f}"
                r2 = f"{statistics.mean(r2s):.4f} ± {statistics.stdev(r2s):.4f}"
                if method == best_mae:
                    mae = f"**{mae}**"
                if method == best_r2:
                    r2 = f"**{r2}**"
                lines.append(f"| {backbone} | {ratio:.2f} | {method.upper()} | {seed_text} | {mae} | {r2} |")
        lines.append("")
    lines += [
        "### UTKFace 5% backbone comparison", "",
        "| Backbone | RAPL MAE | HPL MAE | RAPL Relative MAE Reduction |",
        "|---|---:|---:|---:|",
    ]
    for backbone in ("ResNet-50", "DINOv2 ViT-S/14", "DINOv3 ConvNeXt-Tiny"):
        rapl = statistics.mean(value[1] for value in runs(backbone, .05, "rapl"))
        hpl = statistics.mean(value[1] for value in runs(backbone, .05, "hpl"))
        reduction = 100.0 * (hpl - rapl) / hpl
        lines.append(f"| {backbone} | {rapl:.4f} | {hpl:.4f} | {reduction:.2f}% |")
    lines.append("")
    lines.append(
        f"Integrity: {included} successful formal runs included; "
        f"missing seeds: {'; '.join(missing) if missing else 'none'}."
    )
    REGISTRY.mkdir(parents=True, exist_ok=True)
    (REGISTRY / "utkface_final_backbone_tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
