#!/usr/bin/env python3
"""Persistent parent process for the five missing ResNet-50 runs."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "artifacts/benchmark_queues/utkface_resnet50_missing_runs"
QUEUE.mkdir(parents=True, exist_ok=True)
(QUEUE / "pipeline.pid").write_text(str(os.getpid()) + "\n")
raise SystemExit(subprocess.run(
    [sys.executable, str(ROOT / "scripts/run_utkface_resnet50_missing.py"), "--resume"],
    cwd=ROOT,
).returncode)
