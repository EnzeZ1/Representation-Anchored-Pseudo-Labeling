#!/usr/bin/env python3
"""Persistent parent process for the DINOv2 smoke gate and formal sweep."""

import os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'artifacts'/'benchmark_queues'/'utkface_dinov2_ratios'
QUEUE.mkdir(parents=True,exist_ok=True)
(QUEUE/'pipeline.pid').write_text(str(os.getpid())+'\n')
raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts'/'run_utkface_dinov2_sweep.py'),'--resume'],cwd=ROOT).returncode)
