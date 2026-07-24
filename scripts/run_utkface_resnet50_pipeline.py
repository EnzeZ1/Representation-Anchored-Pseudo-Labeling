#!/usr/bin/env python3
"""Persistent parent process for the ResNet-50 UTKFace sweep."""
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; QUEUE=ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_ratios'; QUEUE.mkdir(parents=True,exist_ok=True)
(QUEUE/'pipeline.pid').write_text(str(os.getpid())+'\n')
raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts'/'run_utkface_resnet50_sweep.py'),'--resume'],cwd=ROOT).returncode)
