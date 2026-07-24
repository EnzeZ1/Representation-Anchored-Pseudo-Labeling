#!/usr/bin/env python3
"""Persistent parent for the deferred ResNet-50 seed-0 verification."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];QUEUE=ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_seed0_verification';QUEUE.mkdir(parents=True,exist_ok=True)
(QUEUE/'pipeline.pid').write_text(str(os.getpid())+'\n')
raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts/run_resnet50_seed0_verification.py')],cwd=ROOT).returncode)
