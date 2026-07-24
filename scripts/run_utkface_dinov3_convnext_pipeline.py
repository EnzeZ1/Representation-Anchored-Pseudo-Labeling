#!/usr/bin/env python3
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'artifacts/benchmark_queues/utkface_dinov3_convnext';Q.mkdir(parents=True,exist_ok=True);(Q/'pipeline.pid').write_text(str(os.getpid())+'\n')
raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts/run_utkface_dinov3_convnext_sweep.py'),'--resume'],cwd=ROOT).returncode)
