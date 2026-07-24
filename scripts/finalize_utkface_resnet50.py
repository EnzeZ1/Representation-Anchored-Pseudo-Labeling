#!/usr/bin/env python3
"""Wait for the persistent queue, then generate registry reports."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; Q=ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_ratios'; LOG=Q/'finalizer.log'
def log(s):
 with LOG.open('a') as h:h.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {s}\n')
log('finalizer started')
while True:
 try: state=json.loads((Q/'queue_state.json').read_text())
 except (OSError,ValueError):time.sleep(30);continue
 statuses=[j['status'] for j in state.get('jobs',[])]
 if len(statuses)==24 and all(s in ('complete','failed') for s in statuses):break
 time.sleep(30)
subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/report_resnet50_utkface.py')],cwd=ROOT,check=True)
log(f"finalized: complete={statuses.count('complete')} failed={statuses.count('failed')}")
