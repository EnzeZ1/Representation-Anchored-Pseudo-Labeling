#!/usr/bin/env python3
"""Generate ResNet-50 UTKFace tables from saved formal artifacts."""
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'artifacts'/'benchmark_registry'; NEW=ROOT/'artifacts'/'benchmarks'/'utkface'/'resnet50'; OLD=ROOT/'artifacts'/'utkface_5pct'
REQUIRED=('best.pt','config.json','metadata.json','metrics.json','history.csv','run.log','test_predictions.npz','analysis_snapshot.npz')
def read(path,ratio,method,seed):
 missing=[name for name in REQUIRED if not (path/name).is_file()]
 if missing:return {'ratio':ratio,'method':method.upper(),'seed':seed,'status':'invalid','error':'missing: '+','.join(missing),'artifact_directory':str(path)}
 try:
  m=json.loads((path/'metrics.json').read_text()); d=json.loads((path/'metadata.json').read_text())
  assert d['status']=='complete' and d['checkpoint_reloaded'] is True and d['test_evaluations']==1 and d['test_used_for_selection'] is False
  assert d['manifest_seed']==seed and all(math.isfinite(float(m[k])) for k in ('validation_mae','test_mae','test_r2'))
 except (OSError,ValueError,KeyError,AssertionError) as exc:return {'ratio':ratio,'method':method.upper(),'seed':seed,'status':'invalid','error':f'integrity check failed: {exc}','artifact_directory':str(path)}
 return {'ratio':ratio,'method':method.upper(),'seed':seed,'status':'complete','best_epoch':m['best_epoch'],'validation_mae':m['validation_mae'],'test_mae':m['test_mae'],'test_r2':m['test_r2'],'runtime_seconds':d['wall_clock_seconds'],'peak_gpu_memory_bytes':d['peak_allocated_cuda_bytes'],'checkpoint_path':str(path/'best.pt'),'artifact_directory':str(path)}
def main():
 rows=[]
 for method in ('rapl','hpl'):
  for seed in range(6):rows.append(read(OLD/method/f'seed_{seed}',.05,method,seed))
 for ratio in (.10,.20):
  for method in ('rapl','hpl'):
   for seed in range(6):rows.append(read(NEW/f'ratio_{ratio:.2f}'/method/f'seed_{seed}',ratio,method,seed))
 REG.mkdir(parents=True,exist_ok=True); fields=sorted({k for row in rows for k in row})
 with (REG/'resnet50_utkface_results.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines=['# ResNet-50 UTKFace formal benchmark','']
 for ratio,count in ((.05,6),(.10,6),(.20,6)):
  suffix=''
  lines += [f'## UTKFace {ratio:.0%} labeled{suffix}','','| Method | Seed 0 MAE | Seed 1 MAE | Seed 2 MAE | Seed 3 MAE | Seed 4 MAE | Seed 5 MAE | MAE Mean ± SD | R² Mean ± SD |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for method in ('RAPL','HPL'):
   group=sorted([r for r in rows if r['ratio']==ratio and r['method']==method],key=lambda r:r['seed'])
   valid=len(group)==count and all(r['status']=='complete' for r in group)
   if not valid:lines.append(f'| {method} | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID | — | — |');continue
   maes=[float(r['test_mae']) for r in group];r2s=[float(r['test_r2']) for r in group];cells=[f'{v:.4f}' for v in maes]
   lines.append(f"| {method} | {' | '.join(cells)} | {statistics.mean(maes):.4f} ± {statistics.stdev(maes):.4f} | {statistics.mean(r2s):.4f} ± {statistics.stdev(r2s):.4f} |")
  lines.append('')
 lines += ['## Historical registry-only metrics','','The manually supplied RAPL 5.759 and HPL 5.920 values remain `historical_metric_only`; they are not formal seed-5 results and are excluded from every aggregate.','']
 (REG/'resnet50_utkface_results.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
