#!/usr/bin/env python3
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REG=ROOT/'artifacts/benchmark_registry';BASE=ROOT/'artifacts/benchmarks/utkface/dinov3_convnext_tiny/ratio_0.05'
def main():
 rows=[]
 for method in ('rapl','hpl'):
  for seed in range(6):
   p=BASE/method/f'seed_{seed}';m=json.loads((p/'metrics.json').read_text());d=json.loads((p/'metadata.json').read_text());rows.append({'method':method.upper(),'seed':seed,'test_mae':m['test_mae'],'test_r2':m['test_r2'],'validation_mae':m['validation_mae'],'best_epoch':m['best_epoch'],'checkpoint_path':str(p/'best.pt'),'runtime_seconds':d['wall_clock_seconds'],'peak_gpu_memory_bytes':d['peak_allocated_cuda_bytes']})
 fields=list(rows[0]);REG.mkdir(parents=True,exist_ok=True)
 with (REG/'dinov3_convnext_utkface_results.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines=['# DINOv3 ConvNeXt-Tiny UTKFace 5% results','','| Method | Seed 0 MAE | Seed 1 MAE | Seed 2 MAE | Seed 3 MAE | Seed 4 MAE | Seed 5 MAE | MAE Mean ± SD | R² Mean ± SD |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in ('RAPL','HPL'):
  g=[r for r in rows if r['method']==method];mae=[r['test_mae'] for r in g];r2=[r['test_r2'] for r in g];lines.append(f"| {method} | {' | '.join(f'{v:.4f}' for v in mae)} | {statistics.mean(mae):.4f} ± {statistics.stdev(mae):.4f} | {statistics.mean(r2):.4f} ± {statistics.stdev(r2):.4f} |")
 lines += ['','## 5% backbone comparison','', '| Backbone | Method | Seeds | MAE mean ± sample SD | R² mean ± sample SD |','|---|---|---:|---:|---:|']
 sources=[('ResNet-50',ROOT/'artifacts/utkface_5pct',5),('DINOv2 ViT-S/14',ROOT/'artifacts/benchmarks/utkface/dinov2/ratio_0.05',6),('DINOv3 ConvNeXt-Tiny',BASE,6)]
 for backbone,base,n in sources:
  for method in ('rapl','hpl'):
   vals=[]
   for seed in range(n):
    m=json.loads((base/method/f'seed_{seed}'/'metrics.json').read_text());vals.append(m)
   lines.append(f"| {backbone} | {method.upper()} | {n} | {statistics.mean(v['test_mae'] for v in vals):.4f} ± {statistics.stdev(v['test_mae'] for v in vals):.4f} | {statistics.mean(v['test_r2'] for v in vals):.4f} ± {statistics.stdev(v['test_r2'] for v in vals):.4f} |")
 (REG/'dinov3_convnext_utkface_results.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
