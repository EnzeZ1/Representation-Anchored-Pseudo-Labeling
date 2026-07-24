#!/usr/bin/env python3
import csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/benchmarks/imdb_wiki';REG=ROOT/'artifacts/benchmark_registry'
def read(b,r,m,s):
 p=BASE/b/f'ratio_{r:.2f}'/m/f'seed_{s}'
 try:
  x=json.loads((p/'metrics.json').read_text());d=json.loads((p/'metadata.json').read_text());assert d['checkpoint_reloaded'] is True and d['test_used_for_selection'] is False and d['test_evaluations']==1;assert all(math.isfinite(float(x[k])) for k in ('validation_mae','test_mae','test_r2'))
  return {'backbone':b,'ratio':r,'method':m,'seed':s,'test_mae':x['test_mae'],'test_r2':x['test_r2'],'validation_mae':x['validation_mae'],'best_epoch':x['best_epoch'],'status':'complete','artifact_directory':str(p)}
 except Exception as e:return {'backbone':b,'ratio':r,'method':m,'seed':s,'status':'invalid','error':str(e),'artifact_directory':str(p)}
def main():
 rows=[read(b,r,m,s) for b in ('resnet50','dinov2') for r in (.05,.10,.20) for m in ('rapl','hpl') for s in range(6)];REG.mkdir(parents=True,exist_ok=True);fields=sorted({k for x in rows for k in x})
 with (REG/'imdb_wiki_results.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 lines=['# IMDB-WIKI-DIR formal benchmark','']
 for r in (.05,.10,.20):
  lines += [f'## IMDB-WIKI {r:.0%} labeled','',r'| Backbone | Labeled Ratio | Method | Seeds | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ |','|---|---:|---|---:|---:|---:|']
  for b,name in (('resnet50','ResNet-50'),('dinov2','DINOv2 ViT-S/14')):
   groups={m:[x for x in rows if x['backbone']==b and x['ratio']==r and x['method']==m and x['status']=='complete'] for m in ('rapl','hpl')}
   for m in ('rapl','hpl'):
    g=groups[m]
    if len(g)!=6:lines.append(f'| {name} | {r:.2f} | {m.upper()} | incomplete | — | — |');continue
    ma=[float(x['test_mae']) for x in g];rr=[float(x['test_r2']) for x in g];lines.append(f'| {name} | {r:.2f} | {m.upper()} | 0--5 | {statistics.mean(ma):.4f} ± {statistics.stdev(ma):.4f} | {statistics.mean(rr):.4f} ± {statistics.stdev(rr):.4f} |')
  lines.append('')
 lines += ['## RAPL relative MAE reduction','', '| Backbone | Ratio | RAPL MAE | HPL MAE | RAPL Relative MAE Reduction |','|---|---:|---:|---:|---:|']
 for b,name in (('resnet50','ResNet-50'),('dinov2','DINOv2 ViT-S/14')):
  for r in (.05,.10,.20):
   means={}
   for m in ('rapl','hpl'):
    v=[float(x['test_mae']) for x in rows if x['backbone']==b and x['ratio']==r and x['method']==m and x['status']=='complete'];means[m]=statistics.mean(v) if len(v)==6 else None
   lines.append(f"| {name} | {r:.2f} | {means['rapl']:.4f} | {means['hpl']:.4f} | {(means['hpl']-means['rapl'])*100/means['hpl']:.2f}% |" if all(means.values()) else f'| {name} | {r:.2f} | — | — | — |')
 (REG/'imdb_wiki_results.md').write_text('\\n'.join(lines)+'\\n')
if __name__=='__main__':main()
