#!/usr/bin/env python3
"""Persistent smoke-gated, phase-barrier DINOv2 UTKFace benchmark queue."""

from __future__ import annotations

import argparse, json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from baselines.benchmark_io import run_complete, write_json

ART=ROOT/'artifacts'; QUEUE_ROOT=ART/'benchmark_queues'/'utkface_dinov2_ratios'
BENCH=ART/'benchmarks'/'utkface'/'dinov2'; DEPS=ROOT/'artifacts'/'utkface_5pct'/'python_deps'
STATE=QUEUE_ROOT/'queue_state.json'; STATUS=QUEUE_ROOT/'run_status.json'; LOG=QUEUE_ROOT/'launcher.log'
RUNNER_PID=QUEUE_ROOT/'runner.pid'; PIPELINE_PID=QUEUE_ROOT/'pipeline.pid'; CONFIG=QUEUE_ROOT/'queue_config.json'
METHODS=('rapl','hpl'); RATIOS=(.05,.10,.20); SEEDS=(0,1,2,3,4,5)
DIGEST='61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56'

def arguments():
 p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true'); p.add_argument('--dry_run',action='store_true'); p.add_argument('--max_parallel',type=int,default=6); p.add_argument('--available_gpus',nargs='+',type=int,default=list(range(6))); p.add_argument('--poll_seconds',type=int,default=20); return p.parse_args()
def manifest(ratio,seed): return ROOT/'data_processing'/'splits'/f"utkface_ratio_{ratio:.2f}_seed_{seed}.json"
def output(ratio,method,seed,smoke=False):
 base=BENCH/'smoke' if smoke else BENCH; return base/f'ratio_{ratio:.2f}'/method/f'seed_{seed}'
def command(ratio,method,seed,smoke=False):
 out=output(ratio,method,seed,smoke); common=['--benchmark_manifest',str(manifest(ratio,seed)),'--benchmark_output_dir',str(out),'--data_dir',str(ROOT/'data'/'utkface_all'),'--seed',str(seed)]
 if method=='rapl':
  cmd=[sys.executable,str(ROOT/'train.py'),'-dataset','utkface','--method','probe','--labeled_ratio',str(ratio),'--backbone','dinov2','--probe_backbone','dinov2','--dino','s','--save',str(out/'best.pt'),*common]
  if smoke: cmd += ['--epochs','1']
  return ROOT,cmd
 cwd=ROOT/'third_party'/'Heteroscedastic-Pseudo-Labels'/'utkface'; cmd=[sys.executable,'main_ours.py','--benchmark_backbone','dinov2',*common]
 if smoke: cmd += ['--benchmark_epochs','1']
 return cwd,cmd
def gpu_state():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True); return {int(a):{'free':int(b),'util':int(c)} for a,b,c in ([v.strip() for v in line.split(',')] for line in text.splitlines())}
def idle(allowed,occupied):
 state=gpu_state(); return [g for g in allowed if g not in occupied and g in state and state[g]['free']>=9000 and state[g]['util']<=10]
def log(message):
 line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"; print(line,flush=True)
 with LOG.open('a') as h: h.write(line+'\n')
def persist(jobs,running,stage):
 payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'stage':stage,'cohort_sha256':DIGEST,'jobs':jobs,'running':{k:{'pid':v['process'].pid,'gpu':v['gpu']} for k,v in running.items()}}; write_json(STATE,payload); write_json(STATUS,payload)
def valid_smoke(path):
 if not run_complete(path): return False
 md=json.loads((path/'metadata.json').read_text()); mt=json.loads((path/'metrics.json').read_text())
 return (md.get('cohort_sha256')==DIGEST and md.get('manifest_seed')==0 and md.get('checkpoint_reloaded') is True and md.get('test_evaluations')==1 and md.get('test_used_for_selection') is False and 'DINOv2 ViT-S/14' in md.get('model','') and all(__import__('math').isfinite(float(mt[k])) for k in ('validation_mae','test_mae','test_r2')))
def run_jobs(jobs,args,stage,phase_barriers=False):
 running={}; stopping=False
 def stop(*_):
  nonlocal stopping; stopping=True; log('Stop requested; no new jobs will launch.')
 signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); persist(jobs,running,stage)
 while True:
  for jid,rec in list(running.items()):
   rc=rec['process'].poll()
   if rc is None: continue
   rec['handle'].close(); job=rec['job']
   if rc==0 and run_complete(job['output_dir']):
    job['status']='complete'; log(f"Completed {jid} GPU={rec['gpu']}")
    subprocess.run([sys.executable,str(ROOT/'scripts'/'update_benchmark_registry.py')],cwd=ROOT,check=False)
   elif job['attempts']<2: job['status']='pending'; job['last_failure']=f'exit {rc}; retry scheduled'; log(f"Failed {jid}; scheduling one retry")
   else: job['status']='failed'; job['last_failure']=f'exit {rc}; retries exhausted'; log(f"Failed {jid}; retries exhausted")
   del running[jid]; persist(jobs,running,stage)
  unfinished=[j for j in jobs if j['status'] in ('pending','running')]
  if not running and not unfinished: break
  active_phase=min((j['phase'] for j in unfinished),default=None) if phase_barriers else None
  pending=[j for j in jobs if j['status']=='pending' and (active_phase is None or j['phase']==active_phase)]
  if not stopping and pending and len(running)<args.max_parallel:
   for gpu,job in zip(idle(args.available_gpus,{v['gpu'] for v in running.values()}),pending[:args.max_parallel-len(running)]):
    out=Path(job['output_dir']); out.mkdir(parents=True,exist_ok=True); handle=(out/'run.log').open('a'); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'UTKFACE_BENCHMARK_ROOT':str(ROOT),'PYTHONPATH':os.pathsep.join([str(DEPS),str(ROOT),env.get('PYTHONPATH','')]),'MPLCONFIGDIR':str(QUEUE_ROOT/'matplotlib'),'NUMBA_CACHE_DIR':str(QUEUE_ROOT/'numba_cache')}); Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True); Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
    job['status']='running'; job['attempts']+=1; job['gpu']=gpu; proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True); running[job['id']]={'process':proc,'gpu':gpu,'job':job,'handle':handle}; log(f"Started {job['id']} pid={proc.pid} GPU={gpu}: {shlex.join(job['command'])}"); persist(jobs,running,stage)
  time.sleep(args.poll_seconds)
 persist(jobs,running,stage); return not any(j['status']=='failed' for j in jobs)
def main():
 args=arguments(); QUEUE_ROOT.mkdir(parents=True,exist_ok=True); RUNNER_PID.write_text(str(os.getpid())+'\n'); write_json(CONFIG,{'dataset':'UTKFace','target_backbone':'DINOv2 ViT-S/14 LVD-142M','probe_backbone':'frozen DINOv2 ViT-S/14 LVD-142M for RAPL; null for HPL','methods':METHODS,'ratios':RATIOS,'rng_seeds':SEEDS,'phase_barriers':True,'max_parallel':args.max_parallel})
 smoke=[]
 for method in METHODS:
  cwd,cmd=command(.05,method,0,True); out=output(.05,method,0,True); smoke.append({'id':f'smoke:{method}','run_id':0,'rng_seed':0,'method':method,'labeled_ratio':.05,'target_backbone':'DINOv2 ViT-S/14 LVD-142M','probe_backbone':'frozen DINOv2 ViT-S/14 LVD-142M' if method=='rapl' else None,'phase':-1,'cwd':str(cwd),'command':cmd,'output_dir':str(out),'attempts':0,'status':'complete' if args.resume and valid_smoke(out) else 'pending'})
 formal=[]
 for phase,ratio in enumerate(RATIOS):
  for method in METHODS:
   for seed in SEEDS:
    cwd,cmd=command(ratio,method,seed); out=output(ratio,method,seed); formal.append({'id':f'phase{phase}:{method}:ratio_{ratio:.2f}:seed_{seed}','run_id':seed,'rng_seed':seed,'method':method,'labeled_ratio':ratio,'target_backbone':'DINOv2 ViT-S/14 LVD-142M','probe_backbone':'frozen DINOv2 ViT-S/14 LVD-142M' if method=='rapl' else None,'phase':phase,'cwd':str(cwd),'command':cmd,'output_dir':str(out),'attempts':0,'status':'complete' if args.resume and run_complete(out) else 'pending'})
 assert len(formal)==36
 assert len({job['id'] for job in formal})==36
 for ratio in RATIOS:
  for method in METHODS:
   group=[job for job in formal if job['labeled_ratio']==ratio and job['method']==method]
   assert len(group)==6 and {job['rng_seed'] for job in group}==set(SEEDS)
 if args.dry_run:
  for j in smoke+formal:
   print(f"cd {shlex.quote(j['cwd'])} && CUDA_VISIBLE_DEVICES=<gpu> {shlex.join(j['command'])}")
  return
 if not run_jobs(smoke,args,'smoke',False) or not all(valid_smoke(Path(j['output_dir'])) for j in smoke): log('Smoke gate failed; formal queue not started'); raise SystemExit(1)
 ok=run_jobs(formal,args,'formal',True); log('DINOv2 formal queue finished'); raise SystemExit(0 if ok else 1)
if __name__=='__main__': main()
