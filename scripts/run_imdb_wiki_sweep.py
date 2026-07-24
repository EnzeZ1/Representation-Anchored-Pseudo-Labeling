#!/usr/bin/env python3
"""Persistent two-phase IMDB-WIKI-DIR ResNet-50 then DINOv2 queue."""

from __future__ import annotations
import argparse, json, math, os, shlex, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from baselines.benchmark_io import REQUIRED_RUN_FILES, write_json

Q=ROOT/'artifacts/benchmark_queues/imdb_wiki_resnet50_dinov2'
OUT=ROOT/'artifacts/benchmarks/imdb_wiki'; DEPS=ROOT/'artifacts/utkface_5pct/python_deps'
STATE=Q/'queue_state.json';STATUS=Q/'run_status.json';CONFIG=Q/'queue_config.json';LOG=Q/'launcher.log'
DIGEST='919fe3e1b959e1fe75e08e83310a84c1c3a9d53a16812a1bb5f1e0117ba97f43'
RATIOS=(.05,.10,.20);METHODS=('rapl','hpl');SEEDS=tuple(range(6));BACKBONES=('resnet50','dinov2')

def arguments():
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry_run',action='store_true');p.add_argument('--max_parallel',type=int,default=6);p.add_argument('--available_gpus',nargs='+',type=int,default=list(range(6)));p.add_argument('--poll_seconds',type=int,default=30);return p.parse_args()
def manifest(r,s):return ROOT/f'data_processing/splits/imdb_wiki_ratio_{r:.2f}_seed_{s}.json'
def canonical(b,r,m,s):return OUT/b/f'ratio_{r:.2f}'/m/f'seed_{s}'
def complete(p):
 p=Path(p)
 if not all((p/n).is_file() for n in REQUIRED_RUN_FILES):return False
 try:
  x=json.loads((p/'metrics.json').read_text());d=json.loads((p/'metadata.json').read_text())
  return d['status']=='complete' and d['checkpoint_reloaded'] is True and d['test_used_for_selection'] is False and d['test_evaluations']==1 and all(math.isfinite(float(x[k])) for k in ('validation_mae','test_mae','test_r2'))
 except (OSError,ValueError,KeyError,TypeError):return False
def command(b,r,m,s,out):
 common=['--benchmark_manifest',str(manifest(r,s)),'--benchmark_output_dir',str(out),'--data_dir',str(ROOT/'data/imdb_wiki'),'--seed',str(s)]
 if m=='rapl':
  cmd=[sys.executable,str(ROOT/'train.py'),'-dataset','imdb_wiki','--method','probe','--labeled_ratio',str(r),'--backbone',b,'--probe_backbone',b,'--save',str(out/'best.pt'),*common]
  if b=='dinov2':cmd += ['--dino','s']
  return ROOT,cmd
 return ROOT/'third_party/Heteroscedastic-Pseudo-Labels/utkface',[sys.executable,'main_ours.py','--benchmark_backbone',b,'--labeled_ratio',str(r),*common]
def inventory():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 states=[{'index':int(a.strip()),'uuid':b.strip(),'free_mib':int(c.strip()),'utilization':int(d.strip())} for a,b,c,d in (line.split(',') for line in rows.splitlines())]
 try:procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 except subprocess.CalledProcessError:procs=''
 busy={}
 for line in procs.splitlines():
  if line.strip():
   u,p,n,m=(x.strip() for x in line.split(',',3));busy.setdefault(u,[]).append({'pid':int(p),'name':n,'memory_mib':m})
 for s in states:s['compute_processes']=busy.get(s['uuid'],[])
 return states
def idle(allowed,reserved):return [g for g in inventory() if g['index'] in allowed and g['index'] not in reserved and not g['compute_processes'] and g['free_mib']>=10000 and g['utilization']<=10]
def log(s):
 line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {s}';print(line,flush=True)
 with LOG.open('a') as h:h.write(line+'\n')
def persist(jobs,running):
 payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'job_count':72,'cohort_sha256':DIGEST,'jobs':jobs,'running':{k:{'pid':v['process'].pid,'gpu':v['gpu']['index'],'gpu_uuid':v['gpu']['uuid'],'attempt':v['job']['attempts']} for k,v in running.items()}}
 write_json(STATE,payload);write_json(STATUS,payload)
def main():
 a=arguments();Q.mkdir(parents=True,exist_ok=True);(Q/'runner.pid').write_text(str(os.getpid())+'\n');jobs=[]
 for phase,b in enumerate(BACKBONES):
  for r in RATIOS:
   for m in METHODS:
    for s in SEEDS:
     out=canonical(b,r,m,s);jobs.append({'id':f'phase{phase}:{b}:{m}:ratio_{r:.2f}:seed_{s}','experiment_id':f'imdb-wiki-r{r:.2f}-{b}-{m}-seed-{s}','phase':phase,'dataset':'IMDB-WIKI-DIR','method':m,'labeled_ratio':r,'rng_seed':s,'target_backbone':b,'probe_backbone':b if m=='rapl' else None,'manifest_path':str(manifest(r,s)),'artifact_directory':str(out),'status':'complete' if a.resume and complete(out) else 'pending','attempts':0,'attempt_history':[]})
 assert len(jobs)==72==len({j['experiment_id'] for j in jobs})
 assert sum(j['phase']==0 for j in jobs)==36 and sum(j['phase']==1 for j in jobs)==36
 write_json(CONFIG,{'job_count':72,'phase_A':'36 ResNet-50 jobs','phase_B':'36 DINOv2 jobs','strict_phase_barrier':True,'jobs':jobs})
 if a.dry_run:
  for j in jobs:
   attempt=Q/'attempts'/j['id'].replace(':','_')/'attempt_1';cwd,cmd=command(j['target_backbone'],j['labeled_ratio'],j['method'],j['rng_seed'],attempt);print(f'cd {cwd} && CUDA_VISIBLE_DEVICES=<gpu> {shlex.join(cmd)}')
  return
 running={};stopping=False
 def stop(*_):
  nonlocal stopping;stopping=True;log('Stop requested; no new launches.')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);persist(jobs,running)
 while True:
  for jid,rec in list(running.items()):
   rc=rec['process'].poll()
   if rc is None:continue
   rec['handle'].close();j=rec['job'];valid=rc==0 and complete(rec['attempt'])
   j['attempt_history'][-1].update({'ended_unix':time.time(),'exit_code':rc,'integrity_valid':valid})
   if valid:
    dest=Path(j['artifact_directory'])
    if dest.exists():raise RuntimeError(f'Refusing to overwrite {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True);os.replace(rec['attempt'],dest);j['status']='complete';j['promoted_from']=str(rec['attempt']);log(f"Completed {jid} attempt={j['attempts']} GPU={rec['gpu']['index']}")
    subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=False)
   elif j['attempts']<2:j['status']='pending';j['last_failure']=f'exit {rc}; retry scheduled';log(f'Failed {jid} exit={rc}; retry scheduled')
   else:j['status']='failed';j['last_failure']=f'exit {rc}; retries exhausted';log(f'Failed {jid}; retries exhausted')
   del running[jid];persist(jobs,running)
  unfinished=[j for j in jobs if j['status'] in ('pending','running')]
  if not running and not unfinished:break
  phase=min((j['phase'] for j in unfinished),default=None)
  pending=[j for j in jobs if j['status']=='pending' and j['phase']==phase]
  if not stopping and pending:
   for g,j in zip(idle(a.available_gpus,{v['gpu']['index'] for v in running.values()}),pending[:a.max_parallel-len(running)]):
    n=j['attempts']+1;attempt=Q/'attempts'/j['id'].replace(':','_')/f'attempt_{n}'
    if attempt.exists():raise RuntimeError(f'Attempt collision {attempt}')
    attempt.mkdir(parents=True);cwd,cmd=command(j['target_backbone'],j['labeled_ratio'],j['method'],j['rng_seed'],attempt);h=(attempt/'run.log').open('a')
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(g['index']),'UTKFACE_BENCHMARK_ROOT':str(ROOT),'PYTHONPATH':os.pathsep.join([str(DEPS),str(ROOT),env.get('PYTHONPATH','')]),'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(Q/'matplotlib'),'NUMBA_CACHE_DIR':str(Q/'numba_cache')});Path(env['MPLCONFIGDIR']).mkdir(exist_ok=True);Path(env['NUMBA_CACHE_DIR']).mkdir(exist_ok=True)
    p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);j.update({'status':'running','attempts':n,'pid':p.pid,'physical_gpu_index':g['index'],'gpu_uuid':g['uuid'],'cuda_visible_devices':str(g['index']),'process_local_device':'cuda:0'});j['attempt_history'].append({'attempt':n,'directory':str(attempt),'pid':p.pid,'started_unix':time.time(),'gpu':g,'command':cmd});running[j['id']]={'process':p,'handle':h,'gpu':g,'job':j,'attempt':attempt};log(f"Started {j['id']} attempt={n} pid={p.pid} GPU={g['index']} UUID={g['uuid']} free={g['free_mib']}MiB");persist(jobs,running)
  time.sleep(a.poll_seconds)
 persist(jobs,running);subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=False);subprocess.run([sys.executable,str(ROOT/'scripts/report_imdb_wiki.py')],cwd=ROOT,check=False);log('IMDB-WIKI queue finished');raise SystemExit(1 if any(j['status']=='failed' for j in jobs) else 0)
if __name__=='__main__':main()
