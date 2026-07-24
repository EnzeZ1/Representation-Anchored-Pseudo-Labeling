#!/usr/bin/env python3
"""Persistent phase-barrier ResNet-50 UTKFace benchmark queue."""

from __future__ import annotations

import argparse, json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from baselines.benchmark_io import run_complete, write_json

QUEUE = ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_ratios'
BENCH = ROOT/'artifacts'/'benchmarks'/'utkface'/'resnet50'
DEPS = ROOT/'artifacts'/'utkface_5pct'/'python_deps'
STATE=QUEUE/'queue_state.json'; STATUS=QUEUE/'run_status.json'; CONFIG=QUEUE/'queue_config.json'; LOG=QUEUE/'launcher.log'
RATIOS=(.10,.20); METHODS=('rapl','hpl'); SEEDS=tuple(range(6)); DIGEST='61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56'

def args():
    p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true'); p.add_argument('--dry_run',action='store_true'); p.add_argument('--max_parallel',type=int,default=6); p.add_argument('--available_gpus',nargs='+',type=int,default=list(range(6))); p.add_argument('--poll_seconds',type=int,default=20); return p.parse_args()
def manifest(ratio,seed): return ROOT/'data_processing'/'splits'/f'utkface_ratio_{ratio:.2f}_seed_{seed}.json'
def output(ratio,method,seed): return BENCH/f'ratio_{ratio:.2f}'/method/f'seed_{seed}'
def command(ratio,method,seed):
    out=output(ratio,method,seed); common=['--benchmark_manifest',str(manifest(ratio,seed)),'--benchmark_output_dir',str(out),'--data_dir',str(ROOT/'data'/'utkface_all'),'--seed',str(seed)]
    if method=='rapl':
        return ROOT,[sys.executable,str(ROOT/'train.py'),'-dataset','utkface','--method','probe','--labeled_ratio',str(ratio),'--backbone','resnet50','--probe_backbone','resnet50','--save',str(out/'best.pt'),*common]
    cwd=ROOT/'third_party'/'Heteroscedastic-Pseudo-Labels'/'utkface'
    return cwd,[sys.executable,'main_ours.py','--benchmark_backbone','resnet50',*common]
def gpu_state():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(a):{'free':int(b),'util':int(c)} for a,b,c in ([v.strip() for v in line.split(',')] for line in text.splitlines())}
def idle(allowed,occupied):
    state=gpu_state(); return [g for g in allowed if g not in occupied and g in state and state[g]['free']>=9000 and state[g]['util']<=10]
def log(message):
    line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'; print(line,flush=True)
    with LOG.open('a') as handle: handle.write(line+'\n')
def persist(jobs,running):
    payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'stage':'formal','cohort_sha256':DIGEST,'jobs':jobs,'running':{key:{'pid':value['process'].pid,'gpu':value['gpu']} for key,value in running.items()}}
    write_json(STATE,payload); write_json(STATUS,payload)
def main():
    opt=args(); QUEUE.mkdir(parents=True,exist_ok=True); (QUEUE/'runner.pid').write_text(str(os.getpid())+'\n')
    jobs=[]
    for phase,ratio in enumerate(RATIOS):
        for method in METHODS:
            for seed in SEEDS:
                cwd,cmd=command(ratio,method,seed); out=output(ratio,method,seed)
                jobs.append({'id':f'phase{phase}:{method}:ratio_{ratio:.2f}:seed_{seed}','dataset':'UTKFace','method':method,'labeled_ratio':ratio,'rng_seed':seed,'target_backbone':'ImageNet-pretrained ResNet-50','probe_backbone':'separately instantiated frozen ImageNet-pretrained ResNet-50' if method=='rapl' else None,'manifest_path':str(manifest(ratio,seed)),'artifact_directory':str(out),'phase':phase,'cwd':str(cwd),'command':cmd,'output_dir':str(out),'attempts':0,'status':'complete' if opt.resume and run_complete(out) else 'pending'})
    assert len(jobs)==24 and len({job['id'] for job in jobs})==24
    assert all(job['labeled_ratio'] in RATIOS and 'dinov2' not in ' '.join(job['command']).lower() for job in jobs)
    for ratio in RATIOS:
        for method in METHODS:
            group=[job for job in jobs if job['labeled_ratio']==ratio and job['method']==method]
            assert len(group)==6 and {job['rng_seed'] for job in group}==set(SEEDS)
    write_json(CONFIG,{'dataset':'UTKFace','job_count':24,'methods':METHODS,'ratios':RATIOS,'rng_seeds':SEEDS,'target_backbone':'ImageNet-pretrained ResNet-50','probe_backbone':'separately instantiated frozen ImageNet-pretrained ResNet-50 for RAPL','phase_barriers':True,'jobs':jobs})
    if opt.dry_run:
        for job in jobs: print(f"cd {shlex.quote(job['cwd'])} && CUDA_VISIBLE_DEVICES=<gpu> {shlex.join(job['command'])}")
        return
    running={}; stopping=False
    def stop(*_):
        nonlocal stopping; stopping=True; log('Stop requested; no new jobs will launch.')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); persist(jobs,running)
    while True:
        for jid,rec in list(running.items()):
            rc=rec['process'].poll()
            if rc is None: continue
            rec['handle'].close(); job=rec['job']
            if rc==0 and run_complete(job['output_dir']):
                job['status']='complete'; log(f"Completed {jid} GPU={rec['gpu']}")
                subprocess.run([sys.executable,str(ROOT/'scripts'/'update_benchmark_registry.py')],cwd=ROOT,check=False)
            elif job['attempts']<2:
                job['status']='pending'; job['last_failure']=f'exit {rc}; retry scheduled'; log(f'Failed {jid}; scheduling one retry')
            else:
                job['status']='failed'; job['last_failure']=f'exit {rc}; retries exhausted'; log(f'Failed {jid}; retries exhausted')
            del running[jid]; persist(jobs,running)
        unfinished=[job for job in jobs if job['status'] in ('pending','running')]
        if not running and not unfinished: break
        active_phase=min((job['phase'] for job in unfinished),default=None)
        pending=[job for job in jobs if job['status']=='pending' and job['phase']==active_phase]
        if not stopping and pending and len(running)<opt.max_parallel:
            for gpu,job in zip(idle(opt.available_gpus,{record['gpu'] for record in running.values()}),pending[:opt.max_parallel-len(running)]):
                out=Path(job['output_dir'])
                if run_complete(out): job['status']='complete'; persist(jobs,running); continue
                if out.exists() and any(out.iterdir()):
                    job['status']='failed'; job['last_failure']='nonempty incomplete output directory collision'; persist(jobs,running); continue
                out.mkdir(parents=True,exist_ok=True); handle=(out/'run.log').open('a'); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'UTKFACE_BENCHMARK_ROOT':str(ROOT),'PYTHONPATH':os.pathsep.join([str(DEPS),str(ROOT),env.get('PYTHONPATH','')]),'MPLCONFIGDIR':str(QUEUE/'matplotlib'),'NUMBA_CACHE_DIR':str(QUEUE/'numba_cache'),'PYTHONUNBUFFERED':'1'}); Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True); Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
                job['status']='running'; job['attempts']+=1; job['gpu']=gpu; proc=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True); running[job['id']]={'process':proc,'gpu':gpu,'job':job,'handle':handle}; log(f"Started {job['id']} pid={proc.pid} GPU={gpu}: {shlex.join(job['command'])}"); persist(jobs,running)
        time.sleep(opt.poll_seconds)
    persist(jobs,running); log('ResNet-50 formal queue finished'); raise SystemExit(1 if any(job['status']=='failed' for job in jobs) else 0)
if __name__=='__main__': main()
