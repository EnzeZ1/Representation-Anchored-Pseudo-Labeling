#!/usr/bin/env python3
"""One-shot detached UTKFace corrected Original HPL campaign; no automatic retries."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/utkface_hpl_corrected_labeled_loop_20260907'
PYTHON = '/nobackup/enzez/.venv/bin/python'
HPL = ROOT / 'third_party/Heteroscedastic-Pseudo-Labels/utkface'
THREADS = {k: '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def jobs():
    result=[]
    for seed in range(6):
        for ratio in (.05,.10,.20):
            for backbone in ('resnet50','dinov2'):
                ident=f'ratio_{ratio:.2f}__{backbone}__seed_{seed}'
                output=OUT/'runs'/ident
                manifest=ROOT/f'data_processing/splits/utkface_ratio_{ratio:.2f}_seed_{seed}.json'
                command=[PYTHON,str(HPL/'main_ours.py'),'--benchmark_manifest',str(manifest),
                    '--benchmark_output_dir',str(output),'--data_dir',str(ROOT/'data/utkface_all'),
                    '--seed',str(seed),'--labeled_ratio',str(ratio),'--benchmark_backbone',backbone,
                    '--training_schedule','original_hpl','--benchmark_method','hpl',
                    '--ssl_mult','2','--anchor_interval','1','--num_epochs','30','--num_workers','0',
                    '--batch_size','32','--w_ulb','10','--lambda2','0.1',
                    '--lr','0.0001','--fc_lr','0.001','--unc_lr','0.0001',
                    '--weight_decay','0.001','--lr_step_period','10','--drp_p','0.05']
                result.append(dict(id=ident,ratio=ratio,backbone=backbone,seed=seed,method='corrected Original HPL',
                    command=command,shell_command=shlex.join(command),cwd=str(HPL),output_dir=str(output),
                    manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    status='pending',attempts=0,gpu=None,pid=None,exit_code=None,signal=None))
    assert len(result)==len({j['id'] for j in result})==36
    return result

def validate(job):
    import numpy as np
    import torch
    p=Path(job['output_dir'])
    for name in ('best.pt','config.json','metadata.json','metrics.json','history.csv','test_predictions.npz'):
        assert (p/name).stat().st_size>0,name
    c=json.loads((p/'config.json').read_text()); m=json.loads((p/'metadata.json').read_text())
    v=json.loads((p/'metrics.json').read_text()); manifest=json.loads(Path(job['manifest']).read_text())
    assert hashlib.sha256(Path(job['manifest']).read_bytes()).hexdigest()==job['manifest_sha256']
    for k,val in dict(training_schedule='original_hpl',benchmark_method='hpl',anchor_interval=1,
        benchmark_backbone=job['backbone'],seed=job['seed'],labeled_ratio=job['ratio'],num_epochs=30,
        w_ulb=10,lambda2=.1,num_workers=0,batch_size=32,lr=.0001,fc_lr=.001,unc_lr=.0001,
        weight_decay=.001,lr_step_period=10,drp_p=.05,ssl_mult=2).items(): assert c[k]==val,(k,c[k])
    assert m['status']=='complete' and m['checkpoint_reloaded'] and not m['label_recycling']
    assert m['schedule_intervention']=='original_hpl' and m['method']=='hpl'
    assert m['final_test_count']==m['test_evaluations']==1 and not m['test_used_for_selection']
    assert m['manifest_seed']==job['seed'] and m['cohort_sha256']==manifest['cohort_sha256']
    assert m['labeled_loader_length']==(manifest['counts']['labeled']*2)//32
    assert not m['labeled_loader_cycled']
    assert m['optimizer_steps_per_epoch']==m['labeled_loader_length']
    assert m['unlabeled_loader_length']==manifest['counts']['unlabeled']//32
    assert m['supervised_anchor_steps_per_epoch']==m['optimizer_steps_per_epoch']
    history=list(csv.DictReader((p/'history.csv').open()))
    assert [int(row['epoch']) for row in history]==list(range(1,31))
    assert all(np.isfinite(float(row['validation_mae'])) for row in history)
    ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==v['best_epoch']==m['best_epoch']
    assert np.isclose(ck['validation_mae_years'],min(float(row['validation_mae']) for row in history))
    assert all(k in ck for k in ('state_dict','uncertainty_learner','optim_feat','optim_fc','optim_unc'))
    with np.load(p/'test_predictions.npz') as predictions:
        pred=predictions['predictions_years'].reshape(-1); target=predictions['targets_years'].reshape(-1)
        assert np.array_equal(predictions['cohort_indices'],manifest['splits']['test'])
        assert len(pred)==len(target)==2370 and np.isfinite(pred).all() and np.isfinite(target).all()
        assert np.isclose(np.abs(pred-target).mean(),v['test_mae'])
        assert np.isclose(1-((pred-target)**2).sum()/((target-target.mean())**2).sum(),v['test_r2'],atol=1e-6)
    assert all(np.isfinite(v[k]) for k in ('validation_mae','test_mae','test_r2'))
    return True

def idle_gpus():
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [(int(i),u.strip()) for i,u,m,v in (r.split(',') for r in rows.splitlines())
            if u.strip() not in {b.strip() for b in busy} and int(m)<100 and int(v)==0]

def main():
    os.environ.update(THREADS)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'queue_state.json').exists(): raise RuntimeError('Campaign already registered; refusing duplicate launch')
    state=dict(controller_pid=os.getpid(),status='active',job_count=36,threads=THREADS,jobs=jobs(),max_attempts=1)
    write(OUT/'queue_config.json',state)
    running={}; stopping=False
    def save(): write(OUT/'queue_state.json',state)
    def stop(*args):
        nonlocal stopping
        stopping=True
        state['status']='draining'
        save()
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    first=True
    while True:
        for gpu,(p,j,log) in list(running.items()):
            rc=p.poll()
            if rc is None: continue
            log.close(); j.update(exit_code=rc,signal=-rc if rc<0 else None,ended_unix=time.time())
            try:
                assert rc==0,f'trainer exit {rc}'
                validate(j); j.update(status='complete',artifacts_validated=True)
            except Exception as exc: j.update(status='failed',artifacts_validated=False,error=repr(exc))
            write(Path(j['output_dir'])/'provenance.json',j)
            del running[gpu];save()
        pending=[j for j in state['jobs'] if j['status']=='pending']
        if pending and not stopping:
            for gpu,uuid in idle_gpus():
                if gpu in running or not pending:continue
                j=pending.pop(0);out=Path(j['output_dir']);out.mkdir(parents=True,exist_ok=False)
                env=os.environ.copy();env.update(THREADS,CUDA_VISIBLE_DEVICES=uuid,
                    UTKFACE_BENCHMARK_ROOT=str(ROOT),PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',
                    MPLCONFIGDIR=str(OUT/'matplotlib'),NUMBA_CACHE_DIR=str(OUT/'numba_cache'))
                log=(out/'run.log').open('x')
                proc=subprocess.Popen(j['command'],cwd=HPL,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                j.update(status='running',attempts=1,gpu=gpu,gpu_uuid=uuid,pid=proc.pid,
                    environment={k:env[k] for k in (*THREADS,'CUDA_VISIBLE_DEVICES','PYTHONPATH')},started_unix=time.time())
                running[gpu]=(proc,j,log);write(out/'provenance.json',j);save()
        if first:
            report=dict(controller_pid=os.getpid(),registered=len(state['jobs']),
                first_wave=[{k:j[k] for k in ('ratio','backbone','seed','gpu','pid')} for j in state['jobs'] if j['status']=='running'],output_root=str(OUT))
            write(OUT/'first_wave.json',report)
            print(json.dumps(report),flush=True)
            if '--ready-fd' in sys.argv: os.close(int(sys.argv[sys.argv.index('--ready-fd')+1]))
            first=False
        if not running and (not pending or stopping):break
        time.sleep(20)
    state['status']='finished' if not stopping else 'stopped';save()

if __name__=='__main__':main()
