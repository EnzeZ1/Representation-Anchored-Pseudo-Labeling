#!/usr/bin/env python3
"""Persistent, resumable 36-run formal UTKFace Temporal Ensembling queue."""
from __future__ import annotations
import argparse, csv, json, math, os, shutil, signal, subprocess, sys, tempfile, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; PYTHON=Path('/nobackup/enzez/.venv/bin/python')
QUEUE=ROOT/'artifacts/benchmark_queues/utkface_temporal_ensembling'; RESULTS=ROOT/'artifacts/benchmarks/utkface/temporal_ensembling'
STATE=QUEUE/'queue_state.json'; LOG=QUEUE/'launcher.log'; DIGEST='61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56'
DATASET_ARG='utkface'; DATASET_LABEL='UTKFace'; DATA_ROOT=ROOT/'data/utkface_all'; MANIFEST_STEM='utkface'; TEST_SIZE=2370
MANIFEST_ROOT=ROOT/'data_processing/splits'; PROTOCOL_VERSION='utkface-benchmark-v1'; BATCH_SIZE=32; NUM_WORKERS=4
MANIFEST_TEMPLATE='{stem}_ratio_0{ratio}_seed_{seed}.json'
UPSTREAM='a32743acbdea54560b7d9d90934c1eb5b49a81a7'; REQUIRED=('best.pt','last.pt','config.json','metadata.json','metrics.json','history.csv','run.log','test_predictions.npz','analysis_snapshot.npz')
def write(path,obj):
 """Atomically replace a JSON file, including a flush to stable storage."""
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp_name=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent)
 tmp=Path(tmp_name)
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
  dir_fd=os.open(path.parent,os.O_RDONLY)
  try: os.fsync(dir_fd)
  finally: os.close(dir_fd)
 finally:
  if tmp.exists(): tmp.unlink()
def valid(path):
 try:
  if not all((path/n).is_file() for n in REQUIRED): return False
  m=json.loads((path/'metadata.json').read_text()); x=json.loads((path/'metrics.json').read_text()); z=np_load(path/'test_predictions.npz')
  import numpy as np
  manifest=json.loads(Path(m['manifest_path']).read_text())
  if DATASET_ARG=='chexchonet_lvidd': expected=np.asarray(manifest['indices']['test'])
  elif DATASET_ARG=='rsna_bone_age': expected=np.asarray(json.loads((DATA_ROOT.parent/'protocol/cohort.json').read_text())['splits']['test'])
  else: expected=np.asarray(manifest['splits']['test'])
  finite_predictions=all(not np.issubdtype(v.dtype,np.number) or np.isfinite(v).all() for v in z.values())
  return m['status']=='complete' and m['dataset']==DATASET_LABEL and m['method']=='temporal_ensembling' and m['protocol_version']==PROTOCOL_VERSION and m['cohort_sha256']==DIGEST and m['official_source_commit']==UPSTREAM and m['stable_sample_identity_verified'] and m['temporal_state_finite'] and m['canonical_prediction_order_verified'] and m['checkpoint_reloaded'] and not m['test_used_for_selection'] and m['test_model_inference_count']==1 and all(math.isfinite(float(x[k])) for k in ('validation_mae','test_mae','test_r2')) and len(z['cohort_indices'])==TEST_SIZE and finite_predictions and np.array_equal(z['cohort_indices'],expected)
 except Exception: return False
def np_load(path):
 import numpy as np
 with np.load(path) as z: return {k:z[k] for k in z.files}
def jobs():
 rows=[]
 # Interleave backbones so both mandated first formal runs launch immediately.
 for ratio in ('.05','.10','.20'):
  for seed in range(6):
   for backbone in ('resnet50','dinov2_vits14'):
    canonical=RESULTS/backbone/f'ratio_0{ratio}'/f'seed_{seed}'
    attempts=[int(p.name[8:]) for p in canonical.glob('attempt_*') if p.name[8:].isdigit()]
    rows.append({'id':f'{backbone}:ratio_0{ratio}:seed_{seed}','backbone':backbone,'labeled_ratio':float(ratio),'seed':seed,'manifest':str(MANIFEST_ROOT/MANIFEST_TEMPLATE.format(stem=MANIFEST_STEM,ratio=ratio,seed=seed)),'output_dir':str(canonical),'attempts':max(attempts,default=0),'retries':0,'status':'complete' if valid(canonical) else 'pending'})
 assert len(rows)==36 and len({x['id'] for x in rows})==36
 return rows
def persist(rows,running,stopping=False):
 payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'total_formal_jobs':36,'stopping':stopping,'jobs':rows,'running':{k:{'pid':v['process'].pid,'physical_gpu':v['gpu'],'attempt':v['attempt']} for k,v in running.items()}}; write(STATE,payload); write(QUEUE/'run_status.json',payload)
def log(s):
 line=f'[{time.strftime("%F %T")}] {s}'; print(line,flush=True)
 with LOG.open('a') as f:f.write(line+'\n')
def log_exception(context):
 log(f'ERROR {context}\n{traceback.format_exc().rstrip()}')
def promote(src,dst):
 if valid(dst): return
 if not valid(src): raise ValueError(f'attempt fails formal integrity: {src}')
 dst.mkdir(parents=True,exist_ok=True)
 # Each artifact is staged in the canonical directory and atomically replaced.
 # A crash can leave harmless dotfiles, but canonical validity cannot become true
 # until the entire required set is present and valid. Re-running is idempotent.
 for name in REQUIRED:
  fd,tmp_name=tempfile.mkstemp(prefix=f'.{name}.',suffix='.promoting',dir=dst); os.close(fd); tmp=Path(tmp_name)
  try:
   shutil.copy2(src/name,tmp)
   with tmp.open('rb') as f: os.fsync(f.fileno())
   os.replace(tmp,dst/name)
  finally:
   if tmp.exists(): tmp.unlink()
 if not valid(dst): raise RuntimeError(f'canonical result invalid after promotion: {dst}')
 write(dst/'successful_attempt.json',{'source_attempt':src.name,'promoted_unix':time.time()})
def infrastructure_failure(logpath):
 text=logpath.read_text(errors='replace').lower()[-20000:]
 tokens=('cuda out of memory','cudnn','nccl','resource temporarily unavailable','connection reset','timed out','input/output error','no space left','worker exited unexpectedly')
 return any(t in text for t in tokens)
def aggregate(rows):
 complete=[]
 for j in rows:
  p=Path(j['output_dir'])
  if valid(p):
   x=json.loads((p/'metrics.json').read_text()); complete.append({**j,'test_mae':x['test_mae'],'test_r2':x['test_r2'],'validation_mae':x['validation_mae'],'best_epoch':x['best_epoch']})
 if complete:
  with (RESULTS/'results_long.csv').open('w',newline='') as f:
   fields=('backbone','labeled_ratio','seed','validation_mae','best_epoch','test_mae','test_r2','output_dir'); w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(({k:r[k] for k in fields} for r in complete))
 lines=[f'# {DATASET_LABEL} Temporal Ensembling formal results','','| Backbone | Ratio | Seeds | Test MAE | Test R² |','|---|---:|---|---:|---:|']
 for b in ('resnet50','dinov2_vits14'):
  for ratio in (.05,.10,.20):
   g=[r for r in complete if r['backbone']==b and math.isclose(r['labeled_ratio'],ratio)]
   if len(g)==6 and {r['seed'] for r in g}==set(range(6)):
    summaries={}
    for key in ('test_mae','test_r2'):
     vals=[float(r[key]) for r in g]; mean=sum(vals)/6; sd=math.sqrt(sum((x-mean)**2 for x in vals)/5); summaries[key]=(mean,sd)
    mae,r2=summaries['test_mae'],summaries['test_r2']
    lines.append(f'| {b} | {ratio:.2f} | 0–5 | {mae[0]:.4f} ± {mae[1]:.4f} | {r2[0]:.4f} ± {r2[1]:.4f} |')
 (RESULTS/'results_summary.md').write_text('\n'.join(lines)+'\n')
def finish_worker(jid,item,rows,running,stopping):
 """Reap and account for one worker without allowing errors to kill the queue."""
 proc=item['process']; rc=proc.wait(); job=item['job']; attempt=item['attempt_dir']
 try:
  item['handle'].close()
  if rc==0 and valid(attempt):
   try:
    promote(attempt,Path(job['output_dir'])); job['status']='complete'; job.pop('last_failure',None); log(f'completed {jid} GPU={item["gpu"]}')
   except Exception:
    job['status']='promotion_failed'; job['last_failure']='promotion exception; valid attempt retained'; log_exception(f'promotion failed for {jid}')
  elif job['retries']<1 and infrastructure_failure(attempt/'run.log'):
   job['retries']+=1; job['status']='pending'; job['last_failure']=f'infrastructure exit {rc}; one retry scheduled'; job['resume_from']=str(attempt/'last.pt') if (attempt/'last.pt').is_file() else None; log(f'infrastructure retry scheduled {jid}')
  else: job['status']='failed'; job['last_failure']=f'deterministic/non-retriable exit {rc}'; log(f'failed without further retry {jid}')
 except Exception:
  job['status']='controller_error'; job['last_failure']=f'worker completion exception; exit {rc}'; log_exception(f'completion handling failed for {jid}')
 finally:
  running.pop(jid,None)
  try: aggregate(rows)
  except Exception: log_exception(f'aggregation failed after {jid}')
  try: persist(rows,running,stopping)
  except Exception: log_exception(f'state persistence failed after {jid}')
def recover_promotions(rows):
 """Reconcile stale state and valid, unpromoted attempts from canonical truth."""
 for job in rows:
  dst=Path(job['output_dir'])
  if valid(dst): job['status']='complete'; continue
  candidates=sorted((p for p in dst.glob('attempt_*') if p.is_dir()),reverse=True)
  source=next((p for p in candidates if valid(p)),None)
  if source is None: continue
  try:
   promote(source,dst); job['status']='complete'; job.pop('last_failure',None); log(f'recovered {job["id"]} from {source.name}')
  except Exception:
   job['status']='promotion_failed'; job['last_failure']='recovery promotion exception; valid attempt retained'; log_exception(f'recovery promotion failed for {job["id"]}')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--gpus',nargs='+',type=int,required=True); p.add_argument('--poll-seconds',type=int,default=15); a=p.parse_args(); assert len(a.gpus)==len(set(a.gpus))
 QUEUE.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True); rows=jobs(); write(QUEUE/'queue_config.json',{'dataset':DATASET_LABEL,'method':'Temporal Ensembling','official_source_repo':'https://github.com/s-laine/tempens','official_source_commit':UPSTREAM,'total_formal_jobs':36,'backbones':['resnet50','dinov2_vits14'],'ratios':[.05,.10,.20],'seeds':list(range(6)),'gpus':a.gpus,'separate_smoke_stage':False,'retry_policy':'ordinary infrastructure failures at most once; deterministic failures never retried'}); write(QUEUE/'runner.json',{'pid':os.getpid(),'started_unix':time.time()}); (QUEUE/'runner.pid').write_text(str(os.getpid())+'\n')
 recover_promotions(rows)
 running={}; stopping=False
 def stop(*_):
  nonlocal stopping; stopping=True; log('stop requested; no new jobs will launch')
 signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop); persist(rows,running)
 while True:
  for jid,item in list(running.items()):
   rc=item['process'].poll()
   if rc is None: continue
   finish_worker(jid,item,rows,running,stopping)
  pending=[j for j in rows if j['status']=='pending']
  if not running and (stopping or not pending): break
  if not stopping:
   free=[g for g in a.gpus if g not in {v['gpu'] for v in running.values()}]
   for gpu,job in zip(free,pending):
    try:
     job['attempts']+=1; attempt=Path(job['output_dir'])/f'attempt_{job["attempts"]}'; attempt.mkdir(parents=True,exist_ok=False)
     cmd=[str(PYTHON),'-m','training.temporal_ensembling_benchmark','--dataset',DATASET_ARG,'--backbone',job['backbone'],'--manifest',job['manifest'],'--data-root',str(DATA_ROOT),'--output-dir',str(attempt),'--seed',str(job['seed']),'--epochs','30','--batch-size',str(BATCH_SIZE),'--num-workers',str(NUM_WORKERS)]
     if job.get('resume_from'): cmd+=['--resume-from',job['resume_from']]
     handle=(attempt/'run.log').open('a'); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'TEMPENS_PHYSICAL_GPU':str(gpu),'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1'}); proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True); job.update(status='running',physical_gpu=gpu); running[job['id']]={'process':proc,'handle':handle,'gpu':gpu,'job':job,'attempt':job['attempts'],'attempt_dir':attempt}; log(f'started {job["id"]} pid={proc.pid} physical GPU={gpu} attempt={job["attempts"]}'); persist(rows,running,stopping)
    except Exception:
     job['status']='controller_error'; job['last_failure']='launch exception'; log_exception(f'launch failed for {job["id"]}')
  time.sleep(a.poll_seconds)
 aggregate(rows); persist(rows,running,stopping); log('queue terminal'); raise SystemExit(1 if any(j['status']=='failed' for j in rows) else 0)
if __name__=='__main__':
 try: main()
 except Exception:
  QUEUE.mkdir(parents=True,exist_ok=True); log_exception('fatal queue-controller exception'); raise
