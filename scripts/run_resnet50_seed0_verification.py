#!/usr/bin/env python3
"""Wait for the active sweep, then reproduce the two ResNet-50 5% seed-0 runs."""
from __future__ import annotations

import json, math, os, shlex, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from baselines.benchmark_io import REQUIRED_RUN_FILES, run_complete, write_json

ACTIVE=ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_ratios'
QUEUE=ROOT/'artifacts'/'benchmark_queues'/'utkface_resnet50_seed0_verification'
OLD=ROOT/'artifacts'/'utkface_5pct'; DEPS=OLD/'python_deps'
STATE=QUEUE/'queue_state.json'; STATUS=QUEUE/'run_status.json'; LOG=QUEUE/'launcher.log'
DIGEST='61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56'
MANIFEST=ROOT/'data_processing'/'splits'/'utkface_ratio_0.05_seed_0.json'
stopping=False

def log(message):
 line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}';print(line,flush=True)
 with LOG.open('a') as h:h.write(line+'\n')
def persist(stage,jobs,running=None,archive=None):
 payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'stage':stage,'cohort_sha256':DIGEST,'jobs':jobs,'running':running or {},'archive_path':str(archive) if archive else None}
 write_json(STATE,payload);write_json(STATUS,payload)
def active_terminal():
 try:x=json.loads((ACTIVE/'run_status.json').read_text())
 except (OSError,ValueError):return False
 return len(x.get('jobs',[]))==24 and all(j.get('status') in ('complete','failed') for j in x['jobs']) and not x.get('running')
def gpu_state():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [{'index':int(a.strip()),'uuid':b.strip(),'free':int(c.strip()),'util':int(d.strip())} for a,b,c,d in (line.split(',') for line in text.splitlines())]
def idle_gpus():return [g for g in gpu_state() if g['free']>=9000 and g['util']<=10]
def validate(path,method):
 if not run_complete(path):return False,'required files or completion metadata invalid'
 try:
  md=json.loads((path/'metadata.json').read_text());mt=json.loads((path/'metrics.json').read_text());cfg=json.loads((path/'config.json').read_text())
  ok=(md['cohort_sha256']==DIGEST and md['manifest_seed']==0 and md['counts']=={'cohort':23709,'train':18969,'labeled':948,'unlabeled':18021,'validation':2370,'test':2370} and md['label_scaler']['mean']==32.84599304199219 and md['label_scaler']['std']==19.87841796875 and md['checkpoint_reloaded'] is True and md['test_evaluations']==1 and md['test_used_for_selection'] is False and cfg['seed']==0 and all(math.isfinite(float(mt[k])) for k in ('validation_mae','test_mae','test_r2')))
  return (ok,'ok' if ok else 'protocol metadata mismatch')
 except (OSError,ValueError,KeyError) as exc:return False,str(exc)
def command(method,path):
 common=['--benchmark_manifest',str(MANIFEST),'--benchmark_output_dir',str(path),'--data_dir',str(ROOT/'data'/'utkface_all'),'--seed','0']
 if method=='rapl':return ROOT,[sys.executable,str(ROOT/'train.py'),'-dataset','utkface','--method','probe','--labeled_ratio','0.05','--backbone','resnet50','--probe_backbone','resnet50','--save',str(path/'best.pt'),*common]
 cwd=ROOT/'third_party'/'Heteroscedastic-Pseudo-Labels'/'utkface';return cwd,[sys.executable,'main_ours.py','--benchmark_backbone','resnet50',*common]
def main():
 global stopping
 QUEUE.mkdir(parents=True,exist_ok=True);(QUEUE/'runner.pid').write_text(str(os.getpid())+'\n')
 jobs=[{'id':f'verify:{m}:ratio_0.05:seed_0','method':m,'labeled_ratio':.05,'rng_seed':0,'target_backbone':'ImageNet-pretrained ResNet-50','probe_backbone':'separately instantiated frozen ImageNet-pretrained ResNet-50' if m=='rapl' else None,'manifest_path':str(MANIFEST),'artifact_directory':str(OLD/m/'seed_0'),'status':'waiting','attempts':0} for m in ('rapl','hpl')]
 def stop(*_):
  global stopping;stopping=True;log('Stop requested while waiting; no verification job launched.')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);persist('waiting_for_24_job_queue',jobs)
 log('Waiting for the existing 24-job ResNet-50 queue to reach terminal status.')
 while not stopping and not active_terminal():time.sleep(30)
 if stopping:return 2
 log('Existing 24-job queue is terminal; preparing immutable archive of previous seed-0 results.')
 timestamp=datetime.now().strftime('%Y%m%d_%H%M%S');archive=OLD/'archive'/f'seed_0_previous_{timestamp}';archive.mkdir(parents=True,exist_ok=False)
 entries={}
 for method in ('rapl','hpl'):
  source=OLD/method/'seed_0';ok,reason=validate(source,method)
  if not source.is_dir():raise FileNotFoundError(source)
  metrics=json.loads((source/'metrics.json').read_text()) if (source/'metrics.json').exists() else None
  target=archive/method;os.replace(source,target)
  entries[method]={'status':'superseded_unverified_previous_run','previous_path':str(source),'archive_path':str(target),'prearchive_validation':{'passed':ok,'detail':reason},'metrics':metrics,'files':sorted(p.name for p in target.iterdir())}
 write_json(archive/'archive_manifest.json',{'created_unix':time.time(),'cohort_sha256':DIGEST,'reason':'Independent current-protocol seed-0 verification rerun','entries':entries})
 for job in jobs:job['status']='pending'
 persist('verification',jobs,archive=archive)
 while not stopping and len(idle_gpus())<2:time.sleep(30)
 if stopping:return 2
 selected=idle_gpus()[:2];running={}
 for gpu,job in zip(selected,jobs):
  method=job['method'];out=Path(job['artifact_directory']);out.mkdir(parents=True,exist_ok=False);cwd,cmd=command(method,out);handle=(out/'run.log').open('a');env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(gpu['index']),'UTKFACE_BENCHMARK_ROOT':str(ROOT),'PYTHONPATH':os.pathsep.join([str(DEPS),str(ROOT),env.get('PYTHONPATH','')]),'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(QUEUE/'matplotlib'),'NUMBA_CACHE_DIR':str(QUEUE/'numba_cache')});Path(env['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True);Path(env['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True)
  started=time.time();proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);job.update({'status':'running','attempts':1,'physical_gpu_index':gpu['index'],'gpu_uuid':gpu['uuid'],'cuda_visible_devices':str(gpu['index']),'process_local_device':'cuda:0','pid':proc.pid,'started_unix':started,'command':cmd});running[job['id']]={'process':proc,'handle':handle,'started':started};log(f"Started {job['id']} pid={proc.pid} GPU={gpu['index']}: {shlex.join(cmd)}")
 persist('verification',jobs,{k:{'pid':v['process'].pid,'gpu':next(j['physical_gpu_index'] for j in jobs if j['id']==k)} for k,v in running.items()},archive)
 for jid,record in list(running.items()):
  rc=record['process'].wait();record['handle'].close();job=next(j for j in jobs if j['id']==jid);job['exit_code']=rc;job['runtime_seconds']=time.time()-record['started'];ok,reason=validate(Path(job['artifact_directory']),job['method']);job['integrity_detail']=reason;job['status']='complete' if rc==0 and ok else 'failed';log(f"Finished {jid}: exit={rc}, integrity={reason}");persist('verification',jobs,archive=archive)
 if all(j['status']=='complete' for j in jobs):
  subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=True)
  subprocess.run([sys.executable,str(ROOT/'scripts/report_resnet50_utkface.py')],cwd=ROOT,check=True)
  old={m:float(entries[m]['metrics']['test_mae']) for m in ('rapl','hpl')};new={j['method']:float(json.loads((Path(j['artifact_directory'])/'metrics.json').read_text())['test_mae']) for j in jobs}
  write_json(QUEUE/'verification_result.json',{'status':'complete','archive_path':str(archive),'previous_test_mae':old,'new_test_mae':new,'absolute_difference':{m:abs(new[m]-old[m]) for m in old},'jobs':jobs})
  log('Both verification runs passed and canonical registries were updated.')
 else:log('At least one verification failed; previous result remains archived and is not used as fallback.')
 persist('complete',jobs,archive=archive);return 0 if all(j['status']=='complete' for j in jobs) else 1
if __name__=='__main__':raise SystemExit(main())
