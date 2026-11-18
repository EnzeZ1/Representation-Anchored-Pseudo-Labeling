#!/usr/bin/env python3
"""Two physical-GPU slots, one supervised attempt per cell, strict promotion."""
from pathlib import Path
import argparse,csv,datetime,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
import numpy as np
B=Path('/nobackup/enze/simregmatch_reorient_20260907'); R=B/'source'
O=R/'artifacts/simregmatch_reorient_5pct_olvi1_20260907'; PY=B/'venv/bin/python'
GPUS={4:'GPU-68642926-d434-854f-5962-598b3f4ea6d7',5:'GPU-5c5f8055-5dac-c88e-f6a1-dca05444a5dd'}
THREADS={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
DATA={'utkface':B/'data/utkface_all','imdb_wiki':B/'data/imdb_wiki','rsna':B/'data/rsna_bone_age/2017/extracted','chexchonet_lvidd':B/'data/chexchonet/1.0.0'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def manifest(ds,s,ratio=.05):
 tag=f'{ratio:.2f}'
 if ds=='rsna':return DATA[ds].parent/'protocol'/f'ratio_{tag}_seed_{s}.json'
 if ds=='chexchonet_lvidd':return R/'artifacts/protected_protocol/chexchonet_lvidd'/f'chexchonet_lvidd_ratio_{tag}_seed_{s}.json'
 return R/'data_processing/splits'/f'{ds}_ratio_{tag}_seed_{s}.json'
def jobs():
 return [{'id':f'{ds}:{ratio:.2f}:{bb}:seed_{s}','dataset':ds,'backbone':bb,'seed':s,'ratio':ratio,'epochs':30,'batch_size':32 if ds=='chexchonet_lvidd' else 16,'manifest':str(manifest(ds,s,ratio)),'manifest_file_sha256':sha(manifest(ds,s,ratio)),'status':'pending','attempts':0} for ratio in (.05,.10) for ds in DATA for bb in ('resnet50','dinov2_vits14') for s in range(6)]
def command(j,a):
 return [str(PY),'main.py','--benchmark_manifest',j['manifest'],'--benchmark_output_dir',str(a),'--data_dir',str(DATA[j['dataset']]),'--labeled-ratio',str(j['ratio']),'--model',j['backbone'],'--seed',str(j['seed']),'--benchmark_epochs',str(j['epochs']),'--batch_size',str(j['batch_size']),'--num_workers','0','--reorient']
def env(gpu):
 assert gpu in GPUS
 return {'HOME':'/afs/cs.wisc.edu/u/e/n/enzez','USER':'enzez','LOGNAME':'enzez','PATH':str(PY.parent)+':/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':GPUS[gpu],'UTKFACE_BENCHMARK_ROOT':str(R),'PYTHONPATH':str(R),'PYTHONUNBUFFERED':'1','TORCH_HOME':str(B/'torch_cache'),'HF_HOME':str(B/'hf_cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',**THREADS}
def validate(a,j):
 checks={}
 try:
  required=('best.pt','config.json','metadata.json','metrics.json','history.csv','test_predictions.npz','run.log','launch.json','early_health.json')
  checks['required_files']=all((a/f).is_file() and (a/f).stat().st_size>0 for f in required)
  if not checks['required_files']:return {'valid':False,'checks':checks,'missing':[f for f in required if not (a/f).is_file()]}
  md=read(a/'metadata.json');c=read(a/'config.json');m=read(a/'metrics.json')
  checks['identity']=md.get('method')=='simregmatch' and md.get('plugin')=='reorient' and md.get('status')=='complete' and md.get('target_backbone')==j['backbone'] and md.get('manifest_seed')==j['seed']
  checks['reorient']=all(md.get(k) is True for k in ('supervised_recycling','preserve_full_unlabeled_trajectory','combined_same_optimizer_update')) and md.get('additional_supervised_optimizer_step') is False
  checks['test_protocol']=md.get('checkpoint_reloaded') is True and md.get('test_used_for_selection') is False and md.get('test_model_inference_count')==1
  checks['manifest']=sha(j['manifest'])==j['manifest_file_sha256'] and md.get('manifest_path')==j['manifest'] and c.get('benchmark_manifest')==j['manifest']
  checks['formal_epoch_budget']=j.get('epochs')==30 and c.get('epochs')==30 and md.get('completed_epochs')==30
  checks['config']=c.get('reorient') is True and c.get('seed')==j['seed'] and c.get('model')==j['backbone'] and c.get('epochs')==j['epochs'] and c.get('batch_size')==j['batch_size'] and c.get('labeled_ratio')==j['ratio'] and c.get('num_workers')==0 and not c.get('reliability_audit_dir')
  checks['canonical_hyperparameters']=all(c.get(k)==v for k,v in {'percentile':.95,'iter_u':5,'dropout':.1,'beta':.6,'t':1,'lambda_u':.01,'optimizer':'adam','lr':.001,'loss':'mse','weight_decay':.0001}.items()) and md.get('percentile_argument_to_numpy')==.95
  checks['finite_metrics']=all(math.isfinite(float(m[k])) for k in ('validation_mae','test_mae','test_r2'))
  history=list(csv.DictReader((a/'history.csv').open()))
  checks['full_epochs']=len(history)==j['epochs'] and [int(x['epoch']) for x in history]==list(range(1,j['epochs']+1)) and md.get('completed_epochs')==j['epochs']
  checks['finite_validation_history']=all(math.isfinite(float(x[k])) for x in history for k in ('validation_loss','validation_mae','validation_r2'))
  expected=O/f"expected_{j['dataset']}_seed_{j['seed']}.npz"
  with np.load(a/'test_predictions.npz',allow_pickle=False) as z,np.load(expected,allow_pickle=False) as e:
   pred=z['predictions'];target=z['targets'];idx=z['cohort_indices']
   checks['test_cohort']=np.array_equal(idx,e['cohort_indices']) and np.array_equal(z['relative_paths'],e['relative_paths']) and np.allclose(target,e['targets'],rtol=0,atol=1e-5)
   checks['finite_predictions']=pred.shape==target.shape==idx.shape and len(idx)>0 and bool(np.isfinite(pred).all()) and bool(np.isfinite(target).all())
   mae=float(np.mean(np.abs(pred.astype(float)-target.astype(float))));r2=1-float(np.sum((pred.astype(float)-target)**2)/np.sum((target.astype(float)-np.mean(target.astype(float)))**2))
   checks['recomputed_metrics']=math.isclose(mae,float(m['test_mae']),rel_tol=1e-5,abs_tol=1e-5) and math.isclose(r2,float(m['test_r2']),rel_tol=1e-5,abs_tol=1e-5)
  checks['checkpoint_size']=md.get('checkpoint_size_bytes')==(a/'best.pt').stat().st_size
  import torch
  checkpoint=torch.load(a/'best.pt',map_location='cpu',weights_only=False)
  checks['readable_checkpoint']=bool(checkpoint['model']) and all(bool(t.isfinite().all()) for t in checkpoint['model'].values() if t.is_floating_point())
  best=min(history,key=lambda x:float(x['validation_mae']))
  checks['validation_selection']=checkpoint['epoch']==int(best['epoch'])==int(m['best_epoch'])==int(md['best_epoch']) and math.isclose(float(checkpoint['validation_mae_years']),float(best['validation_mae'])) and math.isclose(float(m['validation_mae']),float(best['validation_mae']))
  checks['checkpoint_config']=checkpoint['config']['reorient'] is True and checkpoint['config']['seed']==j['seed'] and checkpoint['config']['model']==j['backbone']

  return {'valid':all(checks.values()),'checks':checks,'checked_at':now()}
 except Exception as e:return {'valid':False,'checks':checks,'error':repr(e),'checked_at':now()}
def initialize():
 assert read(O/'data_validation.json')['valid']
 audits={h:read(O/f'audit_{h}.json') for h in ('gpu05','gpu04','gpu03','olvi1')}
 assert all(not a['candidates'] and not a['candidate_directories'] and not a['errors'] for a in audits.values()),'Audit candidates/errors require review'
 assert not (O/'queue.json').exists(),'Refusing queue overwrite'
 js=jobs();write(O/'queue.json',{'created':now(),'jobs':js})
 write(O/'audit_report.json',{'complete_valid':0,'partial_incomplete':0,'missing':48,'scope':audits,'runs':[{**j,'classification':'missing'} for j in js]})
 write(O/'commands.json',[{'id':j['id'],'command':command(j,O/'attempts'/j['dataset']/j['backbone']/f"seed_{j['seed']}"/'attempt_1')} for j in js])
 write(O/'protocol.json',{'method':'canonical SimRegMatch + Reorient','allowed_physical_gpus':GPUS,'num_workers':0,'threads':THREADS,'runs':48,'seeds':list(range(6)),'backbones':['resnet50','dinov2_vits14'],'datasets':list(DATA),'canonical_percentile':.95,'epoch_budgets':{'utkface':30,'imdb_wiki':30,'rsna':30,'chexchonet_lvidd':30},'change_scope':'labeled iterator recycling over the full unlabeled loader; existing drop_last preserved; one combined optimizer step','max_attempts_per_run':1,'seed_semantics':'canonical seed-specific manifests and role-seeded loaders; no new model/RNG initialization changes','no_host_ram_kill_guard':True})
 source={str(p.relative_to(R)):sha(p) for p in R.rglob('*.py') if 'artifacts' not in p.parts and '__pycache__' not in p.parts}
 write(O/'provenance/launch_source_sha256.json',source)
 print(json.dumps({'registered':len(js),'valid':0,'missing':48}))
def gpu_available(g):
 info=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(info[0])==g and info[1].strip()==GPUS[g] and 'A100' in info[2]
 processes=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 return not processes and int(info[3])<1024

def worker(job_id,gpu,a):
 os.umask(0o077);assert gpu in GPUS
 j=next(j for j in read(O/'queue.json')['jobs'] if j['id']==job_id)
 a=Path(a);a.mkdir(parents=True,exist_ok=False)
 cmd=command(j,a);environment=env(gpu);signal_received=[];child=None
 def stop(signum,frame):
  signal_received.append(signum)
  if child is not None and child.poll() is None:os.kill(child.pid,signum)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
 launch={'id':job_id,'wrapper_pid':os.getpid(),'physical_gpu':gpu,'gpu_uuid':GPUS[gpu],'command':cmd,'cwd':str(R/'third_party/SimRegMatch'),'environment':environment,'attempt_path':str(a),'start_timestamp':now()}
 try:
  with (a/'run.log').open('w') as log:
   child=subprocess.Popen(cmd,cwd=R/'third_party/SimRegMatch',env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   launch['trainer_pid']=child.pid;write(a/'launch.json',launch)
   rc=child.wait()
  result=validate(a,j);write(a/'validator.json',result)
  write(a/'exit.json',{'exit_code':rc if rc>=0 else None,'returncode':rc,'terminating_signal':-rc if rc<0 else None,'wrapper_received_signals':signal_received,'end_timestamp':now(),'trainer_pid':child.pid,'physical_gpu':gpu,'attempt_path':str(a),'valid':rc==0 and result['valid']})
  return 0 if rc==0 and result['valid'] else 1
 except BaseException as e:
  write(a/'wrapper_failure.json',{'error':repr(e),'end_timestamp':now(),'launch':launch});return 1

def controller():
 os.umask(0o077)
 lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=read(O/'queue.json');assert not any(j['status']=='running' for j in state['jobs']),'Review interrupted jobs before restart'
 assert read(O/'data_validation.json')['valid']
 for p,digest in read(O/'provenance/launch_source_sha256.json').items():assert sha(R/p)==digest,p
 write(O/'controller.json',{'pid':os.getpid(),'ppid':os.getppid(),'sid':os.getsid(0),'started':now(),'detached_session_leader':os.getsid(0)==os.getpid(),'allowed_physical_gpus':list(GPUS)})
 (O/'controller.pid').write_text(str(os.getpid())+'\n')
 running={};stopping=[]
 def stop(sig,frame):
  stopping.append(sig)
  for proc,j,a in running.values():
   if proc.poll() is None:os.kill(proc.pid,sig)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
 while True:
  for g,(proc,j,a) in list(running.items()):
   if (a/'launch.json').exists():j.update({k:v for k,v in read(a/'launch.json').items() if k in ('trainer_pid','start_timestamp')})
   rc=proc.poll()
   if rc is None:continue
   result=read(a/'exit.json') if (a/'exit.json').exists() else {'valid':False,'reason':'supervisor exited without exit record'}
   validation=validate(a,j) if result.get('valid') else {'valid':False,'reason':'unsuccessful supervised attempt','exit_record':result}
   write(a/'promotion_validator.json',validation)
   j.update(status='complete' if rc==0 and validation['valid'] else 'failed',finished=now(),wrapper_returncode=rc,validation=validation)
   if j['status']=='complete':
    destination=O/'completed'/j['dataset']/j['backbone']/f"seed_{j['seed']}";destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():j.update(status='failed',error='refused output collision')
    else:destination.symlink_to(a,target_is_directory=True);j['validated_final_artifact']=str(destination)
   if j['status']=='failed':write(O/'failures'/f"{j['dataset']}_{j['backbone']}_seed_{j['seed']}.json",j)
   del running[g]
  for g in GPUS:
   pending=[j for j in state['jobs'] if j['status']=='pending']
   if stopping or g in running or not pending:continue
   if not gpu_available(g):continue
   j=pending[0];assert j['attempts']==0 and sha(j['manifest'])==j['manifest_file_sha256']
   a=O/'attempts'/j['dataset']/j['backbone']/f"seed_{j['seed']}"/'attempt_1'
   j.update(status='running',attempts=1,physical_gpu=g,gpu_uuid=GPUS[g],attempt_path=str(a),claimed=now())
   write(O/'queue.json',state)
   proc=subprocess.Popen([str(PY),str(Path(__file__).resolve()),'--worker',j['id'],'--gpu',str(g),'--attempt',str(a)],stdin=subprocess.DEVNULL,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'',**THREADS})
   j['wrapper_pid']=proc.pid;running[g]=(proc,j,a)
  write(O/'queue.json',state)
  if not running and (stopping or all(j['status']!='pending' for j in state['jobs'])):
   write(O/'controller_finished.json',{'ended':now(),'complete':sum(j['status']=='complete' for j in state['jobs']),'failed':sum(j['status']=='failed' for j in state['jobs']),'signals':stopping});return
  time.sleep(15)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--worker');p.add_argument('--gpu',type=int,choices=(4,5));p.add_argument('--attempt');a=p.parse_args()
 if a.initialize:initialize()
 elif a.controller:controller()
 elif a.worker:sys.exit(worker(a.worker,a.gpu,a.attempt))
 else:p.error('choose mode')
