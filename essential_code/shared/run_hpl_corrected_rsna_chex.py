#!/usr/bin/env python3
"""Detached, work-conserving corrected HPL queue; one fresh attempt per key."""
import csv,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/hpl_corrected_rsna_chex_20260908'
PYTHON='/nobackup/enzez/.venv/bin/python'
THREADS={k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def jobs():
 result=[]
 for seed in range(6):
  for ratio in (.05,.1,.2):
   for dataset in ('rsna','chexchonet'):
    for backbone in ('resnet50','dinov2_vits14'):
     ident=f'{dataset}__{backbone}__ratio_{ratio:.2f}__seed_{seed}'
     if dataset=='rsna':
      manifest=Path(f'/nobackup/enzez/data/rsna_bone_age/2017/protocol/ratio_{ratio:.2f}_seed_{seed}.json');m=json.loads(manifest.read_text());nl=len(m['labeled_indices']);nu=len(m['unlabeled_indices']);batch=8;workers=2;module='training.rsna_bone_age_hpl_corrected';extra=[]
     else:
      manifest=ROOT/f'artifacts/protected_protocol/chexchonet_lvidd/chexchonet_lvidd_ratio_{ratio:.2f}_seed_{seed}.json';m=json.loads(manifest.read_text());nl=m['counts']['labeled_images'];nu=m['counts']['unlabeled_images'];batch=32;workers=4;module='training.chexchonet_hpl_corrected';extra=['--data-root','/nobackup/enzez/data/chexchonet/1.0.0']
     target=OUT/dataset/'runs'/f'{backbone}__ratio_{ratio:.2f}__seed_{seed}'
     command=[PYTHON,'-B','-m',module,'--method','hpl','--backbone',backbone,'--ratio',str(ratio),'--seed',str(seed),'--manifest',str(manifest),'--output-dir',str(target),'--epochs','30','--batch-size',str(batch),'--num-workers',str(workers)]+extra
     result.append(dict(id=ident,dataset=dataset,backbone=backbone,ratio=ratio,seed=seed,method='Corrected Original HPL',command=command,output_dir=str(target),manifest=str(manifest),manifest_sha256=digest(manifest),trainer=str(ROOT/(module.replace('.','/')+'.py')),batch_size=batch,labeled_examples=nl,unlabeled_examples=nu,labeled_loader_length=math.ceil(nl/batch),unlabeled_loader_length=math.ceil(nu/batch),steps_per_epoch=math.ceil(nl/batch),labeled_recycling=False,meta_iterator_preserved=True,status='pending',attempts=0,pid=None,gpu=None,exit_code=None))
 assert len(result)==len({j['id'] for j in result})==72
 return result
def validate(j):
 import numpy as np
 import torch
 p=Path(j['output_dir']);c=json.loads((p/'config.json').read_text());m=json.loads((p/'metadata.json').read_text());v=json.loads((p/'metrics.json').read_text());h=list(csv.DictReader((p/'history.csv').open()))
 assert m['status']=='complete' and m['method']=='hpl' and c['method']=='hpl'
 assert m['seed']==j['seed'] and m['backbone']==j['backbone']
 assert m.get('ratio',m.get('labeled_ratio'))==j['ratio']
 assert c['epochs']==30 and [int(x['epoch']) for x in h]==list(range(1,31))
 assert m['checkpoint_reloaded'] and not m['test_used_for_selection'] and m['test_model_inference_count']==1
 assert digest(j['manifest'])==j['manifest_sha256']
 counter='persistent_target_steps_completed' if j['dataset']=='rsna' else 'optimizer_steps'
 assert [int(x[counter]) for x in h]==[i*j['steps_per_epoch'] for i in range(1,31)]
 suffix='months' if j['dataset']=='rsna' else 'cm'
 ck=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
 assert ck['epoch']==v['best_epoch']==int(min(h,key=lambda x:float(x['validation_mae_'+suffix]))['epoch'])
 assert np.isclose(ck['validation_mae_'+suffix],min(float(x['validation_mae_'+suffix]) for x in h))
 assert np.isfinite(v['test_mae_'+suffix]) and np.isfinite(v['test_r2'])
 with np.load(p/'test_predictions.npz',allow_pickle=False) as z:
  pred=z['predictions' if j['dataset']=='rsna' else 'predictions_cm'];target=z['targets' if j['dataset']=='rsna' else 'targets_cm']
  assert len(pred)>0 and len(pred)==len(target) and np.isfinite(pred).all() and np.isfinite(target).all()
  assert np.isclose(np.abs(pred-target).mean(),v['test_mae_'+suffix])
 return True
def gpu_snapshot():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 return [(int(a),b.strip()) for a,b,c,d in (x.split(',') for x in rows.splitlines()) if int(a) in range(8) and b.strip() not in {x.strip() for x in busy} and int(c)<100 and int(d)==0]
def main():
 os.environ.update(THREADS)
 lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'queue_state.json').exists(),'Refusing duplicate campaign controller'
 js=jobs();gpus=gpu_snapshot();assert len(gpus)==8,'All eight requested GPUs must be idle at launch'
 sources={str(p):digest(p) for folder in ('training','models','data_processing') for p in (ROOT/folder).glob('*.py')}
 state=dict(controller_pid=os.getpid(),status='active',jobs=js,total_jobs=72,source_hashes=sources,started_unix=time.time(),max_attempts=1)
 write(OUT/'queue_config.json',state);(OUT/'controller.pid').write_text(str(os.getpid())+'\n')
 running={};draining=False
 def save():state['updated_unix']=time.time();write(OUT/'queue_state.json',state)
 def stop(*_):
  nonlocal draining
  draining=True;state['status']='draining';save()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def launch(gpu,uuid):
  pending=next((x for x in js if x['status']=='pending'),None)
  if pending is None or draining:return
  j=pending;p=Path(j['output_dir']);p.mkdir(parents=True,exist_ok=False)
  assert all(digest(k)==v for k,v in sources.items()),'Source changed since launch'
  assert digest(j['manifest'])==j['manifest_sha256']
  env=os.environ.copy();env.update(THREADS,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',RSNA_PHYSICAL_GPU=str(gpu))
  f=(p/'run.log').open('x');proc=subprocess.Popen(j['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  j.update(status='running',pid=proc.pid,gpu=gpu,gpu_uuid=uuid,attempts=1,started_unix=time.time(),cuda_visible_devices=uuid,local_device='cuda:0',trainer_sha256=digest(j['trainer']))
  running[gpu]=(proc,j,f,uuid);write(p/'provenance.json',j);save()
  print(json.dumps({'event':'launched','id':j['id'],'gpu':gpu,'pid':proc.pid}),flush=True)
 for gpu,uuid in sorted(gpus):launch(gpu,uuid)
 write(OUT/'first_wave.json',{'controller_pid':os.getpid(),'jobs':[{k:j[k] for k in ('id','gpu','pid','steps_per_epoch')} for j in js if j['status']=='running']})
 if '--ready-fd' in sys.argv:os.close(int(sys.argv[sys.argv.index('--ready-fd')+1]))
 while running or (not draining and any(j['status']=='pending' for j in js)):
  finished=[]
  for gpu,(proc,j,f,uuid) in list(running.items()):
   rc=proc.poll()
   if rc is None:continue
   f.close();j.update(status='validating' if rc==0 else 'failed',exit_code=rc,signal=-rc if rc<0 else None,ended_unix=time.time());del running[gpu];finished.append(j);save()
   launch(gpu,uuid)
  for j in finished:
   try:
    assert j['exit_code']==0,f"trainer exit {j['exit_code']}"
    validate(j);j.update(status='complete',artifacts_validated=True)
   except Exception as e:j.update(status='failed',artifacts_validated=False,error=repr(e))
   write(Path(j['output_dir'])/'provenance.json',j);save()
  time.sleep(1)
 state['status']='stopped' if draining else 'finished';save()
if __name__=='__main__':main()
