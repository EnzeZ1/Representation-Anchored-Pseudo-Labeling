#!/usr/bin/env python3
"""Persistent ratio-first eight-GPU CheXchoNet LVIDd queue."""
from __future__ import annotations
import argparse,json,math,os,shlex,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'artifacts/benchmark_queues/chexchonet_lvidd_ratio_first_8gpu';OUT=ROOT/'artifacts/benchmarks/chexchonet/lvidd';PY=Path('/nobackup/enzez/.venv-chexchonet/bin/python')
METHODS=('supervised_standard','rapl','hpl');BACKBONES=('resnet50','dinov2_vits14');RATIOS=(.05,.10,.20);SEEDS=range(6)
LANES={0:('resnet50','supervised_standard'),1:('resnet50','rapl'),2:('resnet50','hpl'),3:('dinov2_vits14','supervised_standard'),4:('dinov2_vits14','rapl'),5:('dinov2_vits14','hpl')}
def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');t.replace(path)
def valid(path):
 try:
  p=Path(path);m=json.loads((p/'metadata.json').read_text());x=json.loads((p/'metrics.json').read_text())
  return (p/'best.pt').is_file() and (p/'history.csv').is_file() and (p/'test_predictions.npz').is_file() and m['status']=='complete' and m['checkpoint_reloaded'] is True and m['test_used_for_selection'] is False and m['test_model_inference_count']==1 and m['test_time_calibration'] is False and m['ordered_prediction_alignment'] is True and all(math.isfinite(float(x[k])) for k in ('test_mae_cm','test_mse_cm2','test_r2'))
 except Exception:return False
def gpu_inventory():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return {int(a):{'uuid':b.strip(),'free_mib':int(c),'utilization':int(d)} for a,b,c,d in (x.split(',') for x in rows.splitlines())}
def jobs(manifest_root):
 result=[]
 for ratio in RATIOS:
  for gpu,(backbone,method) in LANES.items():
   for seed in SEEDS:result.append(make(gpu,backbone,method,ratio,seed,manifest_root,stage=str(ratio)))
 for gpu,backbone in ((6,'resnet50'),(7,'dinov2_vits14')):
  for seed in SEEDS:result.append(make(gpu,backbone,'supervised_standard',1.,seed,manifest_root,stage='full'))
 return result
def make(gpu,b,m,r,s,mroot,stage):
 out=OUT/b/f'ratio_{r:.2f}'/m/f'seed_{s}';manifest=mroot/f'chexchonet_lvidd_ratio_{r:.2f}_seed_{s}.json'
 return {'id':f'{b}:{m}:ratio_{r:.2f}:seed_{s}','gpu':gpu,'backbone':b,'method':m,'ratio':r,'seed':s,'stage':stage,'manifest':str(manifest),'output':str(out),'status':'complete' if valid(out) else 'pending','attempts':0,'attempt_history':[]}
def command(j,data,attempt):return [str(PY),'-m','training.chexchonet_formal','--method',j['method'],'--backbone',j['backbone'],'--data-root',str(data),'--manifest',j['manifest'],'--output-dir',str(attempt),'--seed',str(j['seed']),'--ratio',str(j['ratio'])]
def persist(js,running):
 payload={'updated_unix':time.time(),'runner_pid':os.getpid(),'job_count':len(js),'current_stage':current_stage(js),'jobs':js,'running':{k:{'pid':v['process'].pid,'gpu':v['job']['gpu'],'attempt':v['job']['attempts']} for k,v in running.items()}};atomic(Q/'queue_state.json',payload);atomic(Q/'run_status.json',payload)
def current_stage(js):
 for r in RATIOS:
  group=[j for j in js if j['stage']==str(r)]
  if any(j['status']!='complete' for j in group):return f'ratio_{r:.2f}'
 return 'full_only' if any(j['status']!='complete' for j in js if j['stage']=='full') else 'complete'
def log(s):
 line=f'[{time.strftime("%F %T")}] {s}';print(line,flush=True)
 with (Q/'launcher.log').open('a') as h:h.write(line+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--manifest-root',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--poll-seconds',type=int,default=15);a=p.parse_args();Q.mkdir(parents=True,exist_ok=True);(Q/'runner.pid').write_text(str(os.getpid())+'\n')
 js=jobs(a.manifest_root);assert len(js)==120 and len({j['id'] for j in js})==120
 if any(not Path(j['manifest']).is_file() for j in js):raise RuntimeError('Protected manifest missing')
 atomic(Q/'queue_config.json',{'job_count':120,'ratio_first':True,'low_label_jobs':108,'full_supervised_jobs':12,'lanes':LANES,'jobs':js})
 running={};stop=False
 def halt(*_):
  nonlocal stop;stop=True;log('Stop requested; no new launches')
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);persist(js,running)
 while True:
  for jid,rec in list(running.items()):
   rc=rec['process'].poll()
   if rc is None:continue
   rec['handle'].close();j=rec['job'];ok=rc==0 and valid(rec['attempt']);j['attempt_history'][-1].update({'ended_unix':time.time(),'exit_code':rc,'integrity_valid':ok})
   if ok:
    dest=Path(j['output']);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and not valid(dest): dest.rename(dest.with_name(dest.name+f'_excluded_{int(time.time())}'))
    if dest.exists():raise RuntimeError('Refusing canonical overwrite')
    os.replace(rec['attempt'],dest);j['status']='complete';log(f'Completed {jid} GPU={j["gpu"]}')
   elif j['attempts']<2:j['status']='pending';j['last_failure']=f'exit {rc}; retry scheduled';log(f'Retry scheduled {jid} exit={rc}')
   else:j['status']='failed';j['last_failure']=f'exit {rc}; exhausted';log(f'Exhausted {jid}')
   del running[jid];persist(js,running)
  if not running and not any(j['status']=='pending' for j in js):break
  if not stop:
   stage=current_stage(js);allowed={'full'}
   if stage.startswith('ratio_'):allowed.add(str(float(stage.split('_')[1])))
   inventory=gpu_inventory();occupied={v['job']['gpu'] for v in running.values()}
   for gpu in range(8):
    candidates=[j for j in js if j['gpu']==gpu and j['status']=='pending' and j['stage'] in allowed]
    if not candidates or gpu in occupied:continue
    state=inventory[gpu]
    if state['free_mib']<10000 or state['utilization']>10:continue
    j=candidates[0];n=j['attempts']+1;attempt=Q/'attempts'/j['id'].replace(':','_')/f'attempt_{n}'
    if attempt.exists():raise RuntimeError(f'Attempt collision {attempt}')
    attempt.mkdir(parents=True);h=(attempt/'run.log').open('a');cmd=command(j,a.data_root,attempt);env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(Q/'matplotlib')});Path(env['MPLCONFIGDIR']).mkdir(exist_ok=True)
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);j.update({'status':'running','attempts':n,'pid':proc.pid,'cuda_visible_devices':str(gpu),'process_local_device':'cuda:0'});j['attempt_history'].append({'attempt':n,'directory':str(attempt),'pid':proc.pid,'gpu':gpu,'gpu_uuid':state['uuid'],'command':cmd,'started_unix':time.time()});running[j['id']]={'process':proc,'handle':h,'attempt':attempt,'job':j};log(f'Started {j["id"]} attempt={n} pid={proc.pid} GPU={gpu} free={state["free_mib"]}MiB')
   persist(js,running)
  time.sleep(a.poll_seconds)
 persist(js,running);log('Queue terminal')
if __name__=='__main__':main()
