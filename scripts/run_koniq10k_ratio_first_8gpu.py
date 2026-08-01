#!/usr/bin/env python3
"""Persistent eight-GPU ratio-first KonIQ-10k formal queue."""
import csv,json,math,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PY=Path('/nobackup/enzez/.venv/bin/python');Q=ROOT/'artifacts/benchmark_queues/koniq10k_ratio_first_8gpu';RESULT=ROOT/'artifacts/koniq10k_benchmarks';MAN=ROOT/'artifacts/koniq10k_protocol';LOG=Q/'launcher.log'
LANES={0:('resnet50','supervised_step_matched'),1:('resnet50','rapl'),2:('resnet50','hpl'),3:('dinov2_vits14','supervised_step_matched'),4:('dinov2_vits14','rapl'),5:('dinov2_vits14','hpl')}
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def log(x):s=f"[{time.strftime('%F %T')}] {x}";print(s,flush=True);Q.mkdir(parents=True,exist_ok=True);open(LOG,'a').write(s+'\n')
def valid(p):
 try:
  m=json.loads((p/'metrics.json').read_text());d=json.loads((p/'metadata.json').read_text());z=np_load(p/'test_predictions.npz');return all((p/x).is_file() for x in ('best.pt','config.json','history.csv')) and d['status']=='complete' and d['checkpoint_reloaded'] is True and d['test_used_for_selection'] is False and d['test_model_inference_count']==1 and d['test_time_calibration'] is False and m['checkpoint_reloaded'] is True and m['test_model_inference_count']==1 and all(math.isfinite(float(m[k])) for k in ('best_validation_mse_mos','test_mse_mos','test_mae_mos','test_r2','test_plcc','test_srocc')) and len(z['image_identifiers'])==2015
 except Exception:return False
def np_load(p):import numpy as np;return np.load(p,allow_pickle=False)
def jobs():
 out=[]
 for ratio in (.05,.10,.20):
  for gpu,(b,m) in LANES.items():
   for seed in range(6):out.append(job(gpu,b,m,ratio,seed,stage={.05:1,.10:2,.20:3}[ratio]))
 for gpu,b in ((6,'resnet50'),(7,'dinov2_vits14')):
  for seed in range(6):out.append(job(gpu,b,'supervised_standard',1.,seed,stage=0))
 assert len(out)==120 and len({x['id'] for x in out})==120;return out
def job(g,b,m,r,s,stage):
 p=RESULT/m/b/f'ratio_{r:.2f}'/f'seed_{s}';return {'id':f'{m}:{b}:{r:.2f}:{s}','gpu_owner':g,'backbone':b,'method':m,'ratio':r,'seed':s,'stage':stage,'manifest':str(MAN/f'koniq10k_ratio_{r:.2f}_seed_{s}.json'),'output_dir':str(p),'status':'complete' if valid(p) else 'pending','attempts':len(list(p.glob('attempt_*'))),'failures':0}
def free(g):
 x=subprocess.check_output(['nvidia-smi',f'--id={g}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');p=subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();return not p and int(x[0])>=9000 and int(x[1])<=10
def stage_open(js,j):return j['stage']==0 or all(x['status']=='complete' for x in js if x['stage']<j['stage'] and x['stage']>0)
def persist(js,running):
 x={'updated_unix':time.time(),'runner_pid':os.getpid(),'total_jobs':120,'stage':min([j['stage'] for j in js if j['stage'] and j['status']!='complete'],default='terminal'),'jobs':js,'running':{k:{'pid':v['p'].pid,'gpu':v['gpu'],'attempt':v['attempt']} for k,v in running.items()}};write(Q/'queue_state.json',x);write(Q/'run_status.json',x)
def promote(a,c):
 c.mkdir(parents=True,exist_ok=True)
 for n in ('best.pt','config.json','metadata.json','metrics.json','history.csv','run.log','test_predictions.npz'):os.replace(a/n,c/n)
 write(c/'successful_attempt.json',{'source_attempt':a.name,'promoted_unix':time.time()})
def main():
 Q.mkdir(parents=True,exist_ok=True);(Q/'runner.pid').write_text(str(os.getpid())+'\n');js=jobs();write(Q/'queue_config.json',{'total_jobs':120,'ratio_stage_barriers':[.05,.10,.20],'full_supervised_nonblocking':True,'gpu_lanes':LANES,'jobs':js});running={};stop=False
 def sig(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
 while not stop:
  for jid,x in list(running.items()):
   rc=x['p'].poll()
   if rc is None:continue
   j=next(z for z in js if z['id']==jid);j['exit_code']=rc
   if rc==0 and valid(x['attempt_dir']):promote(x['attempt_dir'],Path(j['output_dir']));j['status']='complete';log(f'complete {jid}')
   elif j['failures']<1:j['failures']+=1;j['status']='pending';log(f'retryable failure {jid} rc={rc}')
   else:j['status']='failed';log(f'exhausted {jid} rc={rc}')
   del running[jid];persist(js,running)
  for gpu in range(8):
   if any(x['gpu']==gpu for x in running.values()) or not free(gpu):continue
   cand=next((j for j in js if j['gpu_owner']==gpu and j['status']=='pending' and stage_open(js,j)),None)
   if not cand:continue
   cand['attempts']+=1;a=Path(cand['output_dir'])/f"attempt_{cand['attempts']}";a.mkdir(parents=True,exist_ok=False);f=open(a/'run.log','w');cmd=[str(PY),'-m','training.koniq10k_benchmark','--method',cand['method'],'--backbone',cand['backbone'],'--ratio',str(cand['ratio']),'--seed',str(cand['seed']),'--manifest',cand['manifest'],'--data-root',str(ROOT/'data/koniq10k/extracted/512x384'),'--output-dir',str(a),'--epochs','30','--batch-size','16','--num-workers','2'];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['KONIQ_PHYSICAL_GPU']=str(gpu);p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);cand['status']='running';running[cand['id']]={'p':p,'gpu':gpu,'attempt':cand['attempts'],'attempt_dir':a,'log':f};log(f"launched {cand['id']} pid={p.pid} gpu={gpu}")
  persist(js,running)
  if not running and all(j['status'] in ('complete','failed') for j in js):break
  time.sleep(10)
 log('queue terminal')
if __name__=='__main__':main()
