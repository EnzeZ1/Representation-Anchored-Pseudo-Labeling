#!/usr/bin/env python3
"""Persistent eight-GPU ratio-first formal RSNA bone-age queue."""
import json,math,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PY=Path('/nobackup/enzez/.venv/bin/python');Q=ROOT/'artifacts/benchmark_queues/rsna_bone_age_ratio_first_8gpu';DATA=Path('/nobackup/enzez/data/rsna_bone_age/2017');PROTO=DATA/'protocol';LOG=Q/'launcher.log'
LANES={0:('resnet50','supervised_step_matched'),1:('resnet50','rapl'),2:('resnet50','hpl'),3:('dinov2_vits14','supervised_step_matched'),4:('dinov2_vits14','rapl'),5:('dinov2_vits14','hpl')}
METHOD_ROOT={'supervised_step_matched':'supervised_step_matched','rapl':'rapl','hpl':'hpl','supervised_standard':'fully_supervised'}
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def log(x):s=f"[{time.strftime('%F %T')}] {x}";print(s,flush=True);Q.mkdir(parents=True,exist_ok=True);open(LOG,'a').write(s+'\n')
def valid(p):
 try:
  m=json.loads((p/'metrics.json').read_text());d=json.loads((p/'metadata.json').read_text());z=np.load(p/'test_predictions.npz',allow_pickle=False);return all((p/x).is_file() for x in ('best.pt','config.json','history.csv')) and d['status']=='complete' and d['checkpoint_reloaded'] is True and d['test_used_for_selection'] is False and d['test_model_inference_count']==1 and d['test_time_calibration'] is False and d['model_inputs']=='pixels_only' and m['test_model_inference_count']==1 and len(z['image_indices'])==1425 and all(math.isfinite(float(m[k])) for k in ('test_mae_months','test_mse_months2','test_rmse_months','test_r2','test_plcc','test_srocc'))
 except Exception:return False
def job(g,b,m,r,s,stage):
 p=ROOT/'artifacts'/METHOD_ROOT[m]/'rsna_bone_age'/b/f'ratio_{r:.2f}'/f'seed_{s}';return {'id':f'{m}:{b}:{r:.2f}:{s}','gpu_owner':g,'backbone':b,'method':m,'ratio':r,'seed':s,'stage':stage,'manifest':str(PROTO/f'ratio_{r:.2f}_seed_{s}.json'),'output_dir':str(p),'status':'complete' if valid(p) else 'pending','attempts':len(list(p.glob('attempt_*'))),'failures':0}
def jobs():
 out=[job(g,b,m,r,s,{.05:1,.10:2,.20:3}[r]) for r in (.05,.10,.20) for g,(b,m) in LANES.items() for s in range(6)];out += [job(g,b,'supervised_standard',1.,s,0) for g,b in ((6,'resnet50'),(7,'dinov2_vits14')) for s in range(6)];assert len(out)==120;return out
def free(g):
 p=subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();return not p
def preflights():
 Q.mkdir(parents=True,exist_ok=True);running=[]
 for gpu,(b,m) in LANES.items():
  path=Q/f'preflight_{b}_{m}.log';cmd=[str(PY),'-m','training.rsna_bone_age_benchmark','--method',m,'--backbone',b,'--ratio','.05','--seed','0','--manifest',str(PROTO/'ratio_0.05_seed_0.json'),'--preflight','--batch-size','8','--num-workers','0'];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);f=open(path,'w');running.append((b,m,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT),f,path))
 rows=[]
 for b,m,p,f,path in running:
  rc=p.wait();f.close();assert rc==0,(path,rc);line=path.read_text().strip().splitlines()[-1];x=json.loads(line);assert x['status']=='pass' and x['test_model_inference_count']==0 and x['model_inputs']=='pixels_only';rows.append((b,m,x))
 for b in ('resnet50','dinov2_vits14'):
  xs=[x for bb,_,x in rows if bb==b];assert len({x['target_init_hash'] for x in xs})==1;assert len({x['persistent_target_optimizer_steps'] for x in xs})==1;assert len({x['labeled_example_exposures'] for x in xs})==1
 write(Q/'preflight_summary.json',{'status':'pass','canonical_test_inference_count':0,'runs':[{'backbone':b,'method':m,**x} for b,m,x in rows]})
def persist(js,running):
 x={'updated_unix':time.time(),'runner_pid':os.getpid(),'total_jobs':120,'stage':min([j['stage'] for j in js if j['stage'] and j['status']!='complete'],default='terminal'),'jobs':js,'running':{k:{'pid':v['process'].pid,'gpu':v['gpu'],'attempt':v['attempt']} for k,v in running.items()}};write(Q/'queue_state.json',x);write(Q/'run_status.json',x)
def promote(a,c):
 c.mkdir(parents=True,exist_ok=True)
 for n in ('best.pt','config.json','metadata.json','metrics.json','history.csv','run.log','test_predictions.npz'):os.replace(a/n,c/n)
 write(c/'successful_attempt.json',{'source_attempt':a.name,'promoted_unix':time.time()})
def main():
 assert (PROTO/'audit_summary.json').is_file();assert all(free(g) for g in range(8));preflights();js=jobs();Q.mkdir(parents=True,exist_ok=True);write(Q/'queue_config.json',{'total_jobs':120,'ratio_stage_barriers':[.05,.10,.20],'full_supervised_nonblocking':True,'gpu_lanes':LANES,'jobs':js});(Q/'runner.pid').write_text(str(os.getpid())+'\n');running={}
 while True:
  for jid,x in list(running.items()):
   rc=x['process'].poll()
   if rc is None:continue
   x['log'].close();j=next(q for q in js if q['id']==jid);j['exit_code']=rc
   if rc==0 and valid(x['attempt_dir']):promote(x['attempt_dir'],Path(j['output_dir']));j['status']='complete';log(f'complete {jid}')
   elif j['failures']<1:j['failures']+=1;j['status']='pending';log(f'retryable failure {jid} rc={rc}')
   else:j['status']='failed';log(f'exhausted {jid} rc={rc}')
   del running[jid]
  for gpu in range(8):
   if any(x['gpu']==gpu for x in running.values()) or not free(gpu):continue
   cand=next((j for j in js if j['gpu_owner']==gpu and j['status']=='pending' and (j['stage']==0 or all(x['status']=='complete' for x in js if 0<x['stage']<j['stage']))),None)
   if not cand:continue
   cand['attempts']+=1;a=Path(cand['output_dir'])/f"attempt_{cand['attempts']}";a.mkdir(parents=True,exist_ok=False);f=open(a/'run.log','w');cmd=[str(PY),'-m','training.rsna_bone_age_benchmark','--method',cand['method'],'--backbone',cand['backbone'],'--ratio',str(cand['ratio']),'--seed',str(cand['seed']),'--manifest',cand['manifest'],'--output-dir',str(a),'--epochs','30','--batch-size','8','--num-workers','2'];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['RSNA_PHYSICAL_GPU']=str(gpu);p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);cand['status']='running';running[cand['id']]={'process':p,'gpu':gpu,'attempt':cand['attempts'],'attempt_dir':a,'log':f};log(f"launched {cand['id']} pid={p.pid} gpu={gpu}")
  persist(js,running)
  if not running and all(j['status'] in ('complete','failed') for j in js):break
  time.sleep(10)
 log('queue terminal')
if __name__=='__main__':main()
