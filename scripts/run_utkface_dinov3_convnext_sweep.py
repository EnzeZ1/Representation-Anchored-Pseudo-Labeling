#!/usr/bin/env python3
"""Persistent preflight-gated DINOv3 ConvNeXt-Tiny UTKFace queue."""
import argparse,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from baselines.benchmark_io import run_complete,write_json
Q=ROOT/'artifacts/benchmark_queues/utkface_dinov3_convnext';OUT=ROOT/'artifacts/benchmarks/utkface/dinov3_convnext_tiny/ratio_0.05';DEPS=ROOT/'artifacts/utkface_5pct/python_deps';CACHE=ROOT/'artifacts/model_cache/huggingface'
STATE=Q/'queue_state.json';STATUS=Q/'run_status.json';CONFIG=Q/'queue_config.json';LOG=Q/'launcher.log';MODEL='facebook/dinov3-convnext-tiny-pretrain-lvd1689m';SHA='bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2';SEEDS=tuple(range(6));METHODS=('rapl','hpl')
def arguments():
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry_run',action='store_true');p.add_argument('--max_parallel',type=int,default=6);p.add_argument('--available_gpus',nargs='+',type=int,default=list(range(6)));p.add_argument('--poll_seconds',type=int,default=30);return p.parse_args()
def log(s):
 line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {s}';print(line,flush=True)
 with LOG.open('a') as h:h.write(line+'\n')
def gpu_state():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);states=[{'index':int(a.strip()),'uuid':b.strip(),'free':int(c.strip()),'util':int(d.strip())} for a,b,c,d in (line.split(',') for line in text.splitlines())]
 try:busy={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines() if line.strip()}
 except subprocess.CalledProcessError:busy=set()
 return [s for s in states if s['uuid'] not in busy]
def idle(allowed,occupied):return [s for s in gpu_state() if s['index'] in allowed and s['index'] not in occupied and s['free']>=9000 and s['util']<=10]
def env(gpu):
 e=os.environ.copy();e.update({'CUDA_VISIBLE_DEVICES':str(gpu),'UTKFACE_BENCHMARK_ROOT':str(ROOT),'PYTHONPATH':os.pathsep.join([str(DEPS),str(ROOT),e.get('PYTHONPATH','')]),'HF_HOME':str(CACHE),'TRANSFORMERS_OFFLINE':'1','HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','MPLCONFIGDIR':str(Q/'matplotlib'),'NUMBA_CACHE_DIR':str(Q/'numba_cache')});Path(e['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True);Path(e['NUMBA_CACHE_DIR']).mkdir(parents=True,exist_ok=True);return e
def command(method,seed):
 out=OUT/method/f'seed_{seed}';common=['--benchmark_manifest',str(ROOT/f'data_processing/splits/utkface_ratio_0.05_seed_{seed}.json'),'--benchmark_output_dir',str(out),'--data_dir',str(ROOT/'data/utkface_all'),'--seed',str(seed)]
 if method=='rapl':return ROOT,[sys.executable,str(ROOT/'train.py'),'-dataset','utkface','--method','probe','--labeled_ratio','0.05','--backbone','dinov3_convnext_tiny','--probe_backbone','dinov3_convnext_tiny','--save',str(out/'best.pt'),*common]
 return ROOT/'third_party/Heteroscedastic-Pseudo-Labels/utkface',[sys.executable,'main_ours.py','--benchmark_backbone','dinov3_convnext_tiny',*common]
def persist(stage,jobs,running):write_json(STATE,{'updated_unix':time.time(),'runner_pid':os.getpid(),'stage':stage,'model_identifier':MODEL,'weight_sha256':SHA,'jobs':jobs,'running':{k:{'pid':v['process'].pid,'gpu':v['gpu']} for k,v in running.items()}});write_json(STATUS,json.loads(STATE.read_text()))
def main():
 a=arguments();Q.mkdir(parents=True,exist_ok=True);(Q/'runner.pid').write_text(str(os.getpid())+'\n');jobs=[]
 for method in METHODS:
  for seed in SEEDS:
   cwd,cmd=command(method,seed);out=OUT/method/f'seed_{seed}';jobs.append({'id':f'{method}:ratio_0.05:seed_{seed}','experiment_id':f'utkface-r0.05-dinov3-convnext-tiny-{method}-seed-{seed}','dataset':'UTKFace','method':method,'labeled_ratio':.05,'rng_seed':seed,'target_backbone':MODEL,'probe_backbone':MODEL if method=='rapl' else None,'manifest_path':str(ROOT/f'data_processing/splits/utkface_ratio_0.05_seed_{seed}.json'),'artifact_directory':str(out),'cwd':str(cwd),'command':cmd,'status':'complete' if a.resume and run_complete(out) else 'pending','attempts':0})
 assert len(jobs)==12 and len({j['experiment_id'] for j in jobs})==12 and all(j['labeled_ratio']==.05 and 'resnet50' not in ' '.join(j['command']).lower() and 'dinov2' not in ' '.join(j['command']).lower() for j in jobs)
 for method in METHODS:assert {j['rng_seed'] for j in jobs if j['method']==method}==set(SEEDS)
 write_json(CONFIG,{'job_count':12,'model_identifier':MODEL,'model_revision':'10d30274b4d445111e2d5bf75ac93bbd94db274b','weight_sha256':SHA,'feature_dimension':768,'jobs':jobs})
 if a.dry_run:
  for j in jobs:print(f"cd {shlex.quote(j['cwd'])} && CUDA_VISIBLE_DEVICES=<gpu> {shlex.join(j['command'])}")
  return
 persist('waiting_for_preflight_gpu',jobs,{})
 while not idle(a.available_gpus,set()):time.sleep(a.poll_seconds)
 gpu=idle(a.available_gpus,set())[0]['index'];preflight=[]
 for method in METHODS:
  path=Q/f'preflight_{method}.log';cmd=[sys.executable,str(ROOT/'scripts/preflight_utkface_dinov3_convnext.py'),'--method',method]
  with path.open('w') as h:rc=subprocess.run(cmd,cwd=ROOT,env=env(gpu),stdout=h,stderr=subprocess.STDOUT).returncode
  preflight.append((method,rc));log(f'Preflight {method} GPU={gpu} exit={rc}')
  if rc:write_json(Q/'preflight_failure.json',{'method':method,'exit_code':rc,'log':str(path),'formal_jobs_launched':0});persist('preflight_failed',jobs,{});return 1
 running={};persist('formal',jobs,running)
 while True:
  for jid,r in list(running.items()):
   rc=r['process'].poll()
   if rc is None:continue
   r['handle'].close();j=r['job'];ok=rc==0 and run_complete(j['artifact_directory'])
   if ok:j['status']='complete';log(f'Completed {jid} GPU={r["gpu"]}');subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=False)
   elif j['attempts']<2:j['status']='pending';j['last_failure']=f'exit {rc}; retry scheduled';log(f'Failed {jid}; scheduling retry')
   else:j['status']='failed';j['last_failure']=f'exit {rc}; retries exhausted';log(f'Failed {jid}; retries exhausted')
   del running[jid];persist('formal',jobs,running)
  pending=[j for j in jobs if j['status']=='pending']
  if not running and not pending:break
  for g,j in zip(idle(a.available_gpus,{v['gpu'] for v in running.values()}),pending[:a.max_parallel-len(running)]):
   out=Path(j['artifact_directory'])
   if run_complete(out):j['status']='complete';continue
   if out.exists() and any(out.iterdir()):j['status']='failed';j['last_failure']='nonempty incomplete output collision';continue
   out.mkdir(parents=True,exist_ok=True);h=(out/'run.log').open('a');j['attempts']+=1;j['status']='running';j['physical_gpu_index']=g['index'];j['gpu_uuid']=g['uuid'];p=subprocess.Popen(j['command'],cwd=j['cwd'],env=env(g['index']),stdout=h,stderr=subprocess.STDOUT,start_new_session=True);j['pid']=p.pid;running[j['id']]={'process':p,'handle':h,'gpu':g['index'],'job':j};log(f"Started {j['id']} pid={p.pid} GPU={g['index']}: {shlex.join(j['command'])}");persist('formal',jobs,running)
  time.sleep(a.poll_seconds)
 subprocess.run([sys.executable,str(ROOT/'scripts/update_benchmark_registry.py')],cwd=ROOT,check=False);subprocess.run([sys.executable,str(ROOT/'scripts/report_dinov3_convnext_utkface.py')],cwd=ROOT,check=False);persist('complete',jobs,{});return 1 if any(j['status']=='failed' for j in jobs) else 0
if __name__=='__main__':raise SystemExit(main())
