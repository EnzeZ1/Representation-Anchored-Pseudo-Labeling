#!/usr/bin/env python3
"""Immutable, Uniform-only detached queue for the canonical frequency sweep."""
import argparse, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts/anchor_frequency_uniform_canonical_utkface_5pct"; Q=ART/"queue"; STATE=Q/"queue_state.json"; LOCK=Q/"queue.lock"
PY=Path("/nobackup/enzez/.venv/bin/python"); TRAIN=ROOT/"training/canonical_uniform_frequency.py"; BACKBONES=("resnet50","dinov2"); SEEDS=(0,1,2); KS=(1,2,4,8,20)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def locked(fn):
 LOCK.touch(exist_ok=True)
 with LOCK.open("r+") as h:
  fcntl.flock(h,fcntl.LOCK_EX);s=json.loads(STATE.read_text());r=fn(s);s["updated_unix"]=time.time();write(STATE,s);return r
def claim(gpu):
 def f(s):
  for j in s["jobs"]:
   if j["training_status"]=="pending":
    j["training_status"]="running";j["attempts"]+=1;j["pid"]=os.getpid();j["physical_gpu"]=gpu;j["cuda_visible_devices"]=str(gpu);j["start_time"]=time.time()
    d=Q/"attempts"/j["id"]/f'attempt_{j["attempts"]}';d.mkdir(parents=True,exist_ok=False);j["attempt_directory"]=str(d);s["workers"][str(gpu)]={"pid":os.getpid(),"job_id":j["id"],"status":"running"};return dict(j)
  s["workers"][str(gpu)]={"pid":os.getpid(),"job_id":None,"status":"complete"};return None
 return locked(f)
def reconcile():
 """Fail closed: stale work is never changed back to pending automatically."""
 def f(s):
  for j in s["jobs"]:
   if j["training_status"]!="running":continue
   try: os.kill(int(j.get("pid",-1)),0); alive=True
   except (OSError,ValueError): alive=False
   if alive:continue
   d=Path(j.get("attempt_directory",""))/"run"; st=d/"status.json"
   saved=json.loads(st.read_text()) if st.is_file() else {}
   if saved.get("training_status")=="complete" and saved.get("final_evaluation_status")=="complete" and (d/"final_test_complete").is_file():
    j.update(training_status="complete",final_evaluation_status="complete",analysis_status=saved.get("analysis_status","pending"),artifact_directory=str(d),reconciliation="preserved_scientific_completion")
   else:j.update(training_status="failed",reconciliation="stale_worker_not_requeued")
 return True
 return locked(f)
def finish(job,code):
 d=Path(job["attempt_directory"]); st=json.loads((d/"status.json").read_text()) if (d/"status.json").is_file() else {}
 def f(s):
  j=next(x for x in s["jobs"] if x["id"]==job["id"]);j.update(training_status=st.get("training_status","failed"),final_evaluation_status=st.get("final_evaluation_status","failed"),analysis_status=st.get("analysis_status","pending"),exit_code=code,end_time=time.time())
  if j["training_status"]=="complete" and j["final_evaluation_status"]=="complete": j["artifact_directory"]=str(d)
  s["workers"][str(job["physical_gpu"])]={"pid":os.getpid(),"job_id":None,"status":"idle"}
 locked(f)
def worker(gpu):
 reconcile()
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),PYTHONUNBUFFERED="1")
 while True:
  j=claim(gpu)
  if j is None:return
  d=Path(j["attempt_directory"]);cmd=[str(PY),str(TRAIN),"--backbone",j["backbone"],"--seed",str(j["seed"]),"--k",str(j["k"]),"--attempt",str(j["attempts"]),"--config-hash",j["config_hash"],"--output",str(d/"run")]
  try:
   with (d/"run.log").open("a") as log: code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  finally:
   run=d/"run"; j["attempt_directory"]=str(run); finish(j,locals().get("code",255))
def initialize(manifest_hash):
 if STATE.exists():raise RuntimeError(f"refusing to overwrite {STATE}")
 jobs=[{"id":f"uniform__{b}__seed_{s}__k_{k}","method":"uniform","backbone":b,"seed":s,"k":k,"attempts":0,"training_status":"pending","final_evaluation_status":"pending","analysis_status":"pending","config_hash":manifest_hash} for b in BACKBONES for s in SEEDS for k in KS]
 write(STATE,{"experiment":"canonical_uniform_frequency_utkface_5pct","jobs":jobs,"workers":{},"created_unix":time.time()})
def prepare():
 if ART.exists():raise RuntimeError(f"refusing to overwrite {ART}")
 import platform, sys, torch, torchvision
 cohort=ROOT/"data_processing/splits/utkface_cohort_v1.json"; jobs=[f"uniform__{b}__seed_{s}__k_{k}" for b in BACKBONES for s in SEEDS for k in KS]
 seeds={}
 for s in SEEDS:
  p=ROOT/f"data_processing/splits/utkface_ratio_0.05_seed_{s}.json"; m=json.loads(p.read_text());seeds[str(s)]={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"scaler":m["label_scaler"]}
 diff=subprocess.run(["git","diff","--binary","HEAD"],cwd=ROOT,stdout=subprocess.PIPE,check=True).stdout
 scientific={"method":"uniform","dataset":"UTKFace","ratio":.05,"cohort_digest":"61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56","seed_manifests":seeds,
 "transforms":{"labeled":"RandomResizedCrop(224,scale=(.8,1),ratio=(.75,4/3))+RandomHorizontalFlip(.5)+ImageNetNormalize","weak":"same as labeled","strong":"weak crop/flip family+RandAugment(num_ops=2,magnitude=10)+ImageNetNormalize","evaluation":"Resize(256)+CenterCrop(224)+ImageNetNormalize"},
 "resnet":{"model":"ResNet50Regressor/IMAGENET1K_V1/Linear(2048,1)/dropout=.2","optimizer":"Adam one group lr=1e-4 weight_decay=.001","scheduler":"StepLR(10,.1) epoch-level"},
 "dinov2":{"model":"DINOv2Regressor/dinov2_vits14/normalized CLS/Linear(384,1)/dropout=.2","optimizer":"AdamW backbone_lr=1e-5 head_lr=1e-4 weight_decay=.05","scheduler":"CosineAnnealingLR(T_max=30) epoch-level"},
 "lambda_u":1.0,"r":1.0,"gradient_clipping":5.0,"epochs":30,"steps_per_epoch":563,"k":[1,2,4,8,20],"anchor_counts":[563,282,141,71,29],"jobs":jobs}
 blob=json.dumps(scientific,sort_keys=True,separators=(",",":")).encode();config_hash=hashlib.sha256(blob).hexdigest()
 manifest={"experiment":"canonical_uniform_frequency_utkface_5pct","created_unix":time.time(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"working_tree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),"uncommitted_diff_sha256":hashlib.sha256(diff).hexdigest(),"runtime":{"python":sys.version,"pytorch":torch.__version__,"torchvision":torchvision.__version__,"cuda":torch.version.cuda,"platform":platform.platform()},"gpu_models":["NVIDIA GeForce RTX 2080 Ti"]*8,"scientific_configuration":scientific,"config_hash":config_hash,
 "posthoc":{"reference_drift":{"k":"all","epochs":[1,5,10,20,30],"reference_subset":"artifacts/anchor_mechanism_utkface_5pct/diagnostics/reference_subset.json","test_data":False},"gradient_steering":{"k":[1],"epochs":[1,5,10,20,30],"pairs_per_seed_epoch":8,"g_ref_microbatch":8,"g_total":"g_s + g_u","test_data":False}}}
 write(ART/"frozen_manifest.json",manifest);initialize(config_hash);print(config_hash)
def main():
 p=argparse.ArgumentParser();p.add_argument("--initialize",action="store_true");p.add_argument("--prepare",action="store_true");p.add_argument("--manifest-hash");p.add_argument("--worker",type=int);a=p.parse_args()
 if a.prepare:prepare()
 elif a.initialize:initialize(a.manifest_hash)
 elif a.worker is not None:worker(a.worker)
 else:p.error("mode required")
if __name__=="__main__":main()
