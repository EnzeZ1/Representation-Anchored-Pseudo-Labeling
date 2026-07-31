#!/usr/bin/env python3
"""Persistent fixed-lane queue for the 72 formal STS-B RAPL/HPL runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path("/nobackup/enzez/.venv/bin/python")
QUEUE=ROOT/"artifacts/benchmark_queues/stsb_rapl_hpl_8gpu"
STATE=QUEUE/"queue_state.json";STATUS=QUEUE/"run_status.json";LOG=QUEUE/"launcher.log";PID=QUEUE/"runner.pid"
LANES={
 0:[("rapl","bilstm_glove",.05)],1:[("hpl","bilstm_glove",.05)],
 2:[("rapl","bilstm_glove",.10)],3:[("hpl","bilstm_glove",.10)],
 4:[("rapl","bilstm_glove",.20)],5:[("hpl","bilstm_glove",.20)],
 6:[("rapl","roberta_base",r) for r in (.05,.10,.20)],
 7:[("hpl","roberta_base",r) for r in (.05,.10,.20)],
}


def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(path)


def log(value):
 line=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {value}";print(line,flush=True)
 with LOG.open("a") as f:f.write(line+"\n")


def valid(path):
 try:
  req=("best.pt","config.json","metadata.json","metrics.json","history.csv","run.log","test_predictions.npz")
  if not all((path/x).is_file() for x in req):return False
  m=json.loads((path/"metrics.json").read_text());d=json.loads((path/"metadata.json").read_text())
  return d["status"]=="complete" and d["protocol_version"]=="stsb-benchmark-v1" and d["augmentation_version"]=="rapl-text-augmentation-v1" and d["checkpoint_reloaded"] is True and d["test_used_for_selection"] is False and d["test_model_inference_count"]==1 and m["checkpoint_reloaded"] is True and m["test_used_for_selection"] is False and m["test_model_inference_count"]==1 and all(math.isfinite(float(m[k])) for k in ("best_validation_mse_score_units","test_mse_score_units","test_r2"))
 except Exception:return False


def output(method,backbone,ratio,seed):return ROOT/f"artifacts/{method}/stsb"/backbone/f"ratio_{ratio:.2f}"/f"seed_{seed}"


def jobs():
 result=[]
 for gpu,groups in LANES.items():
  for method,backbone,ratio in groups:
   for seed in range(6):
    out=output(method,backbone,ratio,seed);attempts=max([int(x.name[8:]) for x in out.glob("attempt_*") if x.name[8:].isdigit()] or [0])
    result.append({"id":f"{method}:{backbone}:ratio_{ratio:.2f}:seed_{seed}","method":method,"backbone":backbone,"labeled_ratio":ratio,"seed":seed,"gpu_owner":gpu,"manifest":str(ROOT/f"data_processing/splits/stsb_ratio_{ratio:.2f}_seed_{seed}.json"),"output_dir":str(out),"attempts":attempts,"failures":0,"status":"complete" if valid(out) else "pending"})
 assert len(result)==72 and len({x["id"] for x in result})==72
 return result


def command(job,attempt):
 out=Path(job["output_dir"])/f"attempt_{attempt}"
 return [str(PYTHON),"-m","training.stsb_ssl","--method",job["method"],"--backbone",job["backbone"],"--manifest",job["manifest"],"--output-dir",str(out),"--seed",str(job["seed"]),"--labeled-ratio",str(job["labeled_ratio"]),"--num-workers","2"],out


def promote(source,destination):
 destination.mkdir(parents=True,exist_ok=True)
 for name in ("best.pt","config.json","metadata.json","metrics.json","history.csv","run.log","test_predictions.npz"):
  os.replace(source/name,destination/name)
 write_json(destination/"successful_attempt.json",{"source_attempt":source.name,"promoted_unix":time.time()})


def recover_valid_attempt(job):
 destination=Path(job["output_dir"])
 attempts=sorted(destination.glob("attempt_*"),key=lambda path:int(path.name[8:]) if path.name[8:].isdigit() else -1,reverse=True)
 for attempt in attempts:
  if valid(attempt):
   promote(attempt,destination);job["status"]="complete";log(f"Recovered completed {job['id']} from {attempt.name}")
   return True
 return False


def gpu_free(gpu):
 text=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip();free,util=[int(x.strip()) for x in text.split(",")]
 p=subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
 return not p and free>=9000 and util<=10


def persist(records,running,stopping=False):
 payload={"updated_unix":time.time(),"runner_pid":os.getpid(),"total_jobs":72,"stopping":stopping,"jobs":records,"running":{k:{"pid":v["process"].pid,"physical_gpu":v["gpu"],"attempt":v["attempt"]} for k,v in running.items()}}
 write_json(STATE,payload);write_json(STATUS,payload)


def main():
 parser=argparse.ArgumentParser();parser.add_argument("--resume",action="store_true");parser.add_argument("--poll-seconds",type=int,default=15);args=parser.parse_args()
 QUEUE.mkdir(parents=True,exist_ok=True);PID.write_text(str(os.getpid())+"\n");records=jobs();running={};stopping=False
 write_json(QUEUE/"queue_config.json",{"dataset":"STS-B-DIR","protocol_version":"stsb-benchmark-v1","augmentation_version":"rapl-text-augmentation-v1","methods":["rapl","hpl"],"backbones":["bilstm_glove","roberta_base"],"ratios":[.05,.10,.20],"seeds":list(range(6)),"total_jobs":72,"lanes":LANES,"selection_metric":"validation MSE in original STS-B score units"})
 def stop(*_):
  nonlocal stopping;stopping=True;log("Stop requested; no new jobs will launch")
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);persist(records,running)
 while True:
  for job_id,item in list(running.items()):
   code=item["process"].poll()
   if code is None:continue
   item["handle"].close();job=item["job"]
   if code==0 and valid(item["attempt_dir"]):promote(item["attempt_dir"],Path(job["output_dir"]));job["status"]="complete";log(f"Completed {job_id} GPU={item['gpu']}")
   elif job["failures"]<1:job["failures"]+=1;job["status"]="pending";job["last_failure"]=f"attempt {item['attempt']} exit {code}";log(f"Retry scheduled {job_id}: exit {code}")
   else:job["status"]="failed";job["last_failure"]=f"attempt {item['attempt']} exit {code}; exhausted";log(f"Failed {job_id}; exhausted")
   del running[job_id];persist(records,running,stopping)
  if not running and (stopping or not any(x["status"]=="pending" for x in records)):break
  if not stopping:
   used={x["gpu"] for x in running.values()}
   for gpu in range(8):
    if gpu in used or not gpu_free(gpu):continue
    job=next((x for x in records if x["gpu_owner"]==gpu and x["status"]=="pending"),None)
    if not job:continue
    if recover_valid_attempt(job):persist(records,running,stopping);continue
    job["attempts"]+=1;job["status"]="running";cmd,out=command(job,job["attempts"]);out.mkdir(parents=True,exist_ok=False);handle=(out/"run.log").open("a")
    env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONPATH":str(ROOT),"PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false"})
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    running[job["id"]]={"process":process,"handle":handle,"gpu":gpu,"attempt":job["attempts"],"attempt_dir":out,"job":job};log(f"Started {job['id']} pid={process.pid} physical GPU={gpu}");persist(records,running,stopping)
  time.sleep(args.poll_seconds)
 persist(records,running,stopping);failed=[x for x in records if x["status"]=="failed"];log(f"Queue terminal complete={sum(x['status']=='complete' for x in records)} failed={len(failed)}");raise SystemExit(bool(failed))


if __name__=="__main__":main()
