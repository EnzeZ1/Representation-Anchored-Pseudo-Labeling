#!/usr/bin/env python3
"""Formal Uniform consistency control with one labeled pass per epoch."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.benchmark_io import write_json, write_npz
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from training.mean_teacher_regression import dataset_spec as shared_spec

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class TrainSet(Dataset):
 def __init__(self,ids,path,target,transform,mean,std,gray=False,labeled=False,strong=None):
  self.ids=list(ids);self.path=path;self.target=target;self.transform=transform;self.mean=mean;self.std=std;self.gray=gray;self.labeled=labeled;self.strong=strong
 def __len__(self):return len(self.ids)
 def __getitem__(self,p):
  i=self.ids[p]
  with Image.open(self.path(i)) as im:
   im=im.convert('L').convert('RGB') if self.gray else im.convert('RGB')
   if self.labeled:return self.transform(im),np.float32((self.target(i)-self.mean)/self.std),np.int64(i)
   return self.transform(im.copy()),self.strong(im),np.int64(i)

class EvalSet(Dataset):
 def __init__(self,ids,path,target,transform,gray=False):self.ids=list(ids);self.path=path;self.target=target;self.transform=transform;self.gray=gray
 def __len__(self):return len(self.ids)
 def __getitem__(self,p):
  i=self.ids[p]
  with Image.open(self.path(i)) as im:x=self.transform(im.convert('L').convert('RGB') if self.gray else im.convert('RGB'))
  return x,np.float32(self.target(i)),np.int64(i)

def spec_for(a):
 if a.dataset!='utkface':return shared_spec(a)
 from baselines.utkface_data import BenchmarkContext
 from data_processing.utkface_protocol import build_labeled_transform,build_weak_transform,build_strong_transform,build_evaluation_transform
 c=BenchmarkContext(ROOT,a.data_root,a.manifest)
 if c.manifest['seed']!=a.seed or not math.isclose(c.manifest['labeled_ratio'],a.ratio):raise RuntimeError('manifest run identity mismatch')
 rec=lambda i:c.record(i)
 return {'dataset':'UTKFace','target':'age','unit':'years','protocol':c.manifest['protocol_version'],'transform':c.manifest['transform_version'],'cohort_digest':c.cohort['cohort_sha256'],'split_digest':None,'counts':c.manifest['counts'],'mean':c.mean,'std':c.std,'ids':None,'labeled':c.indices('labeled'),'unlabeled':c.indices('unlabeled'),'val':c.indices('validation'),'test':c.indices('test'),'path':c.path,'target_fn':lambda i:float(rec(i)['age']),'student_transform':build_weak_transform(),'strong_transform':build_strong_transform(),'labeled_transform':build_labeled_transform(),'evaluation_transform':build_evaluation_transform(),'gray':False}

def loader(ds,batch,seed,role,workers,shuffle,drop):
 offsets={'labeled':11,'unlabeled':23,'validation':37,'test':53};g=torch.Generator();g.manual_seed(seed*1009+offsets[role])
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,drop_last=drop,generator=g)

@torch.no_grad()
def evaluate(model,dl,device,mean,std):
 model.eval();ps=[];ys=[];ids=[]
 for x,y,i in dl:ps.append(model(x.to(device)).float().cpu().numpy()*std+mean);ys.append(y.numpy());ids.append(i.numpy())
 p=np.concatenate(ps);y=np.concatenate(ys);idx=np.concatenate(ids);e=p-y
 return float(np.mean(np.abs(e))),float(1-np.sum(e**2)/(np.sum((y-y.mean())**2)+1e-12)),p,y,idx

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('utkface','imdb_wiki','rsna_bone_age','chexchonet'),required=True);p.add_argument('--backbone',choices=('resnet50','dinov2'),required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if not math.isclose(a.ratio,.05):raise RuntimeError('formal Uniform no-Reorient is 5% only')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('worker must see exactly one CUDA device')
 s=spec_for(a); labeled_transform=s.get('labeled_transform',s['student_transform']);strong=s.get('strong_transform',s.get('teacher_transform'))
 lb=loader(TrainSet(s['labeled'],s['path'],s['target_fn'],labeled_transform,s['mean'],s['std'],s['gray'],True),a.batch_size,a.seed,'labeled',a.workers,True,True)
 ulb_ids=s.get('unlabeled') or [i for i in s['ids'] if i not in set(s['labeled'])]
 ulb=loader(TrainSet(ulb_ids,s['path'],s['target_fn'],s['student_transform'],s['mean'],s['std'],s['gray'],False,strong),a.batch_size,a.seed,'unlabeled',a.workers,True,True)
 val=loader(EvalSet(s['val'],s['path'],s['target_fn'],s['evaluation_transform'],s['gray']),a.batch_size,a.seed,'validation',a.workers,False,False);test=loader(EvalSet(s['test'],s['path'],s['target_fn'],s['evaluation_transform'],s['gray']),a.batch_size,a.seed,'test',a.workers,False,False)
 if not lb or not ulb or len(lb)>len(ulb):raise RuntimeError('invalid no-Reorient trajectory lengths')
 device=torch.device('cuda:0');model=(ResNet50Regressor(True) if a.backbone=='resnet50' else DINOv2Regressor('small')).to(device)
 if a.backbone=='dinov2':opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);optcfg={'optimizer':'AdamW','backbone_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'scheduler':'CosineAnnealingLR','T_max':a.epochs}
 else:opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3);sch=torch.optim.lr_scheduler.StepLR(opt,10,.1);optcfg={'optimizer':'Adam','lr':1e-4,'weight_decay':1e-3,'scheduler':'StepLR','step_size':10,'gamma':.1}
 out=a.output_dir
 if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing nonempty output {out}')
 out.mkdir(parents=True,exist_ok=True);started=time.time();torch.cuda.reset_peak_memory_stats()
 scientific={'method':'uniform','plugin':'none','supervised_recycling':False,'labeled_passes_per_epoch':1,'labeled_iterator_restart':False,'full_unlabeled_trajectory':True,'lambda_u':1,'reliability_weight':1,'learned_reliability_weighting':False,'uncertainty_learner':False,'ema_teacher':False,'temporal_target':False,'confidence_filtering':False,'reorient_active':False,'main_steps_per_epoch':len(ulb),'ours_main_steps_per_epoch':len(ulb),'objective':'y_hat_u=stopgrad(f(weak(x_u))); L_u=MSE(f(strong(x_u)),y_hat_u); L_student=L_sup+L_u while labeled batch exists, else L_u'}
 config={'dataset':s['dataset'],'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'optimization':optcfg,'selection_metric':'validation MAE only','protocol_version':s['protocol'],'transform_version':s['transform'],'manifest_sha256':digest(a.manifest),'cohort_digest':s['cohort_digest'],'split_digest':s['split_digest'],'counts':s['counts'],'label_scaler':{'mean':s['mean'],'std':s['std']},**scientific};write_json(out/'config.json',config)
 best=math.inf;best_epoch=None;history=[];steps=0;labeled_batches=0
 for epoch in range(1,a.epochs+1):
  model.train();li=iter(lb);labeled_exhausted=False;sums=np.zeros(3)
  for weak,strong,_ in ulb:
   weak,strong=weak.to(device),strong.to(device);sup=torch.zeros((),device=device)
   if not labeled_exhausted:
    try:x,y,_=next(li);x,y=x.to(device),y.to(device);sup=F.mse_loss(model(x),y);labeled_batches+=1
    except StopIteration:labeled_exhausted=True
   with torch.no_grad():pseudo=model(weak).detach()
   unsup=F.mse_loss(model(strong),pseudo);loss=sup+unsup
   if not all(torch.isfinite(v) for v in (sup,unsup,loss)):raise RuntimeError('non-finite training loss')
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();steps+=1;sums+=np.array([float(loss),float(sup),float(unsup)])
  if labeled_batches!=epoch*len(lb):raise RuntimeError('labeled pass accounting failure')
  sch.step();vm,vr,_,_,_=evaluate(model,val,device,s['mean'],s['std']);improved=vm<best
  if improved:best=vm;best_epoch=epoch;torch.save({'model':model.state_dict(),'epoch':epoch,'validation_mae':vm,'manifest_sha256':digest(a.manifest),'config':config},out/'best.pt')
  history.append({'epoch':epoch,'loss_total':sums[0]/len(ulb),'loss_supervised':sums[1]/len(ulb),'loss_unlabeled':sums[2]/len(ulb),'validation_mae':vm,'validation_r2':vr,'optimizer_steps':steps,'labeled_batches_consumed':labeled_batches});print(f'epoch={epoch}/{a.epochs} val_mae={vm:.6f}',flush=True)
 with (out/'history.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=history[0]);w.writeheader();w.writerows(history)
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False)
 if ck['epoch']!=best_epoch or ck['validation_mae']!=best or ck['manifest_sha256']!=digest(a.manifest):raise RuntimeError('checkpoint identity failure')
 model.load_state_dict(ck['model']);tm,tr,pred,target,idx=evaluate(model,test,device,s['mean'],s['std'])
 if not np.array_equal(idx,np.asarray(s['test'])) or len(pred)==0 or not np.isfinite([tm,tr,*pred]).all():raise RuntimeError('test contract failure')
 write_npz(out/'test_predictions.npz',cohort_indices=idx,predictions=pred,targets=target);write_json(out/'metrics.json',{'validation_mae':best,'test_mae':tm,'test_r2':tr})
 write_json(out/'metadata.json',{'status':'complete','dataset':s['dataset'],'backbone':a.backbone,'seed':a.seed,'labeled_ratio':a.ratio,'checkpoint_reloaded':True,'test_used_for_selection':False,'test_evaluations':1,'test_model_inference_count':1,'nonempty_test_predictions':True,'finite_mae_r2':True,'predictions_in_manifest_order':True,'prediction_count':len(pred),'best_epoch':best_epoch,'best_validation_mae':best,'optimizer_steps':steps,'expected_optimizer_steps':a.epochs*len(ulb),'labeled_batches_consumed':labeled_batches,'expected_labeled_batches':a.epochs*len(lb),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'wall_clock_seconds':time.time()-started,**scientific});print(json.dumps({'test_mae':tm,'test_r2':tr}),flush=True)
if __name__=='__main__':main()
