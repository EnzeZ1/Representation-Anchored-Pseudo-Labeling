#!/usr/bin/env python3
"""Formal UTKFace regression adapter for CuriousAI/mean-teacher.

The update equations and update order are direct ports of upstream pytorch/main.py
at 546348ff863c998c26be4339021425df973b4a36.  Classification-only softmax MSE
is replaced by MSE between scalar standardized-age predictions; cross entropy is
replaced by the benchmark's standardized-target MSE.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT=Path('/nobackup/enzez/Representation-Anchored-Pseudo-Labeling')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
UPSTREAM_PYTORCH=ROOT/'third_party'/'mean-teacher'/'pytorch'
if str(UPSTREAM_PYTORCH) not in sys.path: sys.path.insert(0,str(UPSTREAM_PYTORCH))
from mean_teacher.data import TwoStreamBatchSampler
from baselines.benchmark_io import write_history, write_json, write_npz
from baselines.utkface_data import BenchmarkContext
from data_processing.utkface_protocol import (build_evaluation_transform,build_labeled_transform,
 build_weak_transform,dataloader_generator,inverse_normalize_age,runtime_metadata,
 seed_dataloader_worker)
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor

UPSTREAM_URL='https://github.com/CuriousAI/mean-teacher.git'
UPSTREAM_SHA='546348ff863c998c26be4339021425df973b4a36'
RESNET_ID='torchvision.ResNet50_Weights.IMAGENET1K_V1 (resnet50-0676ba61.pth)'
DINO_ID='DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)'

class PairDataset(Dataset):
 def __init__(self,ctx): self.ctx=ctx; self.ids=list(ctx.indices('labeled'))+list(ctx.indices('unlabeled')); self.lab=set(ctx.indices('labeled')); self.a=build_labeled_transform(); self.b=build_weak_transform()
 def __len__(self): return len(self.ids)
 def __getitem__(self,pos):
  i=self.ids[pos]; rec=self.ctx.record(i)
  with Image.open(self.ctx.path(i)) as im:
   im=im.convert('RGB'); x1=self.a(im); x2=self.b(im)
  y=(float(rec['age'])-self.ctx.mean)/self.ctx.std if i in self.lab else float('nan')
  return x1,x2,np.float32(y)
class EvalDataset(Dataset):
 def __init__(self,ctx,split): self.ctx=ctx; self.ids=list(ctx.indices(split)); self.t=build_evaluation_transform()
 def __len__(self): return len(self.ids)
 def __getitem__(self,p):
  i=self.ids[p]
  with Image.open(self.ctx.path(i)) as im: x=self.t(im.convert('RGB'))
  return x,np.float32(self.ctx.record(i)['age']),np.int64(i)

def sigmoid_rampup(current,length):
 if length==0:return 1.0
 phase=1.0-np.clip(current,0.0,length)/length; return float(np.exp(-5.0*phase*phase))
def linear_rampup(current,length): return 1.0 if current>=length else current/length
def cosine_rampdown(current,length): return float(.5*(np.cos(np.pi*current/length)+1))
@torch.no_grad()
def update_ema_variables(model,teacher,alpha,global_step):
 # Verbatim upstream mathematics: use true average until EMA is more correct.
 alpha=min(1-1/(global_step+1),alpha)
 for ep,p in zip(teacher.parameters(),model.parameters()): ep.data.mul_(alpha).add_(p.data,alpha=1-alpha)

def loader(ds,batch,shuffle,seed,role,workers):
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,drop_last=False,generator=dataloader_generator(seed,role),worker_init_fn=seed_dataloader_worker)
@torch.no_grad()
def evaluate(model,dl,device,mean,std):
 model.eval(); ps=[]; ys=[]; ids=[]
 for x,y,i in dl:
  standardized=model(x.to(device)).cpu().numpy()
  ps.append(inverse_normalize_age(standardized,mean,std)); ys.append(y.numpy()); ids.append(i.numpy())
 p=np.concatenate(ps); y=np.concatenate(ys); idx=np.concatenate(ids); mae=float(np.mean(np.abs(p-y))); den=float(np.sum((y-y.mean())**2)); r2=float(1-np.sum((y-p)**2)/den)
 return mae,r2,p,y,idx
def file_sha(path):
 h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--backbone',choices=('resnet50','dinov2'),required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--ratio',type=float,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--workers',type=int,default=4); p.add_argument('--consistency',type=float,default=10); p.add_argument('--consistency-rampup',type=float,default=5); p.add_argument('--ema-decay',type=float,default=.9997); p.add_argument('--train-validation-only',action='store_true'); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.benchmark=True
 if not torch.cuda.is_available(): raise RuntimeError('formal Mean Teacher requires CUDA')
 device=torch.device('cuda:0'); out=a.output_dir; out.mkdir(parents=True,exist_ok=True); started=time.time(); torch.cuda.reset_peak_memory_stats()
 ctx=BenchmarkContext(ROOT,a.data_root,a.manifest); assert ctx.manifest['seed']==a.seed and math.isclose(ctx.manifest['labeled_ratio'],a.ratio)
 train_ds=PairDataset(ctx); nl=len(ctx.indices('labeled')); sampler=TwoStreamBatchSampler(range(nl,len(train_ds)),range(nl),a.batch_size,a.batch_size//2); train=DataLoader(train_ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=True,worker_init_fn=seed_dataloader_worker); val=loader(EvalDataset(ctx,'validation'),a.batch_size,False,a.seed,'validation',a.workers); test=None if a.train_validation_only else loader(EvalDataset(ctx,'test'),a.batch_size,False,a.seed,'test',a.workers)
 student=(ResNet50Regressor(True) if a.backbone=='resnet50' else DINOv2Regressor('small')).to(device); teacher=copy.deepcopy(student).to(device)
 for q in teacher.parameters(): q.detach_()
 assert type(student) is type(teacher) and not any(q.requires_grad for q in teacher.parameters())
 ema_reference=next(teacher.parameters()).detach().clone(); ema_update_observed=False
 if a.backbone=='resnet50':
  optimizer=torch.optim.Adam(student.parameters(),lr=1e-4,weight_decay=1e-3);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.1)
  optimization={'optimizer':'Adam','learning_rates':{'all_parameters':1e-4},'weight_decay':1e-3,'betas':[.9,.999],'scheduler':'StepLR','step_size_epochs':10,'gamma':.1,'warmup':None,'amp':False}
 else:
  optimizer=torch.optim.AdamW([{'params':student.backbone.parameters(),'lr':1e-5},{'params':student.head.parameters(),'lr':1e-4}],weight_decay=.05);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
  optimization={'optimizer':'AdamW','learning_rates':{'backbone':1e-5,'head':1e-4},'weight_decay':.05,'betas':[.9,.999],'scheduler':'CosineAnnealingLR','t_max_epochs':a.epochs,'warmup':None,'amp':False}
 architecture=RESNET_ID if a.backbone=='resnet50' else DINO_ID
 config={'dataset':'UTKFace','method':'Mean Teacher','paper_provenance':"Mean Teacher adapted from the authors' official PyTorch implementation.",'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'epochs':a.epochs,'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'supervised_objective':'standardized-target MSE','consistency_objective':'scalar standardized-prediction MSE','optimization':optimization,'optimizer_parameter_owner':'student only','teacher_optimizer':None,'student_architecture':architecture,'teacher_architecture':f'EMA {architecture}','student_teacher_same_backbone':True,'teacher_requires_grad':False,'initial_lr':0.0,'legacy_mean_teacher_lr_ramp_replaced_by_canonical_backbone_protocol':True,'ema_decay':a.ema_decay,'consistency':a.consistency,'consistency_rampup':a.consistency_rampup,'upstream_repo_url':UPSTREAM_URL,'upstream_commit_sha':UPSTREAM_SHA,'upstream_sources':['pytorch/main.py','pytorch/mean_teacher/losses.py','pytorch/mean_teacher/ramps.py','pytorch/mean_teacher/data.py'],'upstream_license':'CC BY-NC 4.0','pretrained_checkpoint_identifier':architecture,'protocol':ctx.protocol_metadata(),'manifest_sha256':file_sha(a.manifest),'selection_metric':'validation MAE only','train_validation_only':a.train_validation_only}; write_json(out/'config.json',config)
 best=float('inf'); best_epoch=None; hist=[]; step=0
 for epoch in range(a.epochs):
  student.train(); teacher.train(); sums=np.zeros(3); n=0
  for j,(xs,xt,y) in enumerate(train):
   lr=optimizer.param_groups[0]['lr']
   xs=xs.to(device); xt=xt.to(device); y=y.to(device); mask=torch.isfinite(y)
   with torch.no_grad(): tp=teacher(xt)
   sp=student(xs)
   sup=F.mse_loss(sp[mask],y[mask]) if mask.any() else sp.sum()*0; cw=a.consistency*sigmoid_rampup(epoch,a.consistency_rampup); cons=F.mse_loss(sp,tp); loss=sup+cw*cons
   if not all(torch.isfinite(v) for v in (sup,cons,loss)): raise RuntimeError(f'non-finite training loss: supervised={sup.item()} consistency={cons.item()} total={loss.item()}')
   optimizer.zero_grad(); loss.backward()
   if not all(q.grad is None or torch.isfinite(q.grad).all() for q in student.parameters()): raise RuntimeError('non-finite student gradient')
   optimizer.step()
   if not all(torch.isfinite(q).all() for q in student.parameters()): raise RuntimeError('non-finite student parameter after optimizer step')
   step+=1; update_ema_variables(student,teacher,a.ema_decay,step)
   if not ema_update_observed: ema_update_observed=not torch.equal(ema_reference,next(teacher.parameters()).detach())
   sums+=np.array([loss.item(),sup.item(),cons.item()]); n+=1
  vm,vr,_,_,_=evaluate(teacher,val,device,ctx.mean,ctx.std); row={'epoch':epoch+1,'loss_total':sums[0]/n,'loss_supervised':sums[1]/n,'loss_consistency_unweighted':sums[2]/n,'consistency_weight':cw,'learning_rate':lr,'validation_mae':vm,'validation_r2':vr}; hist.append(row);scheduler.step()
  print(f"epoch={epoch+1}/{a.epochs} loss={row['loss_total']:.8f} val_mae={vm:.6f}",flush=True)
  if vm<best:
   best=vm; best_epoch=epoch+1; torch.save({'checkpoint_version':'utkface-mean-teacher-v1','teacher':teacher.state_dict(),'student':student.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':best_epoch,'validation_mae':best,'manifest_sha256':config['manifest_sha256'],'cohort_sha256':ctx.cohort['cohort_sha256'],'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'upstream_commit_sha':UPSTREAM_SHA},out/'best.pt')
 write_history(out/'history.csv',hist); ck=torch.load(out/'best.pt',map_location=device,weights_only=False); assert ck['epoch']==best_epoch and ck['validation_mae']==best and ck['manifest_sha256']==config['manifest_sha256'] and ck['cohort_sha256']==ctx.cohort['cohort_sha256'] and ck['backbone']==a.backbone and ck['seed']==a.seed and ck['ratio']==a.ratio; teacher.load_state_dict(ck['teacher'])
 if a.train_validation_only:
  assert ema_update_observed and all(math.isfinite(float(row[k])) for row in hist for k in ('loss_total','loss_supervised','loss_consistency_unweighted','validation_mae','validation_r2'))
  write_json(out/'metrics.json',{'validation_mae':best,'validation_r2_at_last_epoch':hist[-1]['validation_r2']});write_json(out/'metadata.json',{'status':'diagnostic_complete','dataset':'UTKFace','method':'Mean Teacher','backbone':a.backbone,'labeled_ratio':a.ratio,'seed':a.seed,**ctx.protocol_metadata(),'manifest_sha256':config['manifest_sha256'],'train_validation_only':True,'test_evaluations':0,'checkpoint_reloaded':True,'checkpoint_metadata_verified':True,'ema_update_observed':True,'teacher_forward_no_grad':True,'teacher_requires_grad':False,'teacher_optimizer':None,'student_teacher_same_backbone':True,'finite_supervised_consistency_total_losses':True,'finite_gradients':True,'finite_student_teacher_outputs':True,'best_epoch':best_epoch,'best_validation_mae':best,'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(),'wall_clock_seconds':time.time()-started,'runtime':runtime_metadata()});return
 tm,tr,pred,target,idx=evaluate(teacher,test,device,ctx.mean,ctx.std); expected=np.asarray(ctx.indices('test')); assert np.array_equal(idx,expected) and len(pred)==ctx.manifest['counts']['test'] and np.isfinite(pred).all() and np.isfinite([tm,tr]).all()
 write_npz(out/'test_predictions.npz',predictions=pred,targets=target,cohort_indices=idx); write_npz(out/'analysis_snapshot.npz',test_predictions=pred,test_targets=target,test_cohort_indices=idx)
 assert ema_update_observed
 write_json(out/'metrics.json',{'validation_mae':best,'test_mae':tm,'test_r2':tr}); write_json(out/'metadata.json',{'status':'complete','dataset':'UTKFace','method':'Mean Teacher','backbone':a.backbone,'labeled_ratio':a.ratio,'seed':a.seed,**ctx.protocol_metadata(),'manifest_sha256':config['manifest_sha256'],'upstream_repo_url':UPSTREAM_URL,'upstream_commit_sha':UPSTREAM_SHA,'pretrained_checkpoint_identifier':config['pretrained_checkpoint_identifier'],'student_architecture':architecture,'teacher_architecture':f'EMA {architecture}','student_teacher_same_backbone':True,'teacher_requires_grad':False,'teacher_forward_no_grad':True,'teacher_optimizer':None,'ema_update_observed':ema_update_observed,'finite_supervised_consistency_total_losses':True,'best_epoch':best_epoch,'best_validation_mae':best,'checkpoint_reloaded':True,'checkpoint_metadata_verified':True,'test_used_for_selection':False,'test_evaluations':1,'prediction_count':len(pred),'predictions_in_manifest_order':True,'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(),'wall_clock_seconds':time.time()-started,'runtime':runtime_metadata()})
if __name__=='__main__': main()
