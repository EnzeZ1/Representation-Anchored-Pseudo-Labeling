"""Executable protected-data CheXchoNet LVIDd formal trainer."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,time
from itertools import cycle
from pathlib import Path
import higher
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr,spearmanr
from torch import nn
from torch.utils.data import DataLoader
from data_processing.chexchonet import CheXchoNetDataset,discover_metadata,load_records
from data_processing.chexchonet_protocol import PROTOCOL_VERSION,TRANSFORM_VERSION,build_evaluation_transform,build_strong_transform,build_weak_transform,validate_manifest
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from models.hpl_uncertainty import UncertaintyLearner

HPL_UPSTREAM_COMMIT="89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20"
RESNET_ID="torchvision.models.ResNet50_Weights.IMAGENET1K_V1"
DINO_ID="dinov2_vits14"

def write_json(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");tmp.replace(path)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def state_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def construct(backbone,device):
 model=ResNet50Regressor(pretrained=True) if backbone=='resnet50' else DINOv2Regressor(size='small')
 return model.to(device)
def feature(model,x): return model.encode(x)
def loader(ds,batch,seed,role,workers,shuffle):
 g=torch.Generator();g.manual_seed(seed*1009+{'labeled':11,'unlabeled':23,'validation':37,'test':53}[role])
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,drop_last=False,generator=g,worker_init_fn=lambda wid:np.random.seed((torch.initial_seed()+wid)%2**32))
def load_data(args):
 records=load_records(discover_metadata(args.data_root));manifest=json.loads(args.manifest.read_text());validate_manifest(manifest,records)
 if manifest['seed']!=args.seed or not math.isclose(manifest['labeled_ratio'],args.ratio):raise RuntimeError('Manifest assignment mismatch')
 mean,std=manifest['label_scaler']['mean'],manifest['label_scaler']['std'];root=args.data_root/'images'
 weak,strong,evaluation=build_weak_transform(),build_strong_transform(),build_evaluation_transform()
 datasets={'labeled':CheXchoNetDataset(records,root,manifest['indices']['labeled'],mean,std,weak),'unlabeled':CheXchoNetDataset(records,root,manifest['indices']['unlabeled'],mean,std,weak,strong),'validation':CheXchoNetDataset(records,root,manifest['indices']['validation'],mean,std,evaluation),'test':CheXchoNetDataset(records,root,manifest['indices']['test'],mean,std,evaluation)}
 loaders={k:loader(v,args.batch_size,args.seed,k,args.num_workers,k in ('labeled','unlabeled')) for k,v in datasets.items()}
 return records,manifest,datasets,loaders,float(mean),float(std)
def optimize(model,backbone,method,epochs):
 if backbone=='dinov2_vits14':
  opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs);cfg={'optimizer':'AdamW','backbone_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'scheduler':'CosineAnnealingLR','T_max':epochs}
 elif method=='supervised_standard':
  opt=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':1e-4},{'params':model.head.parameters(),'lr':1e-3}],weight_decay=0);sch=torch.optim.lr_scheduler.StepLR(opt,10,.1);cfg={'optimizer':'Adam','backbone_lr':1e-4,'head_lr':1e-3,'weight_decay':0,'scheduler':'StepLR','step_size':10,'gamma':.1}
 else:
  opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3);sch=torch.optim.lr_scheduler.StepLR(opt,10,.1);cfg={'optimizer':'Adam','lr':1e-4,'weight_decay':1e-3,'scheduler':'StepLR','step_size':10,'gamma':.1}
 return opt,sch,cfg
@torch.no_grad()
def evaluate(model,loader_,mean,std,device,predictions=False):
 model.eval();ps=[];ys=[];ids=[]
 for x,y,index in loader_:
  ps.append((model(x.to(device)).float().cpu()*std+mean).numpy());ys.append((y.float()*std+mean).numpy());ids.append(index.numpy())
 p=np.concatenate(ps);y=np.concatenate(ys);idx=np.concatenate(ids);err=p-y
 out={'mae_cm':float(np.mean(np.abs(err))),'mae_mm':float(np.mean(np.abs(err))*10),'mse_cm2':float(np.mean(err**2)),'rmse_cm':float(np.sqrt(np.mean(err**2))),'r2':float(1-np.sum(err**2)/(np.sum((y-y.mean())**2)+1e-12)),'pearson':float(pearsonr(p,y).statistic),'spearman':float(spearmanr(p,y).statistic)}
 if not np.isfinite(list(out.values())).all():raise RuntimeError('Non-finite metric')
 return (out,p,y,idx) if predictions else out
@torch.no_grad()
def fit_probe(anchor,labeled,device):
 anchor.eval();xs=[];ys=[]
 for x,y,_ in labeled:xs.append(feature(anchor,x.to(device)).double().cpu());ys.append(y.double())
 x=torch.cat(xs);y=torch.cat(ys);xm=x.mean(0);ym=y.mean();xc=x-xm;yc=y-ym
 # Ridge-stabilized closed-form affine least squares, fixed a priori.
 gram=xc.T@xc;w=torch.linalg.solve(gram+1e-6*torch.eye(gram.shape[0],dtype=gram.dtype),xc.T@yc);b=ym-xm@w
 probe=nn.Linear(x.shape[1],1);probe.weight.copy_(w.float()[None]);probe.bias.copy_(b.float()[None]);probe.requires_grad_(False);return probe.to(device).eval()
def hpl_meta(model,uncertainty,head_optimizer,unc_optimizer,labeled,weak,strong,meta,device):
 xl,yl,_=labeled;xm,ym,_=meta;xl,yl,xm,ym=xl.to(device),yl.to(device),xm.to(device),ym.to(device);weak,strong=weak.to(device),strong.to(device)
 unc_optimizer.zero_grad(set_to_none=True)
 with higher.innerloop_ctx(model.head,head_optimizer) as (fhead,diffopt):
  with torch.no_grad():fl=feature(model,xl);fw=feature(model,weak);fs=feature(model,strong);fm=feature(model,xm)
  pl=fhead(fl).squeeze(-1);pw=fhead(fw).squeeze(-1).detach();ps=fhead(fs).squeeze(-1)
  ui=torch.stack((ps.detach()-pw,ps.detach()),-1);diffopt.step(F.mse_loss(pl,yl)+(torch.exp(-uncertainty(ui))/2*(ps-pw).pow(2).unsqueeze(-1)).mean())
  pm=fhead(fm).squeeze(-1);umi=torch.stack(((pm-ym).detach(),pm.detach()),-1);loss=F.mse_loss(pm,ym)-uncertainty(umi).mean();loss.backward();unc_optimizer.step()
def run(args):
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Worker must see one CUDA device as cuda:0')
 seed_all(args.seed);device=torch.device('cuda:0');records,manifest,datasets,loaders,mean,std=load_data(args)
 model=construct(args.backbone,device);target_hash=state_hash(model);anchor=probe=uncertainty=None
 if args.method=='rapl':
  anchor=construct(args.backbone,device);anchor.requires_grad_(False);anchor.eval();probe=fit_probe(anchor,loaders['labeled'],device)
  if any(p.requires_grad for p in anchor.parameters()) or any(p.requires_grad for p in probe.parameters()):raise RuntimeError('RAPL frozen-state failure')
 if args.method=='hpl': uncertainty=UncertaintyLearner().to(device)
 optimizer,scheduler,optcfg=optimize(model,args.backbone,args.method,args.epochs)
 uncopt=torch.optim.Adam(uncertainty.parameters(),lr=1e-4,weight_decay=1e-5) if uncertainty else None
 metaopt=torch.optim.Adam(model.head.parameters(),lr=1e-3 if args.backbone=='resnet50' else 1e-4) if uncertainty else None
 config={'dataset':'CheXchoNet','target':'LVIDd','target_unit':'centimeters','method':args.method,'backbone':args.backbone,'objective':'standardized-target MSE','epochs':args.epochs,'batch_size':args.batch_size,'precision':'float32','selection_metric':'lowest validation MAE in original centimeters','optimization':optcfg,'lambda_u':1.0 if args.method!='supervised_standard' else None,'tau':1.0 if args.method=='rapl' else None,'trust_formula':'1/(1+abs(target_pseudo-frozen_probe))' if args.method=='rapl' else None,'hpl':{'upstream_commit':HPL_UPSTREAM_COMMIT,'uncertainty_learner':'MLP(2,128,1)','uncertainty_lr':1e-4,'uncertainty_weight_decay':1e-5,'meta_update_frequency':5,'w_ulb':1.0,'lambda2':1.0} if args.method=='hpl' else None,'pretrained_identifier':RESNET_ID if args.backbone=='resnet50' else DINO_ID,'target_initialization_hash':target_hash,'protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'manifest_sha256':manifest['manifest_sha256'],'cohort_digest':manifest['cohort_digest'],'counts':manifest['counts'],'label_scaler':manifest['label_scaler'],'supervised_condition':'Supervised-Standard' if args.method=='supervised_standard' else None}
 if args.preflight:
  batch=next(iter(loaders['labeled']));x,y,_=batch;x,y=x.to(device),y.to(device);optimizer.zero_grad();loss=F.mse_loss(model(x),y);loss.backward();optimizer.step()
  if args.method!='supervised_standard':
   w,s,_=next(iter(loaders['unlabeled']));w,s=w.to(device),s.to(device);pseudo=model(w).detach();strong=model(s)
   if args.method=='rapl':
    with torch.no_grad():q=probe(feature(anchor,w));u=((1/(1+(pseudo-q.squeeze(-1)).abs())).detach()*(strong-pseudo).pow(2)).mean()
   else:
    u=(torch.exp(-uncertainty(torch.stack((strong.detach()-pseudo,strong.detach()),-1)))/2*(strong-pseudo).pow(2).unsqueeze(-1)).mean()
    hpl_meta(model,uncertainty,metaopt,uncopt,batch,w,s,next(iter(loaders['labeled'])),device)
   if not torch.isfinite(u):raise RuntimeError('Preflight SSL loss nonfinite')
  validation=evaluate(model,loaders['validation'],mean,std,device)
  print(json.dumps({'status':'pass','method':args.method,'backbone':args.backbone,'validation_mae_cm':validation['mae_cm'],'test_model_inference_count':0,'target_init_hash':target_hash,'anchor_frozen':None if anchor is None else not any(p.requires_grad for p in anchor.parameters()),'probe_labeled_images':None if probe is None else len(datasets['labeled']),'hpl_uncertainty_trainable':None if uncertainty is None else any(p.requires_grad for p in uncertainty.parameters()),'hpl_bilevel_step_executed':args.method=='hpl','config':config},sort_keys=True));return
 out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise FileExistsError('Refusing to overwrite completed output')
 write_json(out/'config.json',config);started=time.time();history=[];best=math.inf;best_epoch=None;steps=0;labeled_exposure=0
 for epoch in range(1,args.epochs+1):
  model.train();losses=[];sup_losses=[];unl_losses=[]
  if args.method=='supervised_standard': iterator=((b,None) for b in loaders['labeled'])
  else: iterator=zip(cycle(loaders['labeled']),loaders['unlabeled'])
  meta_iter=cycle(loaders['labeled'])
  for step,(lab,unlab) in enumerate(iterator):
   x,y,_=lab;x,y=x.to(device),y.to(device);labeled_exposure+=len(y)
   if unlab is not None:
    weak,strong,_=unlab
    if args.method=='hpl' and step%5==0:hpl_meta(model,uncertainty,metaopt,uncopt,lab,weak,strong,next(meta_iter),device)
    weak,strong=weak.to(device),strong.to(device)
   optimizer.zero_grad(set_to_none=True);sup=F.mse_loss(model(x),y);unl=torch.tensor(0.,device=device)
   if args.method!='supervised_standard':
    pseudo=model(weak).detach();strong_pred=model(strong)
    if args.method=='rapl':
     with torch.no_grad():anchor_pred=probe(feature(anchor,weak)).squeeze(-1);trust=(1/(1+(pseudo-anchor_pred).abs())).detach()
     unl=(trust*(strong_pred-pseudo).pow(2)).mean()
    else:
     inp=torch.stack(((strong_pred-pseudo).detach(),strong_pred.detach()),-1);weight=(torch.exp(-uncertainty(inp))/2).detach();unl=(weight*(strong_pred-pseudo).pow(2).unsqueeze(-1)).mean()
   loss=sup+unl;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();steps+=1;losses.append(float(loss));sup_losses.append(float(sup));unl_losses.append(float(unl))
  scheduler.step();val=evaluate(model,loaders['validation'],mean,std,device);improved=val['mae_cm']<best
  if improved:
   best=val['mae_cm'];best_epoch=epoch;torch.save({'model_state':model.state_dict(),'anchor_state':anchor.state_dict() if anchor else None,'probe_state':probe.state_dict() if probe else None,'uncertainty_state':uncertainty.state_dict() if uncertainty else None,'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict(),'epoch':epoch,'validation_mae_cm':best,'manifest_sha256':manifest['manifest_sha256'],'scaler':manifest['label_scaler'],'config':config},out/'best.pt')
  history.append({'epoch':epoch,'train_total_mse':np.mean(losses),'train_sup_mse':np.mean(sup_losses),'train_unlabeled_mse':np.mean(unl_losses),'validation_mae_cm':val['mae_cm'],'validation_mse_cm2':val['mse_cm2'],'validation_r2':val['r2'],'best_so_far':int(improved),'optimizer_steps':steps,'labeled_example_exposure':labeled_exposure,'elapsed_seconds':time.time()-started})
  print(f"epoch={epoch}/{args.epochs} val_mae_cm={val['mae_cm']:.6f} best_epoch={best_epoch}",flush=True)
 with (out/'history.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=history[0]);w.writeheader();w.writerows(history)
 restored=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(restored['model_state'])
 if restored['epoch']!=best_epoch or not math.isclose(restored['validation_mae_cm'],best,abs_tol=1e-12):raise RuntimeError('Checkpoint metadata mismatch')
 test,pred,target,indices=evaluate(model,loaders['test'],mean,std,device,True);np.savez_compressed(out/'test_predictions.npz',cohort_indices=indices,predictions_cm=pred,targets_cm=target)
 metrics={'best_epoch':best_epoch,'best_validation_mae_cm':best,**{f'test_{k}':v for k,v in test.items()},'checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1,'test_time_calibration':False};write_json(out/'metrics.json',metrics)
 write_json(out/'metadata.json',{'status':'complete','dataset':'CheXchoNet','target':'LVIDd','target_unit':'centimeters','method':args.method,'backbone':args.backbone,'seed':args.seed,'labeled_ratio':args.ratio,'checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1,'test_time_calibration':False,'restored_epoch_verified':True,'restored_validation_metric_verified':True,'ordered_prediction_alignment':indices.tolist()==manifest['indices']['test'],'manifest_sha256':manifest['manifest_sha256'],'cohort_digest':manifest['cohort_digest'],'split_digest':manifest['split_digest'],'counts':manifest['counts'],'label_scaler':manifest['label_scaler'],'target_initialization_hash':target_hash,'optimizer_steps':steps,'labeled_example_exposure_count':labeled_exposure,'runtime_seconds':time.time()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'process_local_device':'cuda:0'})
 print(json.dumps(metrics,sort_keys=True),flush=True)
def arguments():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=('supervised_standard','rapl','hpl'),required=True);p.add_argument('--backbone',choices=('resnet50','dinov2_vits14'),required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path);p.add_argument('--seed',type=int,required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--num-workers',type=int,default=4);p.add_argument('--preflight',action='store_true');return p.parse_args()
if __name__=='__main__':run(arguments())
