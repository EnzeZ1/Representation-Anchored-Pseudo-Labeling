"""Unified ratio-first KonIQ-10k supervised-step-matched, RAPL, and HPL runner."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,time
from itertools import cycle
from pathlib import Path
import higher,numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr,spearmanr
from torch import nn
from torch.utils.data import DataLoader,Dataset
from data_processing.koniq10k import *
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from models.hpl_uncertainty import UncertaintyLearner

ROOT=Path(__file__).resolve().parents[1]
COHORT_PATH=ROOT/'artifacts/koniq10k_protocol/koniq10k_cohort_v1.json'
UPSTREAM_HPL='89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20'

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def state_hash(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def write_json(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)

class Images(Dataset):
 def __init__(self,cohort,indices,root,transform,mean,std):self.c=cohort;self.i=list(indices);self.root=Path(root);self.t=transform;self.mean=mean;self.std=std
 def __len__(self):return len(self.i)
 def __getitem__(self,k):
  i=self.i[k];r=self.c['records'][i]
  with Image.open(self.root/r['image_name']) as im:x=self.t(im.convert('RGB'))
  return x,torch.tensor((r['mos']-self.mean)/self.std,dtype=torch.float32),i,r['image_name']
class Views(Dataset):
 def __init__(self,cohort,indices,root,mean,std):self.w=Images(cohort,indices,root,build_transform(False),mean,std);self.s=Images(cohort,indices,root,build_transform(True),mean,std)
 def __len__(self):return len(self.w)
 def __getitem__(self,k):
  w=self.w[k];s=self.s[k];return w[0],s[0],w[2],w[3]

def construct(backbone,device):
 if backbone=='resnet50':return ResNet50Regressor(pretrained=True).to(device),{'identifier':'torchvision.ResNet50_Weights.IMAGENET1K_V1','feature_dimension':2048}
 m=DINOv2Regressor(size='small').to(device);return m,{'identifier':m.weight_identifier,'url':m.weight_url,'feature_dimension':384}
def optimize(model,backbone,epochs,steps):
 if backbone=='resnet50':
  o=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':1e-4},{'params':model.head.parameters(),'lr':1e-3}]);s=torch.optim.lr_scheduler.StepLR(o,10,.1);cfg={'optimizer':'Adam','encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':0.,'scheduler':'StepLR(10,0.1)'}
 else:
  o=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);cfg={'optimizer':'AdamW','encoder_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'scheduler':f'CosineAnnealingLR({epochs})'}
 return o,s,cfg
def loaders(args):
 c=load_cohort(COHORT_PATH);m=json.loads(args.manifest.read_text());validate_manifest(m,c)
 if m['seed']!=args.seed or not math.isclose(m['labeled_ratio'],args.ratio):raise ValueError('manifest identity mismatch')
 mean,std=m['label_scaler']['mean'],m['label_scaler']['std']; common=(c,args.data_root,mean,std)
 ds={'labeled':Images(common[0],m['labeled_indices'],common[1],build_transform(False),mean,std),'validation':Images(common[0],m['splits']['validation'],common[1],build_transform(False),mean,std),'test':Images(common[0],m['splits']['test'],common[1],build_transform(False),mean,std)}
 if m['unlabeled_indices']:ds['unlabeled']=Views(c,m['unlabeled_indices'],args.data_root,mean,std)
 out={}
 for role,d in ds.items():
  g=torch.Generator().manual_seed(args.seed+{'labeled':11000,'unlabeled':22000,'validation':33000,'test':44000}[role]);out[role]=DataLoader(d,batch_size=args.batch_size,shuffle=role in ('labeled','unlabeled'),num_workers=args.num_workers,pin_memory=True,drop_last=False,generator=g)
 protocol={'protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'cohort_sha256':c['cohort_sha256'],'membership_sha256':c['membership_sha256'],'manifest_path':str(args.manifest.resolve()),'manifest_file_sha256':sha256(args.manifest),'manifest_payload_sha256':m['manifest_sha256'],'counts':m['counts'],'label_scaler':m['label_scaler']}
 return c,m,out,mean,std,protocol
@torch.no_grad()
def evaluate(model,loader,mean,std,device,preds=False):
 model.eval();ps=[];ys=[];ids=[];names=[]
 for x,y,i,n in loader:
  ps.append(inverse_target(model(x.to(device)).float().cpu(),mean,std));ys.append(inverse_target(y,mean,std));ids.append(i);names.extend(n)
 p=torch.cat(ps).numpy();y=torch.cat(ys).numpy();idx=torch.cat(ids).numpy();mse=float(np.mean((p-y)**2));mae=float(np.mean(abs(p-y)));r2=float(1-np.sum((p-y)**2)/(np.sum((y-y.mean())**2)+1e-12));plcc=float(pearsonr(p,y).statistic);srocc=float(spearmanr(p,y).statistic)
 if not np.isfinite([mse,mae,r2,plcc,srocc]).all():raise RuntimeError('nonfinite metrics')
 z=(mse,mae,r2,plcc,srocc);return (*z,p,y,idx,np.asarray(names)) if preds else z
@torch.no_grad()
def fit_probe(anchor,loader,device):
 anchor.eval();xs=[];ys=[]
 for x,y,_,_ in loader:xs.append(anchor.encode(x.to(device)).double().cpu());ys.append(y.double())
 x=torch.cat(xs);y=torch.cat(ys);xm=x.mean(0);ym=y.mean();xc=x-xm;yc=y-ym;alpha=torch.linalg.lstsq(xc@xc.T,yc[:,None]).solution[:,0];w=xc.T@alpha;b=ym-xm@w;p=nn.Linear(x.shape[1],1);p.weight.copy_(w.float()[None]);p.bias.copy_(b.float()[None]);p.requires_grad_(False);return p.to(device).eval()

def hpl_meta(model,unc,opt,head_opt,labeled,weak,strong,meta):
 opt.zero_grad(set_to_none=True)
 with higher.innerloop_ctx(model.head,head_opt) as (fh,diff):
  with torch.no_grad():fl=model.encode(labeled[0]);fw=model.encode(weak);fs=model.encode(strong);fm=model.encode(meta[0])
  pl=fh(model.drop(fl)).squeeze(-1);pw=fh(model.drop(fw)).squeeze(-1).detach();ps=fh(model.drop(fs)).squeeze(-1);u=torch.stack((ps.detach()-pw,ps.detach()),-1);diff.step(F.mse_loss(pl,labeled[1])+torch.mean(torch.exp(-unc(u))/2*(ps-pw).pow(2).unsqueeze(-1)));pm=fh(model.drop(fm)).squeeze(-1);um=torch.stack(((pm-meta[1]).detach(),pm.detach()),-1);loss=F.mse_loss(pm,meta[1])-unc(um).mean();loss.backward();opt.step()
 return float(loss.detach())

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=('supervised_step_matched','supervised_standard','rapl','hpl'),required=True);p.add_argument('--backbone',choices=('resnet50','dinov2_vits14'),required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('must see exactly one local cuda:0')
 seed_all(a.seed);device=torch.device('cuda:0');model,identity=construct(a.backbone,device);target_hash=state_hash(model);target_backbone_hash=state_hash(model.backbone);c,m,ls,mean,std,protocol=loaders(a)
 anchor=probe=unc=None;anchor_hash=None;anchor_backbone_hash=None
 if a.method=='rapl':
  anchor,_=construct(a.backbone,device);anchor.requires_grad_(False);anchor.eval();anchor_hash=state_hash(anchor);anchor_backbone_hash=state_hash(anchor.backbone)
  if anchor_backbone_hash!=target_backbone_hash:raise RuntimeError('target/anchor pretrained backbone mismatch')
  probe=fit_probe(anchor,ls['labeled'],device)
 if a.method=='hpl':unc=UncertaintyLearner().to(device)
 # Low-label horizontal methods make one target update per unlabeled batch.
 steps=len(ls['unlabeled']) if 'unlabeled' in ls and a.method!='supervised_standard' else len(ls['labeled'])
 opt,sched,optcfg=optimize(model,a.backbone,a.epochs,steps);uncopt=torch.optim.Adam(unc.parameters(),lr=1e-4,weight_decay=1e-5) if unc else None;headopt=torch.optim.Adam(model.head.parameters(),lr=1e-3 if a.backbone=='resnet50' else 1e-4) if unc else None
 cfg={'dataset':'KonIQ-10k','method':a.method,'supervised_control':'step_matched' if a.method=='supervised_step_matched' else None,'unlabeled_gradient_contribution':False if a.method.startswith('supervised') else True,'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'epochs':a.epochs,'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'gradient_accumulation':1,'target_optimizer_steps':steps*a.epochs,'labeled_example_exposures':steps*a.batch_size*a.epochs,'selection':'lowest validation MSE in original MOS units','test_time_calibration':False,'target_init_hash':target_hash,'target_backbone_init_hash':target_backbone_hash,'anchor_init_hash':anchor_hash,'anchor_backbone_init_hash':anchor_backbone_hash,'model':identity,'optimization':optcfg,'protocol':protocol,'method_configuration':{'lambda_u':1.0,'tau':1.0,'trust':'1/(1+abs(target_pseudo-frozen_probe))','probe_fit':'labeled only'} if a.method=='rapl' else {'w_ulb':1.0,'lambda2':1.0,'uncertainty_update_frequency':5,'official_hpl_upstream_commit':UPSTREAM_HPL} if a.method=='hpl' else {'lambda_unlabeled':0.0}}
 if a.preflight:
  batch=next(iter(ls['labeled']));x=batch[0].to(device);y=batch[1].to(device);opt.zero_grad();sup=F.mse_loss(model(x),y);loss=sup;meta_active=False
  if a.method in ('rapl','hpl'):
   view=next(iter(ls['unlabeled']));weak,strong=view[0].to(device),view[1].to(device)
   if a.method=='hpl':hpl_meta(model,unc,uncopt,headopt,(x,y),weak,strong,(x,y));meta_active=True
   with torch.no_grad():pseudo=model(weak).detach()
   strongp=model(strong)
   if a.method=='rapl':
    with torch.no_grad():ap=probe(anchor.encode(weak)).squeeze(-1);w=(1/(1+abs(pseudo-ap))).detach()
    loss=sup+(w*(strongp-pseudo).pow(2)).mean()
   else:
    with torch.no_grad():w=torch.exp(-unc(torch.stack((strongp.detach()-pseudo,strongp.detach()),-1)))/2
    loss=sup+(w.squeeze(-1)*(strongp-pseudo).pow(2)).mean()
  loss.backward();opt.step();print(json.dumps({'status':'pass','test_model_inference_count':0,'loss':float(loss),'target_init_hash':target_hash,'target_backbone_init_hash':target_backbone_hash,'anchor_backbone_init_hash':anchor_backbone_hash,'target_anchor_backbone_equal':None if anchor is None else target_backbone_hash==anchor_backbone_hash,'anchor_frozen':None if anchor is None else not any(x.requires_grad for x in anchor.parameters()),'probe_frozen':None if probe is None else not any(x.requires_grad for x in probe.parameters()),'uncertainty_trainable':None if unc is None else any(x.requires_grad for x in unc.parameters()),'hpl_meta_update_active':meta_active,'target_optimizer_steps':cfg['target_optimizer_steps'],'labeled_example_exposures':cfg['labeled_example_exposures']}));return
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists():raise FileExistsError(out)
 write_json(out/'config.json',cfg);torch.cuda.reset_peak_memory_stats(device);start=time.time();hist=[];best=math.inf;best_epoch=None
 for epoch in range(1,a.epochs+1):
  model.train();tot=supsum=ulsum=0.;n=0;lcycle=cycle(ls['labeled']);mcycle=cycle(ls['labeled'])
  iterator=ls['unlabeled'] if 'unlabeled' in ls and a.method!='supervised_standard' else range(len(ls['labeled']))
  for step,item in enumerate(iterator):
   lb=next(lcycle);lx,ly=lb[0].to(device),lb[1].to(device);opt.zero_grad(set_to_none=True);sp=F.mse_loss(model(lx),ly);ul=torch.zeros((),device=device)
   if a.method in ('rapl','hpl'):
    weak,strong=item[0].to(device),item[1].to(device)
    if a.method=='hpl' and step%5==0:
     mb=next(mcycle);hpl_meta(model,unc,uncopt,headopt,(lx,ly),weak,strong,(mb[0].to(device),mb[1].to(device)))
    with torch.no_grad():pseudo=model(weak).detach()
    strongp=model(strong)
    if a.method=='rapl':
     with torch.no_grad():ap=probe(anchor.encode(weak)).squeeze(-1);trust=(1/(1+abs(pseudo-ap))).detach()
     ul=(trust*(strongp-pseudo).pow(2)).mean()
    else:
     with torch.no_grad():weight=torch.exp(-unc(torch.stack((strongp.detach()-pseudo,strongp.detach()),-1)))/2
     ul=(weight.squeeze(-1)*(strongp-pseudo).pow(2)).mean()
   loss=sp+ul;loss.backward();opt.step();tot+=float(loss);supsum+=float(sp);ulsum+=float(ul);n+=1
  sched.step();vmse,vmae,vr2,vp,vs=evaluate(model,ls['validation'],mean,std,device);improved=vmse<best
  if improved:
   best=vmse;best_epoch=epoch;torch.save({'version':'koniq10k-formal-v1','model':model.state_dict(),'anchor':anchor.state_dict() if anchor else None,'probe':probe.state_dict() if probe else None,'uncertainty':unc.state_dict() if unc else None,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'validation_mse':vmse,'config':cfg},out/'best.pt')
  hist.append({'epoch':epoch,'train_total_mse':tot/n,'train_sup_mse':supsum/n,'train_unlabeled_mse':ulsum/n,'validation_mse_mos':vmse,'validation_mae_mos':vmae,'validation_r2':vr2,'validation_plcc':vp,'validation_srocc':vs,'best_so_far':int(improved),'target_optimizer_steps_completed':epoch*steps});print(f'epoch={epoch}/{a.epochs} validation_mse={vmse:.6f} best_epoch={best_epoch}',flush=True)
 with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);assert ck['epoch']==best_epoch and math.isclose(ck['validation_mse'],best,abs_tol=1e-12)
 test=evaluate(model,ls['test'],mean,std,device,True);tmse,tmae,tr2,tp,ts,pred,target,idx,names=test;np.savez_compressed(out/'test_predictions.npz',cohort_indices=idx,image_identifiers=names,predictions_mos=pred,targets_mos=target)
 metrics={'best_epoch':best_epoch,'best_validation_mse_mos':best,'test_mse_mos':tmse,'test_mae_mos':tmae,'test_r2':tr2,'test_plcc':tp,'test_srocc':ts,'checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1,'test_time_calibration':False};write_json(out/'metrics.json',metrics);write_json(out/'metadata.json',{'status':'complete',**protocol,'method':a.method,'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'checkpoint_selection':'lowest validation MSE in original MOS units','checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1,'test_time_calibration':False,'target_init_hash':target_hash,'target_backbone_init_hash':target_backbone_hash,'anchor_init_hash':anchor_hash,'anchor_backbone_init_hash':anchor_backbone_hash,'runtime_seconds':time.time()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),'physical_gpu':os.environ.get('KONIQ_PHYSICAL_GPU'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'process_local_device':'cuda:0'});print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
