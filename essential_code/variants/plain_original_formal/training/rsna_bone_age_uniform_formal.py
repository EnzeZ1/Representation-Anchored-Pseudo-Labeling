"""Formal RSNA bone-age canonical Uniform runner."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,random,time
from itertools import cycle
from pathlib import Path
import higher
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader,Dataset
from data_processing.rsna_bone_age import *
from models.backbone import ResNet50Regressor
from models.rsna_dinov2_backbone import DINOv2Regressor
from models.hpl_uncertainty import UncertaintyLearner
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def state_hash(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class Images(Dataset):
 def __init__(self,c,idx,root,mode,mean,std):self.c=c;self.i=list(idx);self.root=Path(root);self.t=build_transform(mode);self.mean=mean;self.std=std
 def __len__(self):return len(self.i)
 def __getitem__(self,k):
  i=self.i[k];r=self.c['records'][i]
  with Image.open(self.root/r['image_path']) as im:x=self.t(im)
  return x,torch.tensor((r['bone_age_months']-self.mean)/self.std,dtype=torch.float32),i,r['image_id'],r['male']
class Views(Dataset):
 def __init__(self,c,idx,root,mean,std):self.w=Images(c,idx,root,'weak',mean,std);self.s=Images(c,idx,root,'strong',mean,std)
 def __len__(self):return len(self.w)
 def __getitem__(self,k):
  w=self.w[k];s=self.s[k];return w[0],s[0],w[2],w[3]
def construct(b,device):
 if b=='resnet50':return ResNet50Regressor(True).to(device),{'identifier':'torchvision.ResNet50_Weights.IMAGENET1K_V1','feature_dimension':2048}
 m=DINOv2Regressor('small').to(device);return m,{'identifier':m.weight_identifier,'url':m.weight_url,'feature_dimension':384}
def optimize(model,b,epochs):
 if b=='resnet50':o=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':1e-4},{'params':model.head.parameters(),'lr':1e-3}]);s=torch.optim.lr_scheduler.StepLR(o,10,.1);cfg={'optimizer':'Adam','encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':0.,'scheduler':'StepLR(10,0.1)','warmup':None,'gradient_clipping':None}
 else:o=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,epochs);cfg={'optimizer':'AdamW','encoder_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'scheduler':f'CosineAnnealingLR({epochs})','warmup':None,'gradient_clipping':None}
 return o,s,cfg
def hpl_meta(model,unc,head_opt,unc_opt,lab,weak,strong,meta,device):
 lx,ly,*_=lab;mx,my,*_=meta;lx,ly,mx,my=lx.to(device),ly.to(device),mx.to(device),my.to(device);weak,strong=weak.to(device),strong.to(device);unc_opt.zero_grad(set_to_none=True)
 with higher.innerloop_ctx(model.head,head_opt) as (fhead,diffopt):
  with torch.no_grad():lf=model.encode(lx);wf=model.encode(weak);sf=model.encode(strong);mf=model.encode(mx)
  lp=fhead(lf).squeeze(-1);wp=fhead(wf).squeeze(-1).detach();sp=fhead(sf).squeeze(-1);ui=torch.stack(((sp-wp).detach(),sp.detach()),-1)
  diffopt.step(F.mse_loss(lp,ly)+10.0*(torch.exp(-unc(ui))/2*(sp-wp).pow(2).unsqueeze(-1)).mean());mp=fhead(mf).squeeze(-1);mi=torch.stack(((mp-my).detach(),mp.detach()),-1);(F.mse_loss(mp,my)-.1*unc(mi).mean()).backward();unc_opt.step()
def loaders(a):
 c=load_json(a.protocol_root/'cohort.json');m=load_json(a.manifest);validate_manifest(m,c);mean,std=m['target_scaler']['mean'],m['target_scaler']['std']
 ds={'labeled':Images(c,m['labeled_indices'],a.data_root,'weak',mean,std),'validation':Images(c,c['splits']['validation'],a.data_root,'eval',mean,std),'test':Images(c,c['splits']['test'],a.data_root,'eval',mean,std)}
 if m['unlabeled_indices']:ds['unlabeled']=Views(c,m['unlabeled_indices'],a.data_root,mean,std)
 out={}
 for role,d in ds.items():
  g=torch.Generator().manual_seed(a.seed+{'labeled':11000,'unlabeled':22000,'validation':33000,'test':44000}[role]);out[role]=DataLoader(d,a.batch_size,shuffle=role in ('labeled','unlabeled'),num_workers=a.num_workers,pin_memory=True,generator=g)
 return c,m,out,mean,std
@torch.no_grad()
def evaluate(model,loader,mean,std,device,preds=False):
 model.eval();ps=[];ys=[];ids=[];names=[];sex=[]
 for x,y,i,name,male in loader:ps.append(model(x.to(device)).float().cpu()*std+mean);ys.append(y*std+mean);ids.append(i);names.extend(name);sex.append(male)
 p=torch.cat(ps).numpy();y=torch.cat(ys).numpy();idx=torch.cat(ids).numpy();sx=torch.cat(sex).numpy();mse=float(np.mean((p-y)**2));mae=float(np.mean(abs(p-y)));r2=float(1-np.sum((p-y)**2)/(np.sum((y-y.mean())**2)+1e-12));pl=float(pearsonr(p,y).statistic);sr=float(spearmanr(p,y).statistic);z=(mse,mae,math.sqrt(mse),r2,pl,sr)
 assert np.isfinite(z).all();return (*z,p,y,idx,np.asarray(names),sx) if preds else z
def main():
 reorient=os.environ.get('HPL_REORIENT')=='1'
 p=argparse.ArgumentParser();p.add_argument('--backbone',choices=('resnet50','dinov2_vits14'),required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol-root',type=Path,default=Path('data/rsna_bone_age/2017/protocol'));p.add_argument('--data-root',type=Path,default=Path('data/rsna_bone_age/2017/extracted'));p.add_argument('--output-dir',type=Path);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--num-workers',type=int,default=2);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 assert torch.cuda.is_available() and torch.cuda.device_count()==1;seed_all(a.seed);device=torch.device('cuda:0');model,identity=construct(a.backbone,device);target_hash=state_hash(model);c,m,ls,mean,std=loaders(a)
 steps=len(ls['unlabeled']);lb=len(ls['labeled']);cycles,rem=divmod(steps,lb);exposures=(cycles*len(ls['labeled'].dataset)+min(rem*a.batch_size,len(ls['labeled'].dataset)))*a.epochs;opt,sched,optcfg=optimize(model,a.backbone,a.epochs)
 cfg={'dataset':'RSNA Pediatric Bone Age Challenge 2017','protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'split_policy':c['split_policy'],'split_digest':c['split_digest'],'cohort_digest':c['cohort_digest'],'manifest_digest':m['manifest_digest'],'target_scaler':m['target_scaler'],'model_inputs':'pixels_only','method':'canonical_uniform','backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'epochs':a.epochs,'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'gradient_accumulation':1,'persistent_target_optimizer_steps':steps*a.epochs,'labeled_example_exposures':exposures,'selection':'lowest validation MAE in original months','target_init_hash':target_hash,'model':identity,'optimization':optcfg,'uniform_definition':{'pseudo_target':'stopgrad(f_theta(weak(x_u)))','consistency':'MSE(f_theta(strong(x_u)), pseudo_target)','r':1.0,'lambda_u':1.0,'unlabeled_steps_per_batch':1}}
 if reorient:
  cfg.update(method='hpl',plugin='reorient',supervised_recycling=True,anchor_frequency_k=1,supervised_optimizer_steps_per_unlabeled_update=0,hpl={'upstream_commit':'89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20','w_ulb':10.0,'lambda2':.1,'meta_update_frequency':5,'weight_formula':'exp(-uncertainty)/2'})
  unc=UncertaintyLearner().to(device);uncopt=torch.optim.Adam(unc.parameters(),lr=1e-4);metaopt=torch.optim.Adam(model.head.parameters(),lr=1e-3 if a.backbone=='resnet50' else 1e-4)
 if a.preflight:
  x,y,*_=next(iter(ls['labeled']));weak,strong,*_=next(iter(ls['unlabeled']));x,y,weak,strong=x.to(device),y.to(device),weak.to(device),strong.to(device);opt.zero_grad();pseudo=model(weak).detach();loss=F.mse_loss(model(x),y)+F.mse_loss(model(strong),pseudo);loss.backward();opt.step();print(json.dumps({'status':'pass','test_model_inference_count':0,'target_init_hash':target_hash,'persistent_target_optimizer_steps':cfg['persistent_target_optimizer_steps'],'labeled_example_exposures':exposures,'uniform_definition':cfg['uniform_definition'],'model_inputs':'pixels_only','peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}));return
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'metrics.json').exists();write_json(out/'config.json',cfg);hist=[];best=math.inf;best_epoch=0;start=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,a.epochs+1):
  model.train();lc=cycle(ls['labeled']);mc=cycle(ls['labeled']);iterator=ls['unlabeled'];tot=0
  for step,item in enumerate(iterator):
   lab=next(lc);lx,ly,*_=lab;lx,ly=lx.to(device),ly.to(device);weak,strong=item[0].to(device),item[1].to(device)
   if reorient and step%5==0:hpl_meta(model,unc,metaopt,uncopt,lab,item[0],item[1],next(mc),device)
   opt.zero_grad();sup=F.mse_loss(model(lx),ly)
   with torch.no_grad():pseudo=model(weak).detach()
   strong_pred=model(strong)
   if reorient:
    ui=torch.stack(((strong_pred-pseudo).detach(),strong_pred.detach()),-1);weight=(torch.exp(-unc(ui))/2).detach();ul=(weight*(strong_pred-pseudo).pow(2).unsqueeze(-1)).mean()
   else:ul=F.mse_loss(strong_pred,pseudo)
   loss=sup+(10.0*ul if reorient else ul);loss.backward();opt.step();tot+=float(loss)
  sched.step();metrics=evaluate(model,ls['validation'],mean,std,device);vmae=metrics[1];hist.append({'epoch':epoch,'train_loss':tot/steps,'validation_mae_months':vmae,'persistent_target_steps_completed':epoch*steps})
  if vmae<best:best=vmae;best_epoch=epoch;torch.save({'version':'rsna-bone-age-formal-v1','model':model.state_dict(),'epoch':epoch,'validation_mae_months':vmae,'config':cfg},out/'best.pt')
  print(f'epoch={epoch}/{a.epochs} validation_mae_months={vmae:.6f} best_epoch={best_epoch}',flush=True)
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);z=evaluate(model,ls['test'],mean,std,device,True);mse,mae,rmse,r2,pl,sr,pred,target,idx,names,sex=z;np.savez_compressed(out/'test_predictions.npz',predictions=pred,targets=target,image_indices=idx,image_ids=names,male=sex);metrics={'best_epoch':best_epoch,'best_validation_mae_months':best,'test_mae_months':mae,'test_mse_months2':mse,'test_rmse_months':rmse,'test_r2':r2,'test_plcc':pl,'test_srocc':sr,'checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1,'test_time_calibration':False,'ordered_prediction_alignment':bool(np.array_equal(idx,np.asarray(c['splits']['test'])))};write_json(out/'metrics.json',metrics);write_json(out/'metadata.json',{'status':'complete','method':'canonical_uniform','backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'protocol_version':PROTOCOL_VERSION,'cohort_digest':c['cohort_digest'],'split_digest':c['split_digest'],'manifest_digest':m['manifest_digest'],'target_init_hash':target_hash,**metrics,'model_inputs':'pixels_only','runtime_seconds':time.time()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_gpu':os.environ.get('RSNA_PHYSICAL_GPU'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'process_local_device':'cuda:0'})
 if reorient:
  md=load_json(out/'metadata.json');md.update(method='hpl',plugin='reorient',supervised_recycling=True,anchor_frequency_k=1,supervised_optimizer_steps_per_unlabeled_update=0);write_json(out/'metadata.json',md)
 with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0]);w.writeheader();w.writerows(hist)
if __name__=='__main__':main()
