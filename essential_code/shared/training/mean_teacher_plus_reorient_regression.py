#!/usr/bin/env python3
"""Formal Mean Teacher for frozen IMDB-WIKI, CheXchoNet, and RSNA protocols."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader,Dataset,Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/mean-teacher/pytorch'))
from mean_teacher.data import TwoStreamBatchSampler
from baselines.benchmark_io import write_history,write_json,write_npz
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor

UPSTREAM_URL='https://github.com/CuriousAI/mean-teacher.git'
UPSTREAM_SHA='546348ff863c998c26be4339021425df973b4a36'
RESNET_ID='torchvision.ResNet50_Weights.IMAGENET1K_V1 (resnet50-0676ba61.pth)'
DINO_ID='DINOv2 ViT-S/14 LVD-142M (dinov2_vits14)'
BASELINE_REFERENCE_CAMPAIGN='gpu04:artifacts/formal_mt_adam_20260909'
BASELINE_REFERENCE_SHA256='4a909171d9ce2dc0865ff46bfd94a80530a31b94f6fcd7c3db64e7dd797d7f52'
REORIENT_LOADER_SEED_OFFSET=11000

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def model_sha(model):
 h=hashlib.sha256()
 for name,tensor in model.state_dict().items():
  h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def reorient_worker_seed(_):
 worker_seed=torch.initial_seed()%2**32;random.seed(worker_seed);np.random.seed(worker_seed)
def sigmoid_rampup(current,length):
 phase=1.0-np.clip(current,0.0,length)/length if length else 0.0
 return float(np.exp(-5.0*phase*phase))
@torch.no_grad()
def update_ema_variables(student,teacher,alpha,step):
 alpha=min(1-1/(step+1),alpha)
 for target,source in zip(teacher.parameters(),student.parameters()):target.data.mul_(alpha).add_(source.data,alpha=1-alpha)

class ImagePair(Dataset):
 def __init__(self,ids,labeled,path,target,student_transform,teacher_transform,mean,std,gray=False):
  self.ids=list(ids);self.labeled=set(labeled);self.path=path;self.target=target;self.st=student_transform;self.tt=teacher_transform;self.mean=mean;self.std=std;self.gray=gray
 def __len__(self):return len(self.ids)
 def __getitem__(self,p):
  i=self.ids[p]
  with Image.open(self.path(i)) as image:
   image=image.convert('L').convert('RGB') if self.gray else image.convert('RGB');xs=self.st(image.copy());xt=self.tt(image.copy())
  y=(self.target(i)-self.mean)/self.std if i in self.labeled else float('nan')
  return xs,xt,np.float32(y)
class ImageEval(Dataset):
 def __init__(self,ids,path,target,transform,gray=False):self.ids=list(ids);self.path=path;self.target=target;self.transform=transform;self.gray=gray
 def __len__(self):return len(self.ids)
 def __getitem__(self,p):
  i=self.ids[p]
  with Image.open(self.path(i)) as image:image=image.convert('L').convert('RGB') if self.gray else image.convert('RGB');x=self.transform(image)
  return x,np.float32(self.target(i)),np.int64(i)

def dataset_spec(a):
 m=json.loads(a.manifest.read_text())
 if m['seed']!=a.seed or not math.isclose(m['labeled_ratio'],a.ratio):raise RuntimeError('manifest run identity mismatch')
 if a.dataset=='imdb_wiki':
  from data_processing.imdb_wiki_protocol import load_cohort,load_seed_manifest,validate_cohort,build_labeled_transform,build_weak_transform,build_evaluation_transform
  cohort_path=ROOT/'data_processing/splits/imdb_wiki_dir_cohort_v1.json';cohort=load_cohort(cohort_path);validate_cohort(cohort,a.data_root);m=load_seed_manifest(a.manifest,cohort)
  ids=m['labeled_indices']+m['unlabeled_indices'];labeled=m['labeled_indices'];val=m['splits']['validation'];test=m['splits']['test'];record=lambda i:cohort['records'][i]
  return {'dataset':'IMDB-WIKI-DIR','target':'age','unit':'years','protocol':'imdb-wiki-dir-benchmark-v1','transform':m['transform_version'],'cohort_digest':cohort['cohort_sha256'],'split_digest':None,'counts':m['counts'],'mean':m['label_scaler']['mean'],'std':m['label_scaler']['std'],'ids':ids,'labeled':labeled,'val':val,'test':test,'path':lambda i:a.data_root/record(i)['path'],'target_fn':lambda i:float(record(i)['age']),'student_transform':build_labeled_transform(),'teacher_transform':build_weak_transform(),'evaluation_transform':build_evaluation_transform(),'gray':False}
 if a.dataset=='rsna_bone_age':
  from data_processing.rsna_bone_age import load_json,validate_manifest,build_transform,PROTOCOL_VERSION,TRANSFORM_VERSION
  if a.protocol_root is None:raise RuntimeError('RSNA requires --protocol-root')
  cohort=load_json(a.protocol_root/'cohort.json');validate_manifest(m,cohort)
  if cohort['protocol_version']!=PROTOCOL_VERSION or cohort['transform_version']!=TRANSFORM_VERSION:raise RuntimeError('wrong RSNA protocol identity')
  records=cohort['records']
  return {'dataset':'RSNA Pediatric Bone Age Challenge 2017','target':'bone age','unit':'months','protocol':PROTOCOL_VERSION,'transform':TRANSFORM_VERSION,'cohort_digest':cohort['cohort_digest'],'split_digest':cohort['split_digest'],'counts':m['counts'],'mean':m['target_scaler']['mean'],'std':m['target_scaler']['std'],'ids':m['labeled_indices']+m['unlabeled_indices'],'labeled':m['labeled_indices'],'val':cohort['splits']['validation'],'test':cohort['splits']['test'],'path':lambda i:a.data_root/records[i]['image_path'],'target_fn':lambda i:float(records[i]['bone_age_months']),'student_transform':build_transform('weak'),'teacher_transform':build_transform('strong'),'evaluation_transform':build_transform('eval'),'gray':True}
 from data_processing.chexchonet import discover_metadata,load_records
 from data_processing.chexchonet_protocol import validate_manifest,build_weak_transform,build_strong_transform,build_evaluation_transform
 records=load_records(discover_metadata(a.data_root));validate_manifest(m,records)
 if m['protocol_version']!='chexchonet-lvidd-benchmark-v1':raise RuntimeError('wrong CheXchoNet protocol')
 return {'dataset':'CheXchoNet','target':'LVIDd','unit':'centimeters','protocol':m['protocol_version'],'transform':m['transform_version'],'cohort_digest':m['cohort_digest'],'split_digest':m['split_digest'],'counts':m['counts'],'mean':m['label_scaler']['mean'],'std':m['label_scaler']['std'],'ids':m['indices']['labeled']+m['indices']['unlabeled'],'labeled':m['indices']['labeled'],'val':m['indices']['validation'],'test':m['indices']['test'],'path':lambda i:a.data_root/'images'/records[i]['image_path'],'target_fn':lambda i:float(records[i]['targets']['lvidd']),'student_transform':build_weak_transform(),'teacher_transform':build_strong_transform(),'evaluation_transform':build_evaluation_transform(),'gray':True}

def make_loader(ds,batch,seed,role,workers):
 g=torch.Generator();g.manual_seed(seed*1009+{'validation':37,'test':53}[role])
 return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,drop_last=False,generator=g)
@torch.no_grad()
def evaluate(model,loader,device,mean,std):
 model.eval();ps=[];ys=[];ids=[]
 for x,y,i in loader:ps.append((model(x.to(device)).float().cpu().numpy()*std+mean));ys.append(y.numpy());ids.append(i.numpy())
 p=np.concatenate(ps);y=np.concatenate(ys);idx=np.concatenate(ids);e=p-y
 return float(np.mean(np.abs(e))),float(1-np.sum(e**2)/(np.sum((y-y.mean())**2)+1e-12)),p,y,idx
def optimizer_for(model,backbone,epochs):
 if backbone=='dinov2':
  o=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=.05);s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs);c={'optimizer':'AdamW','backbone_lr':1e-5,'head_lr':1e-4,'weight_decay':.05,'scheduler':'CosineAnnealingLR','T_max':epochs}
 else:
  o=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3);s=torch.optim.lr_scheduler.StepLR(o,10,.1);c={'optimizer':'Adam','lr':1e-4,'weight_decay':1e-3,'scheduler':'StepLR','step_size':10,'gamma':.1}
 return o,s,c
def checkpoint(student,teacher,optimizer,scheduler,epoch,val,a,spec,initial_student_sha256,initial_teacher_sha256):
 return {'checkpoint_version':'formal-regression-mean-teacher-v1','student':student.state_dict(),'teacher':teacher.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'validation_mae':val,'manifest_sha256':sha(a.manifest),'cohort_digest':spec['cohort_digest'],'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'upstream_commit_sha':UPSTREAM_SHA,'method':'mean_teacher_plus_reorient','baseline_reference_trainer_sha256':BASELINE_REFERENCE_SHA256,'executed_trainer_sha256':sha(__file__),'initial_student_sha256':initial_student_sha256,'initial_teacher_sha256':initial_teacher_sha256,'reorient_loader_effective_seed':a.seed+REORIENT_LOADER_SEED_OFFSET}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('imdb_wiki','chexchonet','rsna_bone_age'),required=True);p.add_argument('--backbone',choices=('resnet50','dinov2'),required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--ratio',type=float,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--workers',type=int,default=4);p.add_argument('--consistency',type=float,default=10);p.add_argument('--consistency-rampup',type=float,default=5);p.add_argument('--ema-decay',type=float,default=.9997);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('worker must see exactly one CUDA device')
 spec=dataset_spec(a);device=torch.device('cuda:0');train_ds=ImagePair(spec['ids'],spec['labeled'],spec['path'],spec['target_fn'],spec['student_transform'],spec['teacher_transform'],spec['mean'],spec['std'],spec['gray']);nl=len(spec['labeled']);sampler=TwoStreamBatchSampler(range(nl,len(train_ds)),range(nl),a.batch_size,a.batch_size//2);train=DataLoader(train_ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=True);reorient_generator=torch.Generator().manual_seed(a.seed+REORIENT_LOADER_SEED_OFFSET);reorient=DataLoader(Subset(train_ds,range(nl)),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=False,generator=reorient_generator,worker_init_fn=reorient_worker_seed);val=make_loader(ImageEval(spec['val'],spec['path'],spec['target_fn'],spec['evaluation_transform'],spec['gray']),a.batch_size,a.seed,'validation',a.workers);test=make_loader(ImageEval(spec['test'],spec['path'],spec['target_fn'],spec['evaluation_transform'],spec['gray']),a.batch_size,a.seed,'test',a.workers)
 student=(ResNet50Regressor(True) if a.backbone=='resnet50' else DINOv2Regressor('small')).to(device);teacher=copy.deepcopy(student).to(device)
 for q in teacher.parameters():q.detach_()
 initial_student_sha256=model_sha(student);initial_teacher_sha256=model_sha(teacher);assert initial_student_sha256==initial_teacher_sha256
 optimizer,scheduler,optcfg=optimizer_for(student,a.backbone,a.epochs)
 out=a.output_dir
 if out.exists() and any(out.iterdir()):raise FileExistsError(f'refusing nonempty output {out}')
 out.mkdir(parents=True,exist_ok=True)
 if a.preflight:
  native_iterator=iter(train);reorient_iterator=iter(reorient);xs,xt,y=next(native_iterator);rxs,_,ry=next(reorient_iterator);xs,xt,y,rxs,ry=xs.to(device),xt.to(device),y.to(device),rxs.to(device),ry.to(device);mask=torch.isfinite(y);assert torch.isfinite(ry).all()
  with torch.no_grad():tp=teacher(xt)
  sp=student(xs);native_sup=F.mse_loss(sp[mask],y[mask]);cons=F.mse_loss(sp,tp);reorient_sup=F.mse_loss(student(rxs),ry);loss=native_sup+cons+reorient_sup;optimizer.zero_grad();loss.backward();optimizer.step();update_ema_variables(student,teacher,a.ema_decay,1);print(json.dumps({'status':'pass','dataset':spec['dataset'],'loss_total':float(loss),'loss_native_supervised':float(native_sup),'loss_reorient_supervised':float(reorient_sup),'loss_consistency':float(cons),'optimizer_steps':1,'ema_updates':1,'initial_student_sha256':initial_student_sha256,'initial_teacher_sha256':initial_teacher_sha256,'reorient_loader_effective_seed':a.seed+REORIENT_LOADER_SEED_OFFSET,'test_evaluations':0}));return
 started=time.time();torch.cuda.reset_peak_memory_stats();architecture=RESNET_ID if a.backbone=='resnet50' else DINO_ID
 config={'dataset':spec['dataset'],'target':spec['target'],'target_unit':spec['unit'],'method':'mean_teacher_plus_reorient','paper_method':'Mean Teacher + Reorient','baseline_reference_campaign':BASELINE_REFERENCE_CAMPAIGN,'baseline_reference_trainer_sha256':BASELINE_REFERENCE_SHA256,'executed_trainer_sha256':sha(__file__),'reorient_definition':'one additional coefficient-1 standardized-target supervised MSE in each native MT optimizer update','reorient_coefficient':1.0,'reorient_loader_seed_offset':REORIENT_LOADER_SEED_OFFSET,'reorient_loader_effective_seed':a.seed+REORIENT_LOADER_SEED_OFFSET,'physical_gpu':os.environ.get('MTPR_PHYSICAL_GPU'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'reorient_batch_size':a.batch_size,'optimizer_steps_per_native_iteration':1,'ema_updates_per_native_iteration':1,'initial_student_sha256':initial_student_sha256,'initial_teacher_sha256':initial_teacher_sha256,'backbone':a.backbone,'ratio':a.ratio,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'supervised_objective':'standardized-target MSE','consistency_objective':'student-vs-current-EMA-teacher standardized prediction MSE','optimization':optcfg,'teacher_optimizer':None,'teacher_requires_grad':False,'student_architecture':architecture,'teacher_architecture':'EMA '+architecture,'ema_decay':a.ema_decay,'ema_early_step_correction':True,'consistency':a.consistency,'consistency_rampup':a.consistency_rampup,'selection_metric':'validation MAE only in original target units','protocol_version':spec['protocol'],'transform_version':spec['transform'],'manifest_sha256':sha(a.manifest),'cohort_digest':spec['cohort_digest'],'split_digest':spec['split_digest'],'counts':spec['counts'],'label_scaler':{'mean':spec['mean'],'std':spec['std']},'upstream_repo_url':UPSTREAM_URL,'upstream_commit_sha':UPSTREAM_SHA};write_json(out/'config.json',config)
 best=math.inf;best_epoch=None;history=[];step=0;ema_ref=next(teacher.parameters()).detach().clone();ema_seen=False
 for epoch in range(1,a.epochs+1):
  student.train();teacher.train();sums=np.zeros(4);n=0;native_steps=0;reorient_batches=0
  native_iterator=iter(train);reorient_iterator=iter(reorient)
  for xs,xt,y in native_iterator:
   xs,xt,y=xs.to(device),xt.to(device),y.to(device);mask=torch.isfinite(y)
   with torch.no_grad():tp=teacher(xt)
   sp=student(xs)
   try:rxs,_,ry=next(reorient_iterator)
   except StopIteration:reorient_iterator=iter(reorient);rxs,_,ry=next(reorient_iterator)
   rxs,ry=rxs.to(device),ry.to(device);assert torch.isfinite(ry).all();reorient_sup=F.mse_loss(student(rxs),ry)
   sup=F.mse_loss(sp[mask],y[mask]);cons=F.mse_loss(sp,tp);weight=a.consistency*sigmoid_rampup(epoch-1,a.consistency_rampup);loss=sup+weight*cons+reorient_sup
   if not all(torch.isfinite(v) for v in (sup,cons,loss)):raise RuntimeError('non-finite Mean Teacher loss')
   optimizer.zero_grad();loss.backward();optimizer.step();step+=1;update_ema_variables(student,teacher,a.ema_decay,step);ema_seen=ema_seen or not torch.equal(ema_ref,next(teacher.parameters()).detach());sums+=np.array([float(loss),float(sup),float(reorient_sup),float(cons)]);n+=1;native_steps+=1;reorient_batches+=1
  assert native_steps==len(train) and reorient_batches==native_steps
  vm,vr,_,_,_=evaluate(teacher,val,device,spec['mean'],spec['std']);improved=vm<best
  if improved:best=vm;best_epoch=epoch;torch.save(checkpoint(student,teacher,optimizer,scheduler,epoch,vm,a,spec,initial_student_sha256,initial_teacher_sha256),out/'best.pt')
  scheduler.step();torch.save(checkpoint(student,teacher,optimizer,scheduler,epoch,vm,a,spec,initial_student_sha256,initial_teacher_sha256),out/'last.pt');history.append({'epoch':epoch,'loss_total':sums[0]/n,'loss_supervised':sums[1]/n,'loss_reorient_supervised':sums[2]/n,'loss_consistency_unweighted':sums[3]/n,'native_mt_steps':native_steps,'reorient_labeled_batches':reorient_batches,'consistency_weight':weight,'validation_mae':vm,'validation_r2':vr,'best_so_far':int(improved)});print(f'epoch={epoch}/{a.epochs} val_mae={vm:.6f} best_epoch={best_epoch}',flush=True)
 write_history(out/'history.csv',history);ck=torch.load(out/'best.pt',map_location=device,weights_only=False)
 if ck['epoch']!=best_epoch or ck['validation_mae']!=best or ck['manifest_sha256']!=sha(a.manifest):raise RuntimeError('selected checkpoint identity mismatch')
 teacher.load_state_dict(ck['teacher']);tm,tr,pred,target,idx=evaluate(teacher,test,device,spec['mean'],spec['std']);expected=np.asarray(spec['test'])
 if not np.array_equal(idx,expected) or not np.isfinite([tm,tr,*pred]).all():raise RuntimeError('test contract failure')
 write_npz(out/'test_predictions.npz',cohort_indices=idx,predictions=pred,targets=target);write_npz(out/'analysis_snapshot.npz',test_cohort_indices=idx,test_predictions=pred,test_targets=target)
 metrics={'validation_mae':best,'test_mae':tm,'test_r2':tr};write_json(out/'metrics.json',metrics);metadata={'status':'complete','dataset':spec['dataset'],'target':spec['target'],'target_unit':spec['unit'],'method':'mean_teacher_plus_reorient','paper_method':'Mean Teacher + Reorient','baseline_reference_campaign':BASELINE_REFERENCE_CAMPAIGN,'baseline_reference_trainer_sha256':BASELINE_REFERENCE_SHA256,'executed_trainer_sha256':sha(__file__),'reorient_loader_seed_offset':REORIENT_LOADER_SEED_OFFSET,'reorient_loader_effective_seed':a.seed+REORIENT_LOADER_SEED_OFFSET,'physical_gpu':os.environ.get('MTPR_PHYSICAL_GPU'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'reorient_coefficient':1.0,'optimizer_steps_per_native_iteration':1,'ema_updates_per_native_iteration':1,'initial_student_sha256':initial_student_sha256,'initial_teacher_sha256':initial_teacher_sha256,'backbone':a.backbone,'seed':a.seed,'labeled_ratio':a.ratio,'protocol_version':spec['protocol'],'transform_version':spec['transform'],'manifest_sha256':sha(a.manifest),'cohort_digest':spec['cohort_digest'],'split_digest':spec['split_digest'],'counts':spec['counts'],'label_scaler':{'mean':spec['mean'],'std':spec['std']},'teacher_optimizer':None,'teacher_requires_grad':False,'teacher_forward_no_grad':True,'ema_update_observed':ema_seen,'checkpoint_reloaded':True,'checkpoint_metadata_verified':True,'test_used_for_selection':False,'test_evaluations':1,'test_model_inference_count':1,'predictions_in_manifest_order':True,'prediction_count':len(pred),'best_epoch':best_epoch,'best_validation_mae':best,'upstream_commit_sha':UPSTREAM_SHA,'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(),'wall_clock_seconds':time.time()-started};write_json(out/'metadata.json',metadata);write_json(out/'successful_attempt.json',{'status':'success','completed_unix':time.time(),'manifest_sha256':sha(a.manifest),'best_epoch':best_epoch});print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
