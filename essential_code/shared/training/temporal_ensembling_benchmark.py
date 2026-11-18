"""Formal image-regression Temporal Ensembling port from s-laine/tempens.

The algorithmic correspondence is documented in docs/temporal_ensembling_port.md.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data_processing.utkface_protocol import (build_evaluation_transform,
    build_labeled_transform, dataloader_generator, inverse_normalize_age,
    load_cohort, load_seed_manifest, loader_metadata, runtime_metadata,
    seed_dataloader_worker)
from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor

ROOT=Path(__file__).resolve().parents[1]
PROTOCOLS={
 'utkface': {
  'dataset':'UTKFace', 'cohort':ROOT/'data_processing/splits/utkface_cohort_v1.json',
  'digest':'61c397e0b6ac4be78db1b1c1431a65b031e2a2fd5089361e4e784ad21ad8af56',
  'module':'data_processing.utkface_protocol',
 },
 'imdb_wiki': {
  'dataset':'IMDB-WIKI-DIR', 'cohort':ROOT/'data_processing/splits/imdb_wiki_dir_cohort_v1.json',
  'digest':'919fe3e1b959e1fe75e08e83310a84c1c3a9d53a16812a1bb5f1e0117ba97f43',
  'module':'data_processing.imdb_wiki_protocol',
 },
 'chexchonet_lvidd': {
  'dataset':'CheXchoNet LVIDd', 'digest':'b82d75b491170e123ca60e114f909913e09ecac15bef1153038d8845dc1cf911',
  'split_digest':'ec4348abbdfc1df41bb9d27925c3b3cc346cfc8c7eac2b46f0694bbc7e35620a',
  'protocol_version':'chexchonet-lvidd-benchmark-v1', 'unit':'centimeters',
 },
 'rsna_bone_age': {
  'dataset':'RSNA Pediatric Bone Age Challenge 2017',
  'protocol_version':'rsna-bone-age-benchmark-v1', 'unit':'months',
 },
}
UPSTREAM_REPO='https://github.com/s-laine/tempens'
UPSTREAM_COMMIT='a32743acbdea54560b7d9d90934c1eb5b49a81a7'
ALPHA=.6; UNSUP_MAX=30.; EPOCHS=30; RAMPUP=8; RAMPDOWN=5

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write_json(path,obj):
 p=Path(path); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); q.replace(p)
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def rampup(epoch):
 if epoch<RAMPUP:
  p=1.-max(0.,float(epoch))/RAMPUP; return math.exp(-p*p*5.)
 return 1.
def rampdown(epoch,epochs=EPOCHS):
 if epoch>=epochs-RAMPDOWN:
  ep=(epoch-(epochs-RAMPDOWN))*.5; return math.exp(-(ep*ep)/RAMPDOWN)
 return 1.

class TrainSet(Dataset):
 def __init__(self,cohort,manifest,root,transform):
  self.cohort=cohort; self.root=Path(root); self.indices=list(manifest['splits']['train'])
  self.position={v:k for k,v in enumerate(self.indices)}
  self.labeled=set(manifest['labeled_indices'])
  self.mean=float(manifest['label_scaler']['mean']); self.std=float(manifest['label_scaler']['std']); self.transform=transform
  assert len(self.position)==len(self.indices)
 def __len__(self): return len(self.indices)
 def __getitem__(self,pos):
  canonical=self.indices[pos]; rec=self.cohort['records'][canonical]
  with Image.open(self.root/rec['path']) as im: image=self.transform(im.convert('RGB'))
  target=(float(rec['age'])-self.mean)/self.std
  return image,np.float32(target),np.float32(canonical in self.labeled),pos,canonical

class EvalSet(Dataset):
 def __init__(self,cohort,indices,root,mean,std,transform):
  self.cohort=cohort; self.indices=list(indices); self.root=Path(root); self.mean=mean; self.std=std; self.transform=transform
 def __len__(self): return len(self.indices)
 def __getitem__(self,pos):
  i=self.indices[pos]; rec=self.cohort['records'][i]
  with Image.open(self.root/rec['path']) as im: x=self.transform(im.convert('RGB'))
  return x,np.float32((float(rec['age'])-self.mean)/self.std),i

def args():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=tuple(PROTOCOLS),default='utkface'); p.add_argument('--backbone',choices=('resnet50','dinov2_vits14'),required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output-dir',type=Path); p.add_argument('--seed',type=int,required=True); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--num-workers',type=int,default=4); p.add_argument('--epochs',type=int,default=EPOCHS); p.add_argument('--resume-from',type=Path); p.add_argument('--interface-check',action='store_true'); p.add_argument('--plugin-reorient',action='store_true',help='cycle one labeled batch into every original Temporal Ensembling optimizer step'); return p.parse_args()
def load_protocol(a):
 spec=PROTOCOLS[a.dataset]
 if a.dataset=='chexchonet_lvidd':
  from data_processing.chexchonet import discover_metadata,load_records
  from data_processing.chexchonet_protocol import build_train_transform,build_evaluation_transform as chex_eval_transform,validate_manifest
  records=load_records(discover_metadata(a.data_root)); raw=json.loads(a.manifest.read_text()); validate_manifest(raw,records)
  if raw['seed']!=a.seed: raise ValueError('CheXchoNet manifest seed mismatch')
  if raw['cohort_digest']!=spec['digest'] or raw['split_digest']!=spec['split_digest']: raise ValueError('CheXchoNet canonical identity mismatch')
  cohort={'records':[{'path':str(Path('images')/r['image_path']),'age':r['targets']['lvidd']} for r in records]}
  train=sorted(raw['indices']['labeled']+raw['indices']['unlabeled']); manifest={'seed':raw['seed'],'labeled_ratio':raw['labeled_ratio'],'protocol_version':raw['protocol_version'],'transform_version':raw['transform_version'],'label_scaler':raw['label_scaler'],'counts':raw['counts'],'labeled_indices':raw['indices']['labeled'],'splits':{'train':train,'validation':raw['indices']['validation'],'test':raw['indices']['test']},'manifest_identity':raw['manifest_sha256'],'split_digest':raw['split_digest']}
  return spec,cohort,manifest,build_train_transform(),chex_eval_transform()
 if a.dataset=='rsna_bone_age':
  from data_processing.rsna_bone_age import build_transform,load_json,validate_manifest
  cohort=load_json(a.data_root.parent/'protocol/cohort.json'); raw=load_json(a.manifest); validate_manifest(raw,cohort)
  if raw['seed']!=a.seed: raise ValueError('RSNA manifest seed mismatch')
  if cohort['protocol_version']!=spec['protocol_version']: raise ValueError('RSNA protocol identity mismatch')
  normalized={'records':[{'path':r['image_path'],'age':r['bone_age_months']} for r in cohort['records']]}
  manifest={'seed':raw['seed'],'labeled_ratio':raw['labeled_ratio'],'protocol_version':raw['protocol_version'],'transform_version':raw['transform_version'],'label_scaler':raw['target_scaler'],'counts':raw['counts'],'labeled_indices':raw['labeled_indices'],'splits':{'train':cohort['splits']['train'],'validation':cohort['splits']['validation'],'test':cohort['splits']['test']},'manifest_identity':raw['manifest_digest'],'split_digest':raw['split_digest']}
  spec={**spec,'digest':raw['cohort_digest'],'split_digest':raw['split_digest']}
  return spec,normalized,manifest,build_transform('weak'),build_transform('eval')
 if a.dataset=='imdb_wiki':
  from data_processing.imdb_wiki_protocol import load_cohort as cohort_loader, load_seed_manifest as manifest_loader
 else:
  cohort_loader,manifest_loader=load_cohort,load_seed_manifest
 cohort=cohort_loader(spec['cohort'])
 if cohort['cohort_sha256']!=spec['digest']: raise ValueError('unexpected canonical cohort digest')
 manifest=manifest_loader(a.manifest,cohort)
 if manifest['seed']!=a.seed: raise ValueError('manifest seed mismatch')
 return spec,cohort,manifest,build_labeled_transform(),build_evaluation_transform()
def model_and_identity(name,device):
 if name=='resnet50':
  m=ResNet50Regressor(pretrained=True).to(device); ident={'model':'torchvision ResNet-50','pretrained_identifier':'torchvision.ResNet50_Weights.IMAGENET1K_V1','pretrained_url':'https://download.pytorch.org/models/resnet50-0676ba61.pth','feature_dimension':2048}; maxlrs=(1e-4,1e-3)
 else:
  m=DINOv2Regressor(size='small').to(device); ident={'model':m.model_name,'pretrained_identifier':m.weight_identifier,'pretrained_url':m.weight_url,'checkpoint_sha256':'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9','feature_dimension':m.feature_dim,'pooling':'official DINOv2 forward_features class-token semantics'}; maxlrs=(1e-5,1e-4)
 return m,ident,maxlrs
@torch.no_grad()
def evaluate(model,loader,mean,std,device,full=False):
 model.eval(); ps=[]; ys=[]; ids=[]
 for x,y,i in loader:
  ps.append(inverse_normalize_age(model(x.to(device)).float().cpu(),mean,std)); ys.append(inverse_normalize_age(y.float(),mean,std)); ids.append(i.long())
 p=torch.cat(ps).numpy(); y=torch.cat(ys).numpy(); idx=torch.cat(ids).numpy(); mae=float(np.mean(np.abs(p-y))); r2=float(1-np.sum((p-y)**2)/(np.sum((y-y.mean())**2)+1e-12))
 if not np.isfinite([mae,r2]).all(): raise RuntimeError('non-finite evaluation')
 return (mae,r2,p,y,idx) if full else (mae,r2)

def main():
 a=args(); assert a.epochs==EPOCHS, 'formal configuration fixes 30 shared-protocol epochs'
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('exactly one visible CUDA GPU required')
 started=time.time(); seed_all(a.seed); spec,cohort,manifest,train_transform,eval_transform=load_protocol(a)
 mean=float(manifest['label_scaler']['mean']); std=float(manifest['label_scaler']['std']); train=TrainSet(cohort,manifest,a.data_root,train_transform)
 common=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,worker_init_fn=seed_dataloader_worker)
 loaders={'train':DataLoader(train,shuffle=True,drop_last=False,generator=dataloader_generator(a.seed,'unlabeled'),**common)}
 if a.plugin_reorient:
  labeled_positions=[train.position[i] for i in manifest['labeled_indices']]
  if not labeled_positions: raise RuntimeError('Reorient requires a nonempty labeled manifest')
  loaders['recycled_labeled']=DataLoader(Subset(train,labeled_positions),shuffle=True,drop_last=False,generator=dataloader_generator(a.seed,'labeled'),**common)
 for role in ('validation','test'): loaders[role]=DataLoader(EvalSet(cohort,manifest['splits'][role],a.data_root,mean,std,eval_transform),shuffle=False,drop_last=False,generator=dataloader_generator(a.seed,role),**common)
 device=torch.device('cuda:0'); model,identity,maxlrs=model_and_identity(a.backbone,device); torch.cuda.reset_peak_memory_stats(device)
 medical=a.dataset in ('chexchonet_lvidd','rsna_bone_age')
 if medical and a.backbone=='dinov2_vits14': opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':maxlrs[0]},{'params':model.head.parameters(),'lr':maxlrs[1]}],weight_decay=.05); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
 else: opt=torch.optim.Adam([{'params':model.backbone.parameters(),'lr':maxlrs[0]},{'params':model.head.parameters(),'lr':maxlrs[1]}],betas=(.9,.999),eps=1e-8); scheduler=torch.optim.lr_scheduler.StepLR(opt,10,.1) if medical else None
 n=len(train); ensemble=torch.zeros(n); targets=torch.zeros(n); start_epoch=0; history=[]; best=float('inf'); best_epoch=None
 if a.interface_check:
  model.train(); x,y,mask,pos,canonical=next(iter(loaders['train']))
  mapped=torch.tensor([train.indices[int(p)] for p in pos])
  if not torch.equal(mapped,canonical): raise RuntimeError('canonical sample identity mismatch')
  x=x.to(device); y=y.to(device); mask=mask.to(device); pred=model(x)
  supervised=((pred-y)**2*mask).mean(); consistency=F.mse_loss(pred,targets[pos].to(device)); active_weight=rampup(1)*(UNSUP_MAX*len(manifest['labeled_indices'])/n); recycled=torch.zeros((),device=device)
  if a.plugin_reorient:
   lx,ly,lmask,_,_=next(iter(loaders['recycled_labeled'])); recycled=F.mse_loss(model(lx.to(device)),ly.to(device))
   if not bool(lmask.bool().all()): raise RuntimeError('non-labeled sample in recycled labeled loader')
  total=supervised+active_weight*consistency+recycled
  if not torch.isfinite(torch.stack((supervised,consistency,recycled,total))).all(): raise RuntimeError('non-finite interface-check loss')
  total.backward(); opt.step(); epoch_predictions=torch.zeros(n); epoch_predictions[pos]=pred.detach().cpu()
  temporal_ensemble=ALPHA*ensemble+(1-ALPHA)*epoch_predictions; corrected=temporal_ensemble/(1-ALPHA)
  if not torch.isfinite(corrected).all(): raise RuntimeError('non-finite temporal state')
  print(json.dumps({'status':'pass','dataset':spec['dataset'],'backbone':a.backbone,'manifest_sha256':sha(a.manifest),'cohort_sha256':spec['digest'],'model':identity,'canonical_ids':canonical.tolist(),'temporal_slots':pos.tolist(),'stable_sample_identity':bool(torch.equal(mapped,canonical)),'supervised_loss':float(supervised),'recycled_supervised_loss':float(recycled),'consistency_loss':float(consistency),'active_consistency_weight':active_weight,'total_loss':float(total),'plugin':'reorient' if a.plugin_reorient else None,'test_inference_count':0},sort_keys=True)); return
 if a.output_dir is None: raise ValueError('--output-dir is required for a formal run')
 out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 if (out/'metrics.json').exists(): raise FileExistsError('refusing to overwrite successful result')
 if a.resume_from:
  state=torch.load(a.resume_from,map_location=device,weights_only=False); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); ensemble=state['ensemble_prediction'].cpu(); targets=state['training_targets'].cpu(); start_epoch=state['epoch']; history=state['history']; best=state['best_validation_mae']; best_epoch=state['best_epoch']; torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
  if scheduler is not None and state.get('scheduler') is not None: scheduler.load_state_dict(state['scheduler'])
 protocol={'cohort_sha256':spec['digest'],'cohort_path':str(spec['cohort'].resolve()) if spec.get('cohort') else None,'manifest_path':str(a.manifest.resolve()),'manifest_sha256':sha(a.manifest),'manifest_identity':manifest.get('manifest_identity'),'split_digest':manifest.get('split_digest'),'manifest_seed':a.seed,'protocol_version':manifest['protocol_version'],'transform_version':manifest['transform_version'],'labeled_ratio':float(manifest['labeled_ratio']),'counts':manifest['counts'],'label_scaler':manifest['label_scaler'],'target_unit':spec.get('unit','years')}
 cfg={'dataset':spec['dataset'],'method':'Temporal Ensembling','backbone':a.backbone,'epochs':EPOCHS,'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'seed':a.seed,'model':identity,'protocol':protocol,'official_source_repo':UPSTREAM_REPO,'official_source_commit':UPSTREAM_COMMIT,'prediction_decay':ALPHA,'unsup_weight_max':UNSUP_MAX,'scaled_unsup_weight_max':UNSUP_MAX*len(manifest['labeled_indices'])/n,'rampup_length':RAMPUP,'rampdown_length':RAMPDOWN,'adaptation':'scalar normalized-age prediction replaces class probability vector','plugin':'reorient' if a.plugin_reorient else None,'supervised_recycling':bool(a.plugin_reorient),'anchor_frequency_k':1 if a.plugin_reorient else None,'recycled_labeled_batch_size':a.batch_size if a.plugin_reorient else None,'optimizer_steps_per_mixed_batch':1}; write_json(out/'config.json',cfg)
 scaled=cfg['scaled_unsup_weight_max']
 for epoch in range(start_epoch,EPOCHS):
  t0=time.time(); ru=rampup(epoch); rd=rampdown(epoch); beta1=rd*.9+(1-rd)*.5
  if not medical:
   for group,maxlr in zip(opt.param_groups,maxlrs): group['lr']=ru*rd*maxlr; group['betas']=(beta1,.999)
  weight=0. if epoch==0 else ru*scaled; epoch_pred=torch.zeros(n); seen=torch.zeros(n,dtype=torch.bool); sums=np.zeros(4); count=0; model.train(); labeled_iterator=iter(loaders['recycled_labeled']) if a.plugin_reorient else None
  for x,y,mask,pos,canonical in loaders['train']:
   if not torch.equal(torch.tensor([train.indices[int(p)] for p in pos]),canonical): raise RuntimeError('canonical sample identity mismatch')
   x=x.to(device); y=y.to(device); mask=mask.to(device); temporal=targets[pos].to(device).detach(); opt.zero_grad(set_to_none=True); pred=model(x)
   sup=((pred-y)**2*mask).mean(); cons=F.mse_loss(pred,temporal); recycled=torch.zeros((),device=device)
   if a.plugin_reorient:
    try: lx,ly,lmask,_,_=next(labeled_iterator)
    except StopIteration:
     labeled_iterator=iter(loaders['recycled_labeled']); lx,ly,lmask,_,_=next(labeled_iterator)
    if not bool(lmask.bool().all()): raise RuntimeError('non-labeled sample in recycled labeled loader')
    recycled=F.mse_loss(model(lx.to(device)),ly.to(device))
   loss=sup+weight*cons+recycled
   if not torch.isfinite(loss): raise RuntimeError('non-finite training loss')
   loss.backward(); opt.step(); epoch_pred[pos]=pred.detach().cpu(); seen[pos]=True; sums+=np.array([float(loss),float(sup),float(cons),float(recycled)])*len(y); count+=len(y)
  if not bool(seen.all()): raise RuntimeError('not every canonical training sample executed exactly once')
  ensemble=ALPHA*ensemble+(1-ALPHA)*epoch_pred; targets=ensemble/(1-ALPHA**(epoch+1))
  if not torch.isfinite(ensemble).all() or not torch.isfinite(targets).all(): raise RuntimeError('non-finite temporal state')
  if scheduler is not None: scheduler.step()
  vae,var=evaluate(model,loaders['validation'],mean,std,device); improved=vae<best
  if improved: best=vae; best_epoch=epoch+1; torch.save({'model':model.state_dict(),'epoch':best_epoch,'validation_mae':best,'configuration':cfg},out/'best.pt')
  row={'epoch':epoch+1,'train_loss':sums[0]/count,'train_supervised_mse_masked_mean':sums[1]/count,'train_consistency_mse':sums[2]/count,'train_recycled_supervised_mse':sums[3]/count,'unsup_weight':weight,'rampup':ru,'rampdown':rd,'adam_beta1':beta1,'backbone_learning_rate':opt.param_groups[0]['lr'],'head_learning_rate':opt.param_groups[1]['lr'],'validation_mae':vae,'validation_r2':var,'best_so_far':int(improved),'elapsed_seconds':time.time()-t0}; history.append(row)
  resume={'checkpoint_version':f'{a.dataset}-tempens-v1','model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'epoch':epoch+1,'ensemble_prediction':ensemble,'training_targets':targets,'history':history,'best_validation_mae':best,'best_epoch':best_epoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'train_canonical_indices':train.indices,'manifest_sha256':protocol['manifest_sha256']}; torch.save(resume,out/'last.pt')
  print(f'epoch={epoch+1}/{EPOCHS} loss={row["train_loss"]:.7f} val_mae={vae:.5f} best={best_epoch}',flush=True)
 with (out/'history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
 ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); restored,_=evaluate(model,loaders['validation'],mean,std,device)
 if ck['epoch']!=best_epoch or not np.isclose(restored,best,atol=1e-5,rtol=0): raise RuntimeError('best checkpoint reload verification failed')
 mae,r2,p,y,idx=evaluate(model,loaders['test'],mean,std,device,True); expected=np.asarray(manifest['splits']['test'])
 if not np.array_equal(idx,expected): raise RuntimeError('canonical test prediction order mismatch')
 np.savez_compressed(out/'test_predictions.npz',cohort_indices=idx,relative_image_identifiers=np.asarray([cohort['records'][int(i)]['path'] for i in idx]),predictions=p,targets=y,target_unit=spec.get('unit','years'))
 np.savez_compressed(out/'analysis_snapshot.npz',train_canonical_indices=np.asarray(train.indices),ensemble_prediction=ensemble.numpy(),bias_corrected_targets=targets.numpy())
 metrics={'best_epoch':best_epoch,'validation_mae':best,'test_mae':mae,'test_r2':r2,'checkpoint_reloaded':True,'test_used_for_selection':False,'test_model_inference_count':1}
 metadata={**protocol,'status':'complete','dataset':spec['dataset'],'method':'temporal_ensembling','backbone':a.backbone,'model':identity,'official_source_repo':UPSTREAM_REPO,'official_source_commit':UPSTREAM_COMMIT,'checkpoint_reloaded':True,'restored_epoch_verified':True,'restored_validation_mae_verified':True,'checkpoint_selection':f"lowest validation MAE in original {spec.get('unit','years')}",'test_used_for_selection':False,'test_model_inference_count':1,'stable_sample_identity_verified':True,'temporal_state_finite':True,'canonical_prediction_order_verified':True,'temporal_state_index':'position in canonical official training split, mapped to immutable cohort index','temporal_prediction_history_not_model_ema':True,'bias_correction':f'ensemble_prediction / (1 - {ALPHA} ** (epoch + 1))','consistency_rampup_epochs':RAMPUP,'checkpoint_path':str((out/'best.pt').resolve()),'resume_checkpoint_path':str((out/'last.pt').resolve()),'runtime_seconds':time.time()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device),'physical_batch_size':a.batch_size,'effective_batch_size':a.batch_size,'physical_gpu':os.environ.get('TEMPENS_PHYSICAL_GPU'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'runtime':runtime_metadata(),'plugin':'reorient' if a.plugin_reorient else None,'supervised_recycling':bool(a.plugin_reorient),'anchor_frequency_k':1 if a.plugin_reorient else None}; write_json(out/'metrics.json',metrics); write_json(out/'metadata.json',metadata); print(json.dumps({'status':'complete',**metrics}),flush=True)
if __name__=='__main__': main()
