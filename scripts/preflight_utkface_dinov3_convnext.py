#!/usr/bin/env python3
"""No-gradient DINOv3 ConvNeXt-Tiny UTKFace preflight."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from baselines.utkface_data import BenchmarkContext,TupleDataset
from data_processing.utkface_protocol import dataloader_generator,seed_dataloader_worker
from models.dinov3_convnext_backbone import DINOv3ConvNextTinyRegressor,MODEL_IDENTIFIER,MODEL_REVISION,WEIGHT_SHA256,FEATURE_DIMENSION
def make(ds,role,seed):return DataLoader(ds,batch_size=32,shuffle=role in ('labeled','unlabeled'),drop_last=role in ('labeled','unlabeled'),num_workers=4,worker_init_fn=seed_dataloader_worker,generator=dataloader_generator(seed,role))
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=('rapl','hpl'),required=True);a=p.parse_args();seed=0
 assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.current_device()==0
 context=BenchmarkContext(ROOT,ROOT/'data'/'utkface_all',ROOT/'data_processing/splits/utkface_ratio_0.05_seed_0.json')
 datasets={'labeled':TupleDataset(context,'labeled','labeled',repeat=2 if a.method=='hpl' else 1),'unlabeled':TupleDataset(context,'unlabeled','weak_strong'),'validation':TupleDataset(context,'validation','evaluation'),'test':TupleDataset(context,'test','evaluation')}
 batches={k:next(iter(make(v,k,seed))) for k,v in datasets.items()};x=batches['labeled'][0].cuda()
 if a.method=='rapl':
  target=DINOv3ConvNextTinyRegressor().cuda();probe=DINOv3ConvNextTinyRegressor().cuda()
  assert target is not probe and target.backbone is not probe.backbone
  assert target.weight_identifier==probe.weight_identifier==MODEL_IDENTIFIER and target.weight_checksum_sha256==probe.weight_checksum_sha256==WEIGHT_SHA256
  for q in probe.backbone.parameters():q.requires_grad_(False)
  assert any(q.requires_grad for q in target.backbone.parameters()) and not any(q.requires_grad for q in probe.backbone.parameters())
  with torch.no_grad():
   tf=target.backbone(x);pf=probe.backbone(x);pred=target(x)
   assert tf.shape==pf.shape==(32,FEATURE_DIMENSION) and pred.shape==(32,)
 else:
  official=ROOT/'third_party/Heteroscedastic-Pseudo-Labels/utkface'
  spec=importlib.util.spec_from_file_location('hpl_uncertainty_learner',official/'models/uncertainty_learner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);UncertaintyLearner=module.UncertaintyLearner
  from baselines.dinov3_convnext_hpl import DINOv3ConvNextTinyHPLRegressor
  model=DINOv3ConvNextTinyHPLRegressor().cuda();unc=UncertaintyLearner(2,1).cuda()
  with torch.no_grad():
   pred,feat=model(x);u=unc(torch.zeros(32,2,device='cuda'))
   assert pred.shape==(32,1) and feat.shape==(32,FEATURE_DIMENSION) and u.shape==(32,1)
 print(json.dumps({'status':'passed','method':a.method,'model_identifier':MODEL_IDENTIFIER,'revision':MODEL_REVISION,'weight_sha256':WEIGHT_SHA256,'feature_dimension':FEATURE_DIMENSION,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'local_device':'cuda:0','counts':context.manifest['counts'],'scaler':{'mean':context.mean,'std':context.std},'batch_contracts':{'labeled':list(batches['labeled'][0].shape),'unlabeled_weak':list(batches['unlabeled'][0][0].shape),'unlabeled_strong':list(batches['unlabeled'][0][1].shape),'validation':list(batches['validation'][0].shape),'test':list(batches['test'][0].shape)}},sort_keys=True))
if __name__=='__main__':main()
