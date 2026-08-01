import math
from PIL import Image
import torch
from data_processing.rsna_bone_age import *

def records(n=1000):
 return [{'image_id':str(i),'image_path':f'{i}.png','bone_age_months':float(i%217),'male':bool(i%2)} for i in range(n)]

def test_policy_b_is_exact_disjoint_and_deterministic():
 c1=build_cohort(records());c2=build_cohort(records())
 assert c1['splits']==c2['splits'] and c1['split_digest']==c2['split_digest']
 assert c1['counts']=={'train':800,'validation':100,'test':100}
 a,b,c=map(set,(c1['splits']['train'],c1['splits']['validation'],c1['splits']['test']))
 assert not (a&b or a&c or b&c) and len(a|b|c)==1000

def test_nested_manifests_and_labeled_only_scaler():
 c=build_cohort(records())
 ms=[generate_manifest(c,3,r) for r in RATIOS]
 sets=[set(m['labeled_indices']) for m in ms]
 assert sets[0]<sets[1]<sets[2]<sets[3]
 assert [m['counts']['labeled'] for m in ms]==[40,80,160,800]
 for m in ms:
  validate_manifest(m,c);ys=[c['records'][i]['bone_age_months'] for i in m['labeled_indices']]
  assert m['target_scaler']['sample_count']==len(ys)

def test_anatomy_preserving_resize_pad_and_no_flip_or_crop():
 for size in ((100,400),(400,100),(317,211)):
  im=Image.new('L',size,255);out=build_transform('eval')(im)
  assert out.shape==(3,224,224) and torch.isfinite(out).all()
  restored=out*torch.tensor(IMAGENET_STD)[:,None,None]+torch.tensor(IMAGENET_MEAN)[:,None,None]
  mask=restored[0]>.5;ys,xs=torch.where(mask)
  assert xs.numel()>0 and (xs.max()-xs.min()+1==224 or ys.max()-ys.min()+1==224)

def test_pixels_only_and_no_patient_claim():
 c=build_cohort(records(100));assert c['model_inputs']=='pixels_only';assert c['patient_identifier_available'] is False
