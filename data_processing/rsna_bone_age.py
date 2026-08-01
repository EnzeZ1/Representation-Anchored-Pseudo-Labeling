"""Canonical RSNA 2017 pediatric bone-age protocol and anatomy-preserving views."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF,InterpolationMode

PROTOCOL_VERSION="rsna-bone-age-benchmark-v1"
TRANSFORM_VERSION="rsna-bone-age-anatomy-preserving-v1"
SPLIT_SEED=20260801
RATIOS=(.05,.10,.20,1.00)
IMAGENET_MEAN=(.485,.456,.406);IMAGENET_STD=(.229,.224,.225)

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def load_json(path):return json.loads(Path(path).read_text())
def _stratum(r):return f"{'M' if r['male'] else 'F'}:{int(float(r['bone_age_months']))//12}"
def _stratified_exact(records,targets):
 groups={}
 for i,r in enumerate(records):groups.setdefault(_stratum(r),[]).append(i)
 rng=random.Random(SPLIT_SEED);ordered=[]
 for key in sorted(groups):rng.shuffle(groups[key]);ordered.extend((key,i) for i in groups[key])
 out={k:[] for k in targets};remaining=dict(targets);counts={k:{} for k in targets}
 for key,i in ordered:
  choices=[k for k in targets if remaining[k]>0]
  k=max(choices,key=lambda x:(remaining[x]/targets[x],-(counts[x].get(key,0))))
  out[k].append(i);remaining[k]-=1;counts[k][key]=counts[k].get(key,0)+1
 for k in out:out[k].sort(key=lambda i:str(records[i]['image_id']))
 sets=[set(v) for v in out.values()]
 assert all(not (sets[i]&sets[j]) for i in range(len(sets)) for j in range(i+1,len(sets)))
 return out
def fixed_policy_b_split(records):
 n=len(records);return _stratified_exact(records,{'train':math.floor(.8*n),'validation':math.floor(.1*n),'test':n-math.floor(.8*n)-math.floor(.1*n)})
def build_cohort(records,official_validation_records=None):
 records=sorted(records,key=lambda r:str(r['image_id']))
 if official_validation_records is None:splits=fixed_policy_b_split(records);policy='B'
 else:
  n=len(records);internal=_stratified_exact(records,{'train':math.floor(.9*n),'validation':n-math.floor(.9*n)});test_records=sorted(official_validation_records,key=lambda r:str(r['image_id']));offset=n;records+=test_records;splits={**internal,'test':list(range(offset,offset+len(test_records)))};policy='A'
 body={'protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'split_policy':policy,'split_seed':SPLIT_SEED,'model_inputs':'pixels_only','records':records,'splits':splits,'counts':{k:len(v) for k,v in splits.items()},'patient_identifier_available':False}
 body['split_digest']=digest({k:[records[i]['image_id'] for i in v] for k,v in splits.items()});body['cohort_digest']=digest(records);return body
def generate_manifest(cohort,seed,ratio):
 train=list(cohort['splits']['train']);order=train.copy();random.Random(int(seed)).shuffle(order);n=len(train) if ratio==1 else math.floor(len(train)*float(ratio));labeled=order[:n];unlabeled=order[n:];y=np.asarray([cohort['records'][i]['bone_age_months'] for i in labeled],dtype=np.float64)
 body={'protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'split_policy':cohort['split_policy'],'split_seed':SPLIT_SEED,'split_digest':cohort['split_digest'],'cohort_digest':cohort['cohort_digest'],'seed':int(seed),'labeled_ratio':float(ratio),'model_inputs':'pixels_only','labeled_indices':labeled,'unlabeled_indices':unlabeled,'ordered_identifiers':{'labeled':[cohort['records'][i]['image_id'] for i in labeled],'unlabeled':[cohort['records'][i]['image_id'] for i in unlabeled]},'counts':{'labeled':n,'unlabeled':len(unlabeled),'train':len(train),'validation':len(cohort['splits']['validation']),'test':len(cohort['splits']['test'])},'target_scaler':{'mean':float(y.mean()),'std':float(y.std()+1e-6),'sample_count':n,'source':'labeled training examples only'}}
 body['manifest_digest']=digest(body);return body
def validate_manifest(m,c):
 saved=m['manifest_digest'];b=dict(m);b.pop('manifest_digest');assert saved==digest(b);assert m==generate_manifest(c,m['seed'],m['labeled_ratio'])

def resize_pad(im,size=224):
 w,h=im.size;s=size/max(w,h);nw,nh=max(1,round(w*s)),max(1,round(h*s));im=TF.resize(im,[nh,nw],InterpolationMode.BILINEAR,antialias=True);left=(size-nw)//2;top=(size-nh)//2;return TF.pad(im,[left,top,size-nw-left,size-nh-top],fill=0)
class AnatomyTransform:
 def __init__(self,mode='eval'):self.mode=mode
 def __call__(self,im):
  im=im.convert('L')
  if self.mode!='eval':
   strong=self.mode=='strong';deg=7 if strong else 3;trans=.04 if strong else .02;scale=(.95,1.05) if strong else (.98,1.02);w,h=im.size
   im=TF.affine(im,random.uniform(-deg,deg),[round(random.uniform(-trans,trans)*w),round(random.uniform(-trans,trans)*h)],random.uniform(*scale),0,InterpolationMode.BILINEAR,fill=0)
  im=resize_pad(im).convert('RGB');return TF.normalize(TF.to_tensor(im),IMAGENET_MEAN,IMAGENET_STD)
def build_transform(mode='eval'):return AnatomyTransform(mode)
