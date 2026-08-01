#!/usr/bin/env python3
"""Audit official RSNA files, create private extracted data and local manifests."""
import argparse,csv,json,math,os,stat,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import numpy as np
from data_processing.rsna_bone_age import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path('/nobackup/enzez/data/rsna_bone_age/2017'));p.add_argument('--protocol-root',type=Path,default=Path('/nobackup/enzez/data/rsna_bone_age/2017/protocol'));a=p.parse_args();os.umask(0o077)
 train_zip=a.data_root/'Bone Age Training Set.zip';ann_zip=a.data_root/'Bone Age Training Set Annotations.zip';val_zip=a.data_root/'Bone Age Validation Set.zip'
 for z in (train_zip,ann_zip,val_zip):
  with zipfile.ZipFile(z) as f:bad=f.testzip();assert bad is None,(z,bad)
 extract=a.data_root/'extracted';extract.mkdir(mode=0o700,parents=True,exist_ok=True)
 for z in (train_zip,ann_zip,val_zip):
  with zipfile.ZipFile(z) as f:f.extractall(extract)
 for nested in list(extract.rglob('*.zip')):
  with zipfile.ZipFile(nested) as f:assert f.testzip() is None;f.extractall(nested.parent)
 for d,ds,fs in os.walk(extract):os.chmod(d,0o700);[os.chmod(Path(d)/x,0o600) for x in fs]
 train_csv=extract/'train.csv';validation_csv=extract/'Bone Age Validation Set/Validation Dataset.csv';assert train_csv.is_file() and validation_csv.is_file()
 with train_csv.open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==12611 and {'id','boneage','male'}<=set(rows[0])
 images={p.stem:p for p in extract.rglob('*.png')};records=[];seen=set();corrupt=[]
 for r in rows:
  iid=str(r['id']);assert iid not in seen;seen.add(iid);age=float(r['boneage']);assert math.isfinite(age);path=images.get(iid);assert path is not None,iid
  try:
   with Image.open(path) as im:im.verify()
  except Exception as e:corrupt.append((iid,str(e)));continue
  records.append({'image_id':iid,'image_path':str(path.relative_to(extract)),'bone_age_months':age,'male':str(r['male']).strip().lower() in ('true','1','male','m')})
 assert not corrupt,corrupt;assert len(records)==12611
 with validation_csv.open(newline='') as f:vrows=list(csv.DictReader(f));assert len(vrows)==1425
 validation=[]
 for r in vrows:
  iid=str(r['Image ID']);assert iid not in seen;seen.add(iid);age=float(r['Bone Age (months)']);assert math.isfinite(age);path=images.get(iid);assert path is not None,iid
  try:
   with Image.open(path) as im:im.verify()
  except Exception as e:corrupt.append((iid,str(e)));continue
  validation.append({'image_id':iid,'image_path':str(path.relative_to(extract)),'bone_age_months':age,'male':str(r['male']).strip().lower() in ('true','1','male','m')})
 assert not corrupt and len(validation)==1425
 cohort=build_cohort(records,validation);a.protocol_root.mkdir(mode=0o700,parents=True,exist_ok=True);write_json(a.protocol_root/'cohort.json',cohort)
 manifests=[]
 for seed in range(6):
  for ratio in RATIOS:
   m=generate_manifest(cohort,seed,ratio);write_json(a.protocol_root/f'ratio_{ratio:.2f}_seed_{seed}.json',m);manifests.append(m)
 for seed in range(6):
  sets=[set(generate_manifest(cohort,seed,r)['labeled_indices']) for r in RATIOS];assert sets[0]<sets[1]<sets[2]<sets[3]
 for f in a.protocol_root.iterdir():os.chmod(f,0o600)
 ages=np.asarray([r['bone_age_months'] for r in records]);summary={'policy':'A','training_images':len(records),'official_validation_images':len(validation),'official_validation_annotations_available':True,'corrupt':len(corrupt),'bone_age_unit':'months','bone_age_min':float(ages.min()),'bone_age_max':float(ages.max()),'bone_age_mean':float(ages.mean()),'bone_age_std':float(ages.std()),'quantiles':{str(q):float(np.quantile(ages,q)) for q in (0,.25,.5,.75,1)},'sex_field_available':True,'split_counts':cohort['counts'],'split_digest':cohort['split_digest'],'patient_identifier_available':False}
 write_json(a.protocol_root/'audit_summary.json',summary);os.chmod(a.protocol_root/'audit_summary.json',0o600);print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
