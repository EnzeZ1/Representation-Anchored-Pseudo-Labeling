#!/usr/bin/env python3
"""Audit official KonIQ-10k sources and write the annotated canonical cohort."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from PIL import Image
from data_processing.koniq10k import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--split-csv',type=Path,required=True);p.add_argument('--darus-scores',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 author=pd.read_csv(a.split_csv);darus=pd.read_csv(a.darus_scores,sep=None,engine='python'); merged=author.merge(darus,on='image_name',suffixes=('_author','_darus'),validate='one_to_one')
 if len(merged)!=10073 or set(author.image_name)!=set(darus.image_name):raise RuntimeError('Official metadata identifier disagreement')
 if not np.allclose(merged.MOS_author,merged.MOS_zscore,rtol=0,atol=1e-8) or not np.allclose(merged.SD_author,merged.SD_darus,rtol=0,atol=1e-8):raise RuntimeError('Official MOS/SD disagreement')
 names=set(author.image_name);files={x.name:x for x in a.images.glob('*.jpg')};missing=sorted(names-files.keys());extra=sorted(files.keys()-names)
 if missing:raise RuntimeError(f'missing annotated images: {missing[:10]}')
 bad=[]
 for name in sorted(names):
  try:
   with Image.open(files[name]) as im:im.load();size=im.size
   if size!=(512,384):bad.append((name,size))
  except Exception as e:bad.append((name,str(e)))
 if bad:raise RuntimeError(f'decode/size failures: {bad[:10]}')
 records=[]
 for row in author.sort_values('image_name').itertuples():records.append({'image_name':row.image_name,'split':row.set,'mos':float(row.MOS),'sd':float(row.SD),'rating_distribution':[float(row.c1),float(row.c2),float(row.c3),float(row.c4),float(row.c5)],'rating_count':int(row.c_total)})
 cohort={'protocol_version':PROTOCOL_VERSION,'transform_version':TRANSFORM_VERSION,'records':records,'cohort_size':len(records),'cohort_sha256':cohort_digest(records),'membership_sha256':membership_digest(records),'provenance':{'archive_md5':'dc213332574e8431a86c7ff34e1fa924','archive_sha256':sha256(a.archive),'split_metadata_sha256':sha256(a.split_csv),'darus_metadata_sha256':sha256(a.darus_scores),'archive_only_images_excluded':extra,'archive_only_count':len(extra),'annotated_cohort_policy':'intersection of original-author split metadata and DaRUS score identifiers'}}
 validate_cohort(cohort);write_json(a.output,cohort);print(json.dumps({'cohort_sha256':cohort['cohort_sha256'],'membership_sha256':cohort['membership_sha256'],'annotated':len(records),'archive_only_excluded':len(extra),'splits':author['set'].value_counts().to_dict(),'mos_range':[float(author.MOS.min()),float(author.MOS.max())],'sd_range':[float(author.SD.min()),float(author.SD.max())]},sort_keys=True))
if __name__=='__main__':main()
