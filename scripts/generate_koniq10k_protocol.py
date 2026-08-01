#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.koniq10k import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();c=load_cohort(a.cohort);ms=[]
 for s in range(6):
  for r in RATIOS:
   m=generate_manifest(c,s,r);write_json(a.output_dir/f'koniq10k_ratio_{r:.2f}_seed_{s}.json',m);ms.append(m)
 validate_family(c,ms);print(json.dumps({'status':'pass','manifests':24,'cohort_sha256':c['cohort_sha256']}))
if __name__=='__main__':main()
