#!/usr/bin/env python3
"""Generate protected local patient manifests only after release validation."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from data_processing.chexchonet import audit_release, discover_metadata, load_records
from data_processing.chexchonet_protocol import RATIOS, SEEDS, generate_patient_manifest, save_manifest, validate_nested

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    audit=audit_release(a.data_root, decode=True)
    if not audit.ready: raise RuntimeError("Authorized release failed full readiness audit")
    if audit.nonpositive_lvidd / audit.metadata_rows > .01: raise RuntimeError("Nonpositive LVIDd exceeds 1%; protocol generation stopped")
    records=load_records(discover_metadata(a.data_root))
    for seed in SEEDS:
        manifests={ratio:generate_patient_manifest(records,seed=seed,ratio=ratio) for ratio in RATIOS}; validate_nested(manifests)
        for ratio,manifest in manifests.items(): save_manifest(manifest,a.output/f"chexchonet_lvidd_ratio_{ratio:.2f}_seed_{seed}.json")
if __name__=="__main__": main()
