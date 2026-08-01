#!/usr/bin/env python3
"""Declarative formal plan. Refuses launch until protected manifests exist."""
import argparse, json
from pathlib import Path

METHODS=("supervised_step_matched","rapl","hpl"); BACKBONES=("resnet50","dinov2_vits14"); RATIOS=(.05,.10,.20); SEEDS=range(6)
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest-root",type=Path,required=True);p.add_argument("--print-plan",action="store_true");a=p.parse_args()
    jobs=[{"target":"lvidd","method":m,"backbone":b,"ratio":r,"seed":s} for r in RATIOS for m in METHODS for b in BACKBONES for s in SEEDS]
    jobs += [{"target":"lvidd","method":"supervised","backbone":b,"ratio":1.0,"seed":s} for b in BACKBONES for s in SEEDS]
    missing=[str(a.manifest_root/f"chexchonet_lvidd_ratio_{j['ratio']:.2f}_seed_{j['seed']}.json") for j in jobs if not (a.manifest_root/f"chexchonet_lvidd_ratio_{j['ratio']:.2f}_seed_{j['seed']}.json").is_file()]
    if missing: raise RuntimeError(f"Protected validated manifests unavailable ({len(set(missing))} missing); refusing launch")
    if a.print_plan: print(json.dumps({"job_count":len(jobs),"jobs":jobs},indent=2))
    else: raise RuntimeError("Formal launch is approval-gated; use --print-plan for audit only")
if __name__=="__main__":main()
