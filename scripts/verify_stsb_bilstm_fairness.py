#!/usr/bin/env python3
"""Verify identical standard target initialization and distinct trained anchors."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from training.stsb_ssl import model_state_hash, seed_everything, supervised_bilstm_checkpoint
from training.supervised_stsb import construct

ROOT=Path(__file__).resolve().parents[1]


def standard_hash(seed):
    seed_everything(seed);rapl,_,_=construct("bilstm_glove",torch.device("cpu"));rapl_hash=model_state_hash(rapl)
    del rapl
    seed_everything(seed);hpl,_,_=construct("bilstm_glove",torch.device("cpu"));hpl_hash=model_state_hash(hpl)
    return rapl_hash,hpl_hash


def main():
    records=[]
    for seed in range(6):
        rapl_hash,hpl_hash=standard_hash(seed)
        if rapl_hash!=hpl_hash:raise RuntimeError(f"Target initialization mismatch for seed {seed}")
        for ratio in (.05,.10,.20):
            manifest=ROOT/f"data_processing/splits/stsb_ratio_{ratio:.2f}_seed_{seed}.json"
            args=SimpleNamespace(seed=seed,labeled_ratio=ratio,manifest=manifest)
            checkpoint=supervised_bilstm_checkpoint(args)
            anchor_state=torch.load(checkpoint,map_location="cpu",weights_only=True)["model_state"]
            # Hash the exact checkpoint state without constructing a second target.
            seed_everything(seed);anchor,_,_=construct("bilstm_glove",torch.device("cpu"));anchor.load_state_dict(anchor_state);anchor.requires_grad_(False)
            anchor_hash=model_state_hash(anchor)
            if anchor_hash==rapl_hash or any(p.requires_grad for p in anchor.parameters()):raise RuntimeError("Anchor fairness/freeze check failed")
            records.append({"seed":seed,"ratio":ratio,"rapl_target_init_hash":rapl_hash,"hpl_target_init_hash":hpl_hash,"anchor_hash":anchor_hash,"anchor_checkpoint":str(checkpoint)})
    print(json.dumps({"status":"pass","checks":len(records),"records":records},sort_keys=True))


if __name__=="__main__":main()
