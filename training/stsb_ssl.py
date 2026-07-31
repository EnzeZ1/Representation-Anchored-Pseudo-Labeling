"""Formal RAPL and official-semantics HPL training under stsb-benchmark-v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from itertools import cycle
from pathlib import Path

import higher
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data_processing.stsb import (
    STSBTextDataset, dataloader_generator, file_sha256, load_cohort,
    load_manifest, loader_metadata, seed_dataloader_worker,
)
from data_processing.stsb_ssl import SSLPairCollator, STSBUnlabeledViews
from models.hpl_uncertainty import UncertaintyLearner
from training.supervised_stsb import (
    COHORT_PATH, construct, evaluate, model_forward, runtime_metadata, to_device,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
SUPERVISED = ROOT / "artifacts/supervised_baselines/stsb"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def features(model, batch, backbone):
    if backbone == "bilstm_glove":
        return model.forward_features(batch["sentence1"], batch["sentence2"], batch["mask1"], batch["mask2"])
    return model.forward_features(batch["input_ids"], batch["attention_mask"])


def make_loaders(args, collator):
    cohort = load_cohort(COHORT_PATH)
    manifest = load_manifest(args.manifest, cohort)
    if manifest["protocol_version"] != "stsb-benchmark-v1" or manifest["augmentation_version"] != "rapl-text-augmentation-v1":
        raise RuntimeError("Frozen STS-B protocol identity mismatch")
    if int(manifest["seed"]) != args.seed or not math.isclose(float(manifest["labeled_ratio"]), args.labeled_ratio):
        raise RuntimeError("Manifest seed/ratio mismatch")
    mean, std = float(manifest["label_scaler"]["mean"]), float(manifest["label_scaler"]["std"])
    datasets = {
        "labeled": STSBTextDataset(cohort, manifest["labeled_indices"], mean, std),
        "unlabeled": STSBUnlabeledViews(cohort, manifest["unlabeled_indices"], args.seed),
        "validation": STSBTextDataset(cohort, manifest["splits"]["validation"], mean, std),
        "test": STSBTextDataset(cohort, manifest["splits"]["test"], mean, std),
    }
    loaders = {}
    for role, dataset in datasets.items():
        shuffle = role in {"labeled", "unlabeled"}
        loaders[role] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
            collate_fn=SSLPairCollator(collator) if role == "unlabeled" else collator,
            worker_init_fn=seed_dataloader_worker,
            generator=dataloader_generator(args.seed, role),
        )
    protocol = {
        "protocol_version": manifest["protocol_version"],
        "augmentation_version": manifest["augmentation_version"],
        "cohort_sha256": cohort["cohort_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_payload_sha256": manifest["manifest_sha256"],
        "counts": manifest["counts"], "label_scaler": manifest["label_scaler"],
    }
    return cohort, manifest, datasets, loaders, mean, std, protocol


def target_optimization(model, backbone, epochs, steps):
    if backbone == "bilstm_glove":
        encoder = [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
        feat = torch.optim.Adam(encoder, lr=1e-4, weight_decay=1e-5)
        head = torch.optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=1e-5)
        return feat, head, None, None, {"optimizer":"Adam","encoder_lr":1e-4,"head_lr":1e-3,"weight_decay":1e-5,"scheduler":None}
    feat = torch.optim.AdamW(model.backbone.parameters(), lr=2e-5, weight_decay=0.01)
    head = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    total = epochs * steps; warmup = int(total * .10)
    sf = get_linear_schedule_with_warmup(feat, warmup, total)
    sh = get_linear_schedule_with_warmup(head, warmup, total)
    return feat, head, sf, sh, {"optimizer":"AdamW","encoder_lr":2e-5,"head_lr":1e-4,"weight_decay":.01,"scheduler":"linear","warmup_ratio":.10,"gradient_clipping":1.0}


@torch.no_grad()
def fit_probe(anchor, loader, backbone, device):
    anchor.eval(); values=[]; targets=[]
    for batch in loader:
        batch=to_device(batch,device); values.append(features(anchor,batch,backbone).double().cpu()); targets.append(batch["target"].double().cpu())
    x=torch.cat(values); y=torch.cat(targets)
    # Minimum-norm closed-form least squares in the dual; equivalent to an
    # unregularized affine linear probe and tractable for the 8192-D BiLSTM.
    xm=x.mean(0); ym=y.mean(); xc=x-xm; yc=y-ym
    alpha=torch.linalg.lstsq(xc @ xc.T, yc[:,None]).solution[:,0]
    weight=xc.T @ alpha; bias=ym-xm@weight
    probe=nn.Linear(x.shape[1],1,dtype=torch.float64)
    probe.weight.copy_(weight[None]); probe.bias.copy_(bias[None]); probe.requires_grad_(False)
    return probe.float().to(device).eval()


def supervised_bilstm_checkpoint(args):
    path=SUPERVISED/"bilstm_glove"/f"ratio_{args.labeled_ratio:.2f}"/f"seed_{args.seed}"/"best.pt"
    metadata=json.loads((path.parent/"metadata.json").read_text())
    if not (metadata["checkpoint_reloaded"] is True and metadata["test_used_for_selection"] is False and metadata["test_model_inference_count"]==1):
        raise RuntimeError("BiLSTM supervised anchor source failed integrity checks")
    if Path(metadata["manifest_path"]).resolve()!=args.manifest.resolve() or metadata["manifest_sha256"]!=file_sha256(args.manifest):
        raise RuntimeError("BiLSTM supervised anchor manifest mismatch")
    return path


def construct_method(args, device):
    model, collator, identity=construct(args.backbone,device)
    anchor=None; probe=None; provenance=None
    if args.method=="rapl":
        anchor, _, anchor_identity=construct(args.backbone,device)
        if args.backbone=="bilstm_glove":
            source=supervised_bilstm_checkpoint(args)
            state=torch.load(source,map_location=device,weights_only=True)["model_state"]
            model.load_state_dict(state); anchor.load_state_dict(state)
            provenance={"source":"supervised_best_checkpoint_same_seed_ratio","path":str(source),"manifest_sha256":file_sha256(args.manifest)}
        else:
            provenance={"source":"independent_generic_pretrained_copy","identifier":"FacebookAI/roberta-base"}
        anchor.eval(); anchor.requires_grad_(False)
        if any(p.requires_grad for p in anchor.parameters()): raise RuntimeError("RAPL anchor is not frozen")
        identity["anchor"]=anchor_identity
    return model,anchor,collator,identity,provenance


def hpl_meta_step(model, uncertainty, opt_head, opt_unc, labeled, weak, strong, meta, backbone, lambda2):
    opt_unc.zero_grad(set_to_none=True)
    with higher.innerloop_ctx(model.head, opt_head) as (fhead, diffopt):
        with torch.no_grad():
            fl=features(model,labeled,backbone); fw=features(model,weak,backbone); fs=features(model,strong,backbone); fm=features(model,meta,backbone)
            if backbone == "roberta_base":
                fl=model.dropout(fl); fw=model.dropout(fw); fs=model.dropout(fs); fm=model.dropout(fm)
        pl=fhead(fl).squeeze(-1); pw=fhead(fw).squeeze(-1).detach(); ps=fhead(fs).squeeze(-1)
        ui=torch.stack((ps.detach()-pw,ps.detach()),dim=-1)
        inner=F.mse_loss(pl,labeled["target"])+torch.mean(torch.exp(-uncertainty(ui))/2*(ps-pw).pow(2).unsqueeze(-1))
        diffopt.step(inner)
        pm=fhead(fm).squeeze(-1)
        umi=torch.stack(((pm-meta["target"]).detach(),pm.detach()),dim=-1)
        loss=F.mse_loss(pm,meta["target"])-lambda2*uncertainty(umi).mean()
        loss.backward(); opt_unc.step()
    return float(loss.detach())


def train_epoch(args, model, anchor, probe, uncertainty, loaders, optimizers, schedulers, device, epoch):
    feat_opt,head_opt,unc_opt,meta_head_opt=optimizers; sf,sh=schedulers
    model.train(); loaders["unlabeled"].dataset.set_epoch(epoch)
    lit=cycle(loaders["labeled"]); mit=cycle(loaders["labeled"])
    totals={"total":0.,"supervised":0.,"unlabeled":0.,"meta":[]}; steps=len(loaders["unlabeled"])
    for step, views in enumerate(loaders["unlabeled"]):
        labeled=to_device(next(lit),device); meta=to_device(next(mit),device)
        weak=to_device(views["weak"],device); strong=to_device(views["strong"],device)
        if args.method=="hpl" and step%5==0:
            totals["meta"].append(hpl_meta_step(model,uncertainty,meta_head_opt,unc_opt,labeled,weak,strong,meta,args.backbone,1.0))
        feat_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
        pl=model_forward(model,labeled,args.backbone); sup=F.mse_loss(pl,labeled["target"])
        with torch.no_grad(): pseudo=model_forward(model,weak,args.backbone).detach()
        strong_pred=model_forward(model,strong,args.backbone)
        if args.method=="rapl":
            with torch.no_grad(): anchor_pred=probe(features(anchor,weak,args.backbone)).squeeze(-1)
            trust=(1/(1+(pseudo-anchor_pred).abs())).detach(); unl=(trust*(strong_pred-pseudo).pow(2)).mean()
        else:
            with torch.no_grad():
                ui=torch.stack((strong_pred.detach()-pseudo,strong_pred.detach()),dim=-1)
                weight=torch.exp(-uncertainty(ui))/2
            unl=(weight.squeeze(-1)*(strong_pred-pseudo).pow(2)).mean()
        loss=sup+unl
        loss.backward()
        if args.backbone=="roberta_base": torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        head_opt.step(); feat_opt.step()
        if sf: sf.step(); sh.step()
        totals["total"]+=float(loss.detach());totals["supervised"]+=float(sup.detach());totals["unlabeled"]+=float(unl.detach())
    return {k:(sum(v)/len(v) if isinstance(v,list) and v else (v/steps if not isinstance(v,list) else None)) for k,v in totals.items()}


def save_checkpoint(path,args,model,anchor,probe,uncertainty,optimizers,schedulers,epoch,val_mse,val_mae,config,protocol):
    payload={"checkpoint_version":"stsb-ssl-v1","method":args.method,"model_state":model.state_dict(),"epoch":epoch,"validation_mse":val_mse,"validation_mae":val_mae,"optimizers":[x.state_dict() if x else None for x in optimizers],"schedulers":[x.state_dict() if x else None for x in schedulers],"manifest_sha256":protocol["manifest_sha256"],"config":config}
    if args.method=="rapl": payload.update(anchor_state=anchor.state_dict(),probe_state=probe.state_dict())
    else: payload.update(uncertainty_state=uncertainty.state_dict())
    torch.save(payload,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--method",choices=("rapl","hpl"),required=True);p.add_argument("--backbone",choices=("bilstm_glove","roberta_base"),required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output-dir",type=Path);p.add_argument("--seed",type=int,required=True);p.add_argument("--labeled-ratio",type=float,required=True);p.add_argument("--epochs",type=int);p.add_argument("--batch-size",type=int);p.add_argument("--num-workers",type=int,default=2);p.add_argument("--preflight",action="store_true");args=p.parse_args()
    args.epochs=args.epochs or (200 if args.backbone=="bilstm_glove" else 10);args.batch_size=args.batch_size or (32 if args.backbone=="bilstm_glove" else 16)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("Worker must see one local cuda:0")
    seed_everything(args.seed);device=torch.device("cuda:0");model,anchor,collator,identity,provenance=construct_method(args,device)
    cohort,manifest,datasets,loaders,mean,std,protocol=make_loaders(args,collator)
    if args.method=="rapl": probe=fit_probe(anchor,loaders["labeled"],args.backbone,device);uncertainty=None
    else: probe=None;uncertainty=UncertaintyLearner().to(device)
    feat_opt,head_opt,sf,sh,opt_cfg=target_optimization(model,args.backbone,args.epochs,len(loaders["unlabeled"]));unc_opt=torch.optim.Adam(uncertainty.parameters(),lr=1e-4,weight_decay=1e-5) if uncertainty else None
    meta_head_opt=(head_opt if args.method=="hpl" and args.backbone=="bilstm_glove" else torch.optim.Adam(model.head.parameters(),lr=1e-4,weight_decay=1e-5) if args.method=="hpl" else None)
    method_cfg={"lambda_u":1.0,"tau":1.0,"trust_formula":"1/(1+abs(target_pseudo-frozen_probe))"} if args.method=="rapl" else {"w_ulb":1.0,"lambda2":1.0,"uncertainty_update_frequency":5,"uncertainty_lr":1e-4,"uncertainty_weight_decay":1e-5,"bilevel_head_optimizer":"official HPL Adam","bilevel_head_lr":1e-3 if args.backbone=="bilstm_glove" else 1e-4}
    config={"dataset":"STS-B-DIR","method":args.method,"backbone":args.backbone,"epochs":args.epochs,"batch_size":args.batch_size,"precision":"float32","selection_metric":"lowest validation MSE in original STS-B score units","protocol":protocol,"model":identity,"optimization":opt_cfg,"method_configuration":method_cfg,"anchor_provenance":provenance,"official_hpl_upstream_commit":"89f9f8bd467a0d3f81a8ada8708c3fe4fe31ca20" if args.method=="hpl" else None}
    if args.preflight:
        labeled=to_device(next(iter(loaders["labeled"])),device);views=next(iter(loaders["unlabeled"]));weak=to_device(views["weak"],device);strong=to_device(views["strong"],device)
        with torch.no_grad(): out=[model_forward(model,x,args.backbone).shape for x in (labeled,weak,strong)]
        print(json.dumps({"status":"pass","method":args.method,"backbone":args.backbone,"forward_shapes":[list(x) for x in out],"anchor_frozen":None if anchor is None else not any(p.requires_grad for p in anchor.parameters()),"uncertainty_trainable":None if uncertainty is None else any(p.requires_grad for p in uncertainty.parameters()),"probe_labeled_count":manifest["counts"]["labeled"] if probe else None,"test_model_inference_count":0,"config":config},sort_keys=True));return
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"metrics.json").exists():raise FileExistsError(output)
    write_json(output/"config.json",config);started=time.time();torch.cuda.reset_peak_memory_stats(device);history=[];best=math.inf;best_epoch=None
    for epoch in range(1,args.epochs+1):
        losses=train_epoch(args,model,anchor,probe,uncertainty,loaders,(feat_opt,head_opt,unc_opt,meta_head_opt),(sf,sh),device,epoch)
        vmse,vmae,vr2=evaluate(model,loaders["validation"],mean,std,device,args.backbone);improved=vmse<best
        if improved:best=vmse;best_epoch=epoch;save_checkpoint(output/"best.pt",args,model,anchor,probe,uncertainty,(feat_opt,head_opt,unc_opt,meta_head_opt),(sf,sh),epoch,vmse,vmae,config,protocol)
        history.append({"epoch":epoch,"train_total_mse":losses["total"],"train_sup_mse":losses["supervised"],"train_unlabeled_mse":losses["unlabeled"],"train_meta_loss":losses["meta"],"validation_mse":vmse,"validation_mae":vmae,"validation_r2":vr2,"best_so_far":int(improved),"elapsed_seconds":time.time()-started})
        print(f"epoch={epoch}/{args.epochs} loss={losses['total']:.8f} validation_mse={vmse:.6f} best_epoch={best_epoch}",flush=True)
    with (output/"history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    restored=torch.load(output/"best.pt",map_location=device,weights_only=True);model.load_state_dict(restored["model_state"])
    if restored["epoch"]!=best_epoch or not math.isclose(float(restored["validation_mse"]),best,abs_tol=1e-12):raise RuntimeError("Restored selection metadata mismatch")
    if args.method=="rapl":anchor.load_state_dict(restored["anchor_state"]);probe.load_state_dict(restored["probe_state"])
    else:uncertainty.load_state_dict(restored["uncertainty_state"])
    tmse,tmae,tr2,pred,target,indices,identifiers=evaluate(model,loaders["test"],mean,std,device,args.backbone,predictions=True)
    np.savez_compressed(output/"test_predictions.npz",cohort_indices=indices,stable_identifiers=identifiers,predictions_score_units=pred,targets_score_units=target)
    metrics={"best_epoch":best_epoch,"best_validation_mse_score_units":best,"test_mse_score_units":tmse,"test_mae_score_units":tmae,"test_r2":tr2,"checkpoint_reloaded":True,"test_used_for_selection":False,"test_model_inference_count":1}
    write_json(output/"metrics.json",metrics)
    write_json(output/"metadata.json",{"status":"complete","method":args.method,"dataset":"STS-B-DIR","backbone":args.backbone,"seed":args.seed,"labeled_ratio":args.labeled_ratio,"checkpoint_path":str(output/"best.pt"),"checkpoint_selection":"lowest validation MSE in original STS-B score units","checkpoint_reloaded":True,"restored_epoch_verified":True,"restored_validation_metric_verified":True,"test_used_for_selection":False,"test_model_inference_count":1,**protocol,"anchor_provenance":provenance,"runtime_seconds":time.time()-started,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(device),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"process_local_device":"cuda:0","runtime":runtime_metadata()})
    print(json.dumps(metrics,sort_keys=True),flush=True)


if __name__=="__main__":main()
