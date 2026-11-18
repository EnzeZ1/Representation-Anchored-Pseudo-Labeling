"""Canonical gpu03 Uniform anchor-frequency training, isolated from HPL."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from training.train import make_data_utkface

EPOCHS = 30
STEPS = 563
DIAGNOSTIC_EPOCHS = {1, 5, 10, 20, 30}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_runtime(backbone: str, device: torch.device):
    if backbone == "resnet50":
        model = ResNet50Regressor(pretrained=True, dropout=.20).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    elif backbone == "dinov2":
        model = DINOv2Regressor(size="small", dropout=.20).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4},
        ], weight_decay=.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    else:
        raise ValueError(backbone)
    return model, optimizer, scheduler


def uniform_losses(model, xuw, xus, xl=None, yl=None):
    """The single loss-construction path used by preflight and formal training."""
    with torch.no_grad():
        pseudo = model(xuw).detach()
    strong = model(xus)
    r = torch.ones_like(pseudo)
    loss_u = (r * (strong - pseudo).square()).mean()
    if xl is None:
        return {"total": loss_u, "supervised": None, "unlabeled": loss_u, "r": r}
    loss_s = F.mse_loss(model(xl), yl)
    return {"total": loss_s + loss_u, "supervised": loss_s, "unlabeled": loss_u, "r": r}


def clip_and_step(model, optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(norm)


def evaluate(model, loader, mean, std, device, predictions=False):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            ps.append((model(x.to(device, non_blocking=True))*std+mean).cpu())
            ys.append((y*std+mean).cpu())
    p=torch.cat(ps).numpy(); y=torch.cat(ys).numpy()
    mae=float(np.mean(np.abs(p-y))); r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))
    return (mae,r2,p,y) if predictions else (mae,r2)


def next_labeled(state, loader):
    try: batch=next(state["iterator"])
    except StopIteration:
        state["iterator"]=iter(loader); state["traversals"] += 1; batch=next(state["iterator"])
    state["consumed"] += 1
    return batch


def runtime_description(model, optimizer, scheduler):
    names={id(p):n for n,p in model.named_parameters()}
    groups=[]
    for i,g in enumerate(optimizer.param_groups):
        ps=list(g["params"])
        groups.append({"index":i,"tensor_count":len(ps),"parameter_count":sum(p.numel() for p in ps),
                       "representative_names":[names[id(p)] for p in ps[:5]],"lr":g["lr"],"weight_decay":g["weight_decay"]})
    return {"model_class":type(model).__name__,"backbone_class":type(model.backbone).__name__,
            "head_class":type(model.head).__name__,"total_parameters":sum(p.numel() for p in model.parameters()),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "dropout_locations":[n for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout)],
            "dropout_p":[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)],
            "optimizer_class":type(optimizer).__name__,"optimizer_groups":groups,
            "scheduler_class":type(scheduler).__name__,"gradient_clipping":{"function":"torch.nn.utils.clip_grad_norm_","max_norm":5.0}}


def args_for_data(a):
    a.dataset="utkface"; a.method="probe"; a.data_dir=str(Path(__file__).resolve().parents[1]/"data/utkface_all")
    a.labeled_ratio=.05; a.img_size=224; a.batch_size=32; a.workers=4; a.pretrained=True
    a.probe_backbone=a.backbone; a.augmentation_intervention="weak_strong"; a.labeled_anchor_mode="canonical"
    a.benchmark_manifest=str(Path(__file__).resolve().parents[1]/f"data_processing/splits/utkface_ratio_0.05_seed_{a.seed}.json")
    return a


def preflight(backbone: str, output: Path):
    device=torch.device("cuda"); seed_all(918273)
    a=args_for_data(argparse.Namespace(backbone=backbone,seed=0,dino="s"))
    lab,unlab,_,_,mean,std=make_data_utkface(a)
    model,opt,sched=build_runtime(backbone,device)
    xl,yl=next(iter(lab)); xuw,xus=next(iter(unlab))
    xl=xl.to(device);yl=yl.to(device);xuw=xuw.to(device);xus=xus.to(device)
    anchor=uniform_losses(model,xuw,xus,xl,yl); non=uniform_losses(model,xuw,xus)
    opt.zero_grad(set_to_none=True); anchor["total"].backward()
    dry_grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.zero_grad(set_to_none=True)
    checks={"anchor_identity":torch.allclose(anchor["total"],anchor["supervised"]+anchor["unlabeled"]),
            "non_anchor_identity":torch.allclose(non["total"],non["unlabeled"]),
            "r_unique":torch.unique(anchor["r"]).cpu().tolist(),"lambda_u":1.0,
            "loader_lengths":{"labeled":len(lab),"unlabeled":len(unlab)},
            "dry_clip_grad_norm":float(dry_grad_norm),"optimizer_step_performed":False,
            "runtime":runtime_description(model,opt,sched)}
    checks["pass"]=all((checks["anchor_identity"],checks["non_anchor_identity"],checks["r_unique"]==[1.0],len(unlab)==STEPS))
    opt.zero_grad(set_to_none=True); del model,opt,sched; torch.cuda.empty_cache(); atomic_json(output,checks)
    if not checks["pass"]: raise RuntimeError(checks)


def train(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    status={"training_status":"running","final_evaluation_status":"pending","analysis_status":"pending"}; atomic_json(out/"status.json",status)
    seed_all(a.seed); device=torch.device("cuda")
    a=args_for_data(a); lab,unlab,val,test,mean,std=make_data_utkface(a)
    assert len(unlab)==STEPS and len(lab)==29
    model,opt,sched=build_runtime(a.backbone,device); runtime=runtime_description(model,opt,sched)
    label_state={"iterator":iter(lab),"consumed":0,"traversals":1}; best=float("inf"); best_epoch=None; rows=[]
    metadata={"method":"uniform","backbone":a.backbone,"seed":a.seed,"k":a.k,"attempt_number":a.attempt,
              "config_hash":a.config_hash,"manifest_path":a.benchmark_manifest,"cohort_digest":a.benchmark_metadata["cohort_sha256"],
              "scaler_mean":mean,"scaler_std":std,**runtime,"lambda_u":1.0,"r_mode":"constant_one",
              "epochs":EPOCHS,"steps_per_epoch":STEPS,"total_optimizer_steps":EPOCHS*STEPS,
              "training_status":"running","final_evaluation_status":"pending","analysis_status":"pending"}
    atomic_json(out/"metadata.json",metadata)
    for epoch in range(1,EPOCHS+1):
        model.train(); anchors=0; unlabeled=0; start_consumed=label_state["consumed"]; start_traversals=label_state["traversals"]
        sums={"total":0.,"supervised":0.,"unlabeled":0.}
        for t,(xuw,xus) in enumerate(unlab):
            assert t<STEPS; xuw=xuw.to(device,non_blocking=True);xus=xus.to(device,non_blocking=True)
            if t%a.k==0:
                xl,yl=next_labeled(label_state,lab); xl=xl.to(device,non_blocking=True);yl=yl.to(device,non_blocking=True)
                losses=uniform_losses(model,xuw,xus,xl,yl); anchors+=1; sums["supervised"]+=float(losses["supervised"].detach())
            else: losses=uniform_losses(model,xuw,xus)
            assert torch.all(losses["r"]==1); clip_and_step(model,opt,losses["total"]); unlabeled+=1
            sums["total"]+=float(losses["total"].detach());sums["unlabeled"]+=float(losses["unlabeled"].detach())
        assert unlabeled==STEPS and anchors==math.ceil(STEPS/a.k)
        val_mae,val_r2=evaluate(model,val,mean,std,device); improved=val_mae<best
        if improved:
            best=val_mae;best_epoch=epoch;torch.save({"model":model.state_dict(),"epoch":epoch,"validation_mae":best},out/"best.pt")
        if epoch in DIAGNOSTIC_EPOCHS: torch.save({"model":model.state_dict(),"epoch":epoch},out/f"epoch_{epoch}.pt")
        sched.step()
        rows.append({"epoch":epoch,"loss_total":sums["total"]/STEPS,"loss_supervised_anchor_mean":sums["supervised"]/anchors,
                     "loss_unlabeled":sums["unlabeled"]/STEPS,"validation_mae":val_mae,"validation_r2":val_r2,
                     "direct_supervised_anchors":anchors,"labeled_batches_consumed":label_state["consumed"]-start_consumed,
                     "labeled_loader_traversals":label_state["traversals"]-start_traversals,"cumulative_labeled_batches":label_state["consumed"],
                     "unlabeled_batches_consumed":unlabeled,"lr":[g["lr"] for g in opt.param_groups]})
        with (out/"history.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    metadata.update(training_status="complete",best_validation_epoch=best_epoch,best_validation_mae=best,
                    direct_supervised_anchors_per_epoch=math.ceil(STEPS/a.k),labeled_batches_actually_consumed=label_state["consumed"],
                    unlabeled_batches_actually_consumed=EPOCHS*STEPS)
    status["training_status"]="complete"; atomic_json(out/"metadata.json",metadata);atomic_json(out/"status.json",status)
    marker=out/"final_test_complete"; started_marker=out/"final_test_started"
    assert not marker.exists() and not started_marker.exists()
    started_tmp=out/"final_test_started.tmp";started_tmp.write_text(json.dumps({"started_unix":time.time()})+"\n");os.replace(started_tmp,started_marker)
    checkpoint=torch.load(out/"best.pt",map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"])
    mae,r2,p,y=evaluate(model,test,mean,std,device,True)
    np.savez_compressed(out/"test_predictions.npz",predictions_years=p,targets_years=y)
    atomic_json(out/"metrics.json",{"best_validation_epoch":best_epoch,"best_validation_mae":best,"test_mae":mae,"test_r2":r2})
    marker_tmp=out/"final_test_complete.tmp";marker_tmp.write_text(json.dumps({"completed_unix":time.time(),"test_evaluations":1})+"\n");os.replace(marker_tmp,marker)
    status["final_evaluation_status"]="complete"; metadata.update(final_evaluation_status="complete",final_test_mae=mae,final_test_r2=r2)
    atomic_json(out/"metadata.json",metadata);atomic_json(out/"status.json",status)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preflight",action="store_true");p.add_argument("--backbone",choices=("resnet50","dinov2"),required=True)
    p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--k",type=int,default=1);p.add_argument("--attempt",type=int,default=1);p.add_argument("--config-hash",default="preflight");p.add_argument("--dino",default="s")
    a=p.parse_args(); preflight(a.backbone,Path(a.output)) if a.preflight else train(a)
