"""Fixed patient-level LVIDd protocol; generated manifests remain protected."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from data_processing.chexchonet import lvidd_cohort_indices

PROTOCOL_VERSION="chexchonet-lvidd-benchmark-v1"
TRANSFORM_VERSION="chexchonet-anatomy-preserving-v1"
PRIMARY_TARGET="lvidd"; TARGET_UNIT="centimeters"
RATIOS=(.05,.10,.20,1.00); SEEDS=tuple(range(6)); SPLIT_SEED=20260801
NORMALIZE=transforms.Normalize((.485,.456,.406),(.229,.224,.225))

def build_evaluation_transform():
    return transforms.Compose([transforms.ToTensor(),NORMALIZE])
def _affine(degrees,translate,scale):
    return transforms.Compose([transforms.RandomAffine(degrees=degrees,translate=(translate,translate),scale=scale,interpolation=InterpolationMode.BILINEAR,fill=0),transforms.ToTensor(),NORMALIZE])
def build_weak_transform(): return _affine(2,.01,(.99,1.01))
def build_strong_transform(): return _affine(5,.03,(.97,1.03))
def build_train_transform(): return build_weak_transform()

def _digest(payload): return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def fixed_patient_split(records):
    eligible=lvidd_cohort_indices(records); patients=sorted({records[i]["patient_id"] for i in eligible})
    shuffled=patients.copy(); random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val=int(len(shuffled)*.05); n_test=int(len(shuffled)*.05)
    return {"train":shuffled[n_val+n_test:],"validation":shuffled[:n_val],"test":shuffled[n_val:n_val+n_test]}

def generate_patient_manifest(records,*,seed,ratio,target=PRIMARY_TARGET):
    if ratio not in RATIOS or seed not in SEEDS or target!=PRIMARY_TARGET: raise ValueError("Unsupported formal protocol")
    eligible=lvidd_cohort_indices(records); splits=fixed_patient_split(records); train=list(splits["train"])
    order=train.copy(); random.Random(seed).shuffle(order)
    n=len(train) if ratio==1 else max(1,int(len(train)*ratio))
    labeled=order[:n]; unlabeled=order[n:]
    groups={"labeled":set(labeled),"unlabeled":set(unlabeled),"validation":set(splits["validation"]),"test":set(splits["test"])}
    indices={role:[i for i in eligible if records[i]["patient_id"] in members] for role,members in groups.items()}
    values=np.asarray([records[i]["targets"][target] for i in indices["labeled"]],dtype=np.float64)
    stable=[hashlib.sha256((records[i]["patient_id"]+"\0"+records[i]["image_path"]+"\0"+format(records[i]["targets"][target],".17g")).encode()).hexdigest() for i in eligible]
    manifest={"protocol_version":PROTOCOL_VERSION,"transform_version":TRANSFORM_VERSION,"target":target,"target_unit":TARGET_UNIT,"split_seed":SPLIT_SEED,"seed":seed,"labeled_ratio":ratio,"patient_splits":splits,"labeled_patients":labeled,"unlabeled_patients":unlabeled,"indices":indices,"cohort_digest":_digest(sorted(stable)),"split_digest":_digest(splits),"counts":{"eligible_images":len(eligible),"train_patients":len(train),"validation_patients":len(splits["validation"]),"test_patients":len(splits["test"]),"labeled_patients":len(labeled),"unlabeled_patients":len(unlabeled),**{f"{k}_images":len(v) for k,v in indices.items()}},"label_scaler":{"mean":float(values.mean()),"std":float(values.std()),"source":"labeled training images only","ddof":0}}
    manifest["manifest_sha256"]=_digest(manifest); validate_manifest(manifest,records); return manifest

def validate_manifest(m,records):
    if m["protocol_version"]!=PROTOCOL_VERSION or m["transform_version"]!=TRANSFORM_VERSION: raise ValueError("Protocol identity mismatch")
    roles=("labeled","unlabeled","validation","test"); sets=[set(m["indices"][r]) for r in roles]
    if any(sets[i]&sets[j] for i in range(4) for j in range(i+1,4)): raise ValueError("Image overlap")
    pats=[set(m["labeled_patients"]),set(m["unlabeled_patients"]),set(m["patient_splits"]["validation"]),set(m["patient_splits"]["test"])]
    if any(pats[i]&pats[j] for i in range(4) for j in range(i+1,4)): raise ValueError("Patient overlap")
    if sets[0]|sets[1]|sets[2]|sets[3] != set(lvidd_cohort_indices(records)): raise ValueError("Incomplete cohort coverage")
    vals=np.asarray([records[i]["targets"][PRIMARY_TARGET] for i in m["indices"]["labeled"]])
    if not np.isclose(m["label_scaler"]["mean"],vals.mean()) or not np.isclose(m["label_scaler"]["std"],vals.std()): raise ValueError("Scaler mismatch")
    p=dict(m); got=p.pop("manifest_sha256")
    if got!=_digest(p): raise ValueError("Manifest checksum mismatch")
def validate_nested(manifests):
    ordered=[manifests[r] for r in RATIOS]
    if any(x["patient_splits"]!=ordered[0]["patient_splits"] for x in ordered[1:]): raise ValueError("Split drift")
    sets=[set(x["labeled_patients"]) for x in ordered]
    if not (sets[0]<sets[1]<sets[2]<sets[3]): raise ValueError("Not nested")
def save_manifest(m,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");path.chmod(0o600)
