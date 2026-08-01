"""Canonical KonIQ-10k protocol and quality-preserving full-image views."""

from __future__ import annotations

import hashlib, json, random
from pathlib import Path
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROTOCOL_VERSION = "koniq10k-benchmark-v1"
TRANSFORM_VERSION = "koniq10k-quality-preserving-v1"
RATIOS = (0.05, 0.10, 0.20, 1.00)
EXPECTED = {0.05: 352, 0.10: 705, 0.20: 1411, 1.00: 7058}
IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def payload_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def cohort_digest(records):
    return hashlib.sha256("".join(f"{r['image_name']}\t{r['split']}\t{float(r['mos']):.17g}\n" for r in sorted(records,key=lambda x:x['image_name'])).encode()).hexdigest()

def membership_digest(records):
    return hashlib.sha256("".join(f"{r['split']}\t{r['image_name']}\n" for r in sorted(records,key=lambda x:(x['split'],x['image_name']))).encode()).hexdigest()

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(p)

def load_cohort(path):
    x=json.loads(Path(path).read_text()); validate_cohort(x); return x

def validate_cohort(x):
    records=x["records"]; names=[r["image_name"] for r in records]
    if len(records)!=10073 or len(set(names))!=10073: raise ValueError("KonIQ cohort must have 10,073 unique images")
    counts={s:sum(r["split"]==s for r in records) for s in ("training","validation","test")}
    if counts!={"training":7058,"validation":1000,"test":2015}: raise ValueError(counts)
    if any(not np.isfinite([r["mos"],r["sd"]]).all() for r in records): raise ValueError("non-finite metadata")
    if x["cohort_sha256"]!=cohort_digest(records): raise ValueError("cohort digest mismatch")
    if x["membership_sha256"]!=membership_digest(records): raise ValueError("membership digest mismatch")

def generate_manifest(cohort,seed,ratio):
    validate_cohort(cohort);ratio=float(ratio)
    train=[i for i,r in enumerate(cohort["records"]) if r["split"]=="training"]
    val=[i for i,r in enumerate(cohort["records"]) if r["split"]=="validation"]
    test=[i for i,r in enumerate(cohort["records"]) if r["split"]=="test"]
    order=train.copy();random.Random(int(seed)).shuffle(order);n=EXPECTED[ratio]
    labeled=order[:n];unlabeled=order[n:]
    y=np.asarray([cohort["records"][i]["mos"] for i in labeled],dtype=np.float64)
    body={"protocol_version":PROTOCOL_VERSION,"transform_version":TRANSFORM_VERSION,"seed":int(seed),"labeled_ratio":ratio,"cohort_sha256":cohort["cohort_sha256"],"membership_sha256":cohort["membership_sha256"],"splits":{"train":train,"validation":val,"test":test},"labeled_indices":labeled,"unlabeled_indices":unlabeled,"ordered_identifiers":{"labeled":[cohort['records'][i]['image_name'] for i in labeled],"unlabeled":[cohort['records'][i]['image_name'] for i in unlabeled],"validation":[cohort['records'][i]['image_name'] for i in val],"test":[cohort['records'][i]['image_name'] for i in test]},"counts":{"cohort":10073,"train":7058,"labeled":n,"unlabeled":7058-n,"validation":1000,"test":2015},"label_scaler":{"mean":float(y.mean()),"std":float(y.std()+1e-6),"sample_count":n,"source":"labeled training subset only"}}
    body["manifest_sha256"]=payload_digest(body);return body

def validate_manifest(manifest,cohort):
    saved=manifest["manifest_sha256"];body=dict(manifest);body.pop("manifest_sha256")
    if saved!=payload_digest(body):raise ValueError("manifest payload digest mismatch")
    expected=generate_manifest(cohort,manifest["seed"],manifest["labeled_ratio"])
    if expected!=manifest:raise ValueError("manifest is not deterministic canonical output")

def validate_family(cohort,manifests):
    by={(m["seed"],float(m["labeled_ratio"])):m for m in manifests}
    if set(by)!={(s,r) for s in range(6) for r in RATIOS}:raise ValueError("manifest matrix incomplete")
    for m in manifests:validate_manifest(m,cohort)
    for s in range(6):
        sets=[set(by[s,r]["labeled_indices"]) for r in RATIOS]
        if not (sets[0]<sets[1]<sets[2]<sets[3]):raise ValueError(f"seed {s} subsets not nested")

def build_transform(strong=False):
    ops=[transforms.Resize((252,336),interpolation=InterpolationMode.BILINEAR,antialias=True)]
    if strong:ops.insert(0,transforms.RandomHorizontalFlip(p=1.0))
    ops.extend([transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    return transforms.Compose(ops)

def inverse_target(x,mean,std):return x*std+mean
