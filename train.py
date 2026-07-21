import argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from backbone import ResNet50Regressor
from dinov2_backbone import DINOv2Regressor
from hpl import HPLTrainer, HPLConfig, UncertaintyLearner
from probe_filter import ProbeFilteredTrainer, ProbeConfig, train_probe
from stsb import make_data_stsb, TextRegressor
from hpl_data import make_data_hpl_official


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_model(args, device):
    """Create the appropriate backbone model."""
    if args.backbone == 'dinov2':
        return DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
    return ResNet50Regressor(pretrained=args.pretrained).to(device)


def age_from_name(p):
    try: return float(p.name.split('_')[0])
    except Exception: return None


class UTKFace(Dataset):
    def __init__(self, items, tfm, mean=None, std=None, unlabeled=False, strong_tfm=None):
        self.items, self.tfm, self.strong_tfm = items, tfm, strong_tfm
        self.mean, self.std, self.unlabeled = mean, std, unlabeled

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert('RGB')
        if self.unlabeled:
            if self.strong_tfm is None: return self.tfm(img)
            return self.tfm(img), self.strong_tfm(img)
        y = torch.tensor((y - self.mean) / self.std, dtype=torch.float32)
        return self.tfm(img), y


def make_data(args):
    if args.dataset == 'imdb_wiki':
        return make_data_imdb_wiki(args)
    if args.dataset == 'stsb':
        return make_data_stsb(args)
    if args.dataset == 'utkface_official':
        return make_data_hpl_official(args)
    return make_data_utkface(args)


def make_data_imdb_wiki(args):
    import json
    meta_path = Path(args.data_dir) / 'metadata.json'
    if not meta_path.exists():
        raise RuntimeError(f'{meta_path} not found. Run preprocess_imdb_wiki.py first.')
    with open(meta_path) as f:
        all_items = json.load(f)
    items = [(Path(item['path']), float(item['age'])) for item in all_items]
    items = [(p, y) for p, y in items if p.exists()]
    if not items:
        raise RuntimeError('No valid images found in metadata.json')
    random.shuffle(items)

    n = len(items); n_test = int(0.1 * n); n_val = int(0.1 * n)
    test, val, train = items[:n_test], items[n_test:n_test+n_val], items[n_test+n_val:]
    n_lab = max(1, int(args.labeled_ratio * len(train)))
    labeled, unlabeled = train[:n_lab], train[n_lab:]

    y = np.array([v for _, v in labeled], dtype=np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-6)

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    weak = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 10), transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(args.img_size), transforms.ToTensor(), norm])

    use_strong = args.method in ('hpl', 'probe')
    lab = UTKFace(labeled, weak, y_mean, y_std)
    unlab = UTKFace(unlabeled, weak, unlabeled=True, strong_tfm=strong if use_strong else None)
    val_ds = UTKFace(val, eval_tfm, y_mean, y_std)
    test_ds = UTKFace(test, eval_tfm, y_mean, y_std)

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=drop)

    print(f'IMDB-WIKI: labeled={len(labeled)}, unlabeled={len(unlabeled)}, val={len(val)}, test={len(test)}')
    print(f'label scaler: mean={y_mean:.2f}, std={y_std:.2f}')
    return loader(lab), loader(unlab), loader(val_ds, False, False), loader(test_ds, False, False), y_mean, y_std


def make_data_utkface(args):
    root = Path(args.data_dir)
    files = list(root.glob('*.jpg')) + list(root.glob('*.png')) + list(root.glob('*.jpeg'))
    items = [(p, age_from_name(p)) for p in files]
    items = [(p, y) for p, y in items if y is not None and 0 <= y <= 120]
    if not items: raise RuntimeError('No UTKFace images found.')
    random.shuffle(items)

    n = len(items); n_test = int(0.1 * n); n_val = int(0.1 * n)
    test, val, train = items[:n_test], items[n_test:n_test+n_val], items[n_test+n_val:]
    n_lab = max(1, int(args.labeled_ratio * len(train)))
    labeled, unlabeled = train[:n_lab], train[n_lab:]

    y = np.array([v for _, v in labeled], dtype=np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-6)

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    weak = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 10), transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(args.img_size), transforms.ToTensor(), norm])

    use_strong = args.method in ('hpl', 'probe')
    lab = UTKFace(labeled, weak, y_mean, y_std)
    unlab = UTKFace(unlabeled, weak, unlabeled=True, strong_tfm=strong if use_strong else None)
    val_ds = UTKFace(val, eval_tfm, y_mean, y_std)
    test_ds = UTKFace(test, eval_tfm, y_mean, y_std)

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=drop)

    print(f'UTKFace: labeled={len(labeled)}, unlabeled={len(unlabeled)}, val={len(val)}, test={len(test)}')
    print(f'label scaler: mean={y_mean:.2f}, std={y_std:.2f}')
    return loader(lab), loader(unlab), loader(val_ds, False, False), loader(test_ds, False, False), y_mean, y_std


def cycle(loader):
    while True:
        for b in loader: yield b


@torch.no_grad()
def eval_mae(model, loader, mean, std, device):
    model.eval(); ps=[]; ys=[]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        ps.append((p.cpu() * std + mean)); ys.append((y.cpu() * std + mean))
    p, y = torch.cat(ps), torch.cat(ys)
    mae = (p-y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    return mae, r2


# ── Supervised ──

def train_supervised(args, loaders, scaler, device):
    lab, _, val, test = loaders
    mean, std = scaler
    model = ResNet50Regressor(pretrained=args.pretrained).to(device)
    opt = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': args.lr},
                            {'params': model.head.parameters(), 'lr': args.fc_lr}])
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for x, y in lab:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = F.mse_loss(model(x), y); loss.backward(); opt.step()
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST supervised: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')


def train_supervised_text(args, loaders, scaler, device):
    lab, _, val, test = loaders
    mean, std = scaler
    feat_dim = next(iter(lab))[0].shape[1]
    model = TextRegressor(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for x, y in lab:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = F.mse_loss(model(x), y); loss.backward(); opt.step()
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in val:
                ps.append(model(x.to(device)).cpu() * std + mean)
                ys.append(y.cpu() * std + mean)
        p, y = torch.cat(ps), torch.cat(ys)
        mae = (p - y).abs().mean().item()
        r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in test:
            ps.append(model(x.to(device)).cpu() * std + mean)
            ys.append(y.cpu() * std + mean)
    p, y = torch.cat(ps), torch.cat(ys)
    mae = (p - y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    print(f'TEST supervised: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')


# ── HPL ──

def train_hpl(args, loaders, scaler, device):
    lab, unlab, val, test = loaders
    mean, std = scaler

    if args.dataset == 'stsb':
        feat_dim = next(iter(lab))[0].shape[1]
        model = TextRegressor(feat_dim).to(device)
    else:
        model = ResNet50Regressor(pretrained=args.pretrained).to(device)

    unc = UncertaintyLearner().to(device)
    opt_feat = torch.optim.Adam(model.backbone.parameters(), lr=args.lr)
    opt_fc = torch.optim.Adam(model.head.parameters(), lr=args.fc_lr)
    opt_unc = torch.optim.Adam(unc.parameters(), lr=args.unc_lr)
    trainer = HPLTrainer(model, unc, opt_feat, opt_fc, opt_unc,
                         HPLConfig(w_ulb=10.0, lambda2=0.1, update_unc_every=5))

    lit, uit, mit = cycle(lab), cycle(unlab), cycle(lab)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for _ in range(max(len(lab), len(unlab))):
            x_l, y_l = next(lit); x_m, y_m = next(mit); x_w, x_s = next(uit)
            trainer.step(x_l.to(device), y_l.to(device), x_w.to(device), x_s.to(device),
                        x_m.to(device), y_m.to(device))
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST hpl: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')

    if args.save:
        import os; os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'uncertainty': unc.state_dict(),
            'mean': mean, 'std': std,
        }, args.save)
        print(f'Saved to {args.save}')



# ── Probe (RAPL) ──

def train_probe_method(args, loaders, scaler, device):
    if args.dataset == 'stsb':
        return train_probe_text(args, loaders, scaler, device)

    lab, unlab, val, test = loaders
    mean, std = scaler

    # Frozen backbone for probe — decoupled from trainable backbone
    _pb = args.probe_backbone or args.backbone
    if _pb == 'dinov2':
        frozen = DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
    else:
        frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.eval()
    probe = train_probe(frozen.backbone, lab, device)
    probe = probe.to(device)

    # Trainable model
    if args.backbone == 'dinov2':
        model = DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
        opt = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.head.parameters(), 'lr': 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        model = ResNet50Regressor(pretrained=args.pretrained).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    cfg = ProbeConfig(lambda_u=args.lambda_u)
    trainer = ProbeFilteredTrainer(model, frozen.backbone, probe, opt, cfg)

    lit, uit = cycle(lab), cycle(unlab)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for _ in range(max(len(lab), len(unlab))):
            x_l, y_l = next(lit); x_u_w, x_u_s = next(uit)
            trainer.step(x_l.to(device), y_l.to(device),
                        x_u_w.to(device), x_u_s.to(device))
        scheduler.step()
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST probe: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')

    if args.save:
        import os; os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'frozen_backbone': frozen.backbone.state_dict(),
            'probe': probe.state_dict(),
            'mean': mean, 'std': std,
        }, args.save)
        print(f'Saved to {args.save}')


def train_probe_text(args, loaders, scaler, device):
    lab, unlab, val, test = loaders
    mean, std = scaler
    feat_dim = next(iter(lab))[0].shape[1]

    feats, labels = [], []
    for x, y in lab:
        feats.append(x); labels.append(y)
    feats_all = torch.cat(feats)
    labels_all = torch.cat(labels)

    Z = torch.cat([feats_all, torch.ones(len(feats_all), 1)], dim=1)
    sol = torch.linalg.lstsq(Z, labels_all.unsqueeze(1)).solution
    probe = nn.Linear(feat_dim, 1)
    probe.weight.data.copy_(sol[:-1, 0].unsqueeze(0))
    probe.bias.data.copy_(sol[-1:, 0])
    probe.requires_grad_(False)
    probe.eval().to(device)

    pred = feats_all @ sol[:-1, 0] + sol[-1, 0]
    mae = (pred - labels_all).abs().mean()
    print(f'Probe trained: MAE={mae:.3f} (normalized)')

    model = TextRegressor(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    lit, uit = cycle(lab), cycle(unlab)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for _ in range(max(len(lab), len(unlab))):
            x_l, y_l = next(lit); x_u_w, x_u_s = next(uit)
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

            loss_sup = F.mse_loss(model(x_l), y_l)
            with torch.no_grad():
                pseudo = model(x_u_w).detach()
                probe_est = probe(x_u_w).squeeze(-1)
            disagreement = (probe_est - pseudo).abs()
            r = (1.0 / (1.0 + disagreement)).detach()
            pred_s = model(x_u_s)
            loss_u = (r * (pred_s - pseudo).pow(2)).mean()
            loss = loss_sup + args.lambda_u * loss_u
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in val:
                ps.append(model(x.to(device)).cpu() * std + mean)
                ys.append(y.cpu() * std + mean)
        p, y = torch.cat(ps), torch.cat(ys)
        val_mae = (p - y).abs().mean().item()
        r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
        print(f'ep {ep+1}: val_mae={val_mae:.3f}, r2={r2:.4f}')
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in test:
            ps.append(model(x.to(device)).cpu() * std + mean)
            ys.append(y.cpu() * std + mean)
    p, y = torch.cat(ps), torch.cat(ys)
    test_mae = (p - y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    print(f'TEST probe: mae={test_mae:.3f}, r2={r2:.4f} (best val epoch)')


# ── Main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', default='utkface', choices=['utkface', 'imdb_wiki', 'stsb', 'utkface_official'])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--method', default='probe', choices=['probe', 'hpl', 'supervised'])
    p.add_argument('--labeled_ratio', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--fc_lr', type=float, default=1e-3)
    p.add_argument('--unc_lr', type=float, default=1e-4)
    p.add_argument('--lambda_u', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', type=str, default=None, help='Path to save checkpoint')
    p.add_argument('--backbone', default='resnet50', choices=['resnet50', 'dinov2'])
    p.add_argument('--probe_backbone', default=None, choices=['resnet50', 'dinov2'])
    p.add_argument('--dino', default='s', choices=['s', 'b', 'l'])
    p.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    p.set_defaults(pretrained=True)

    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = make_data(args)
    scaler = loaders[-2:]
    loaders = loaders[:4]

    if args.method == 'probe':
        train_probe_method(args, loaders, scaler, device)
    elif args.method == 'hpl':
        train_hpl(args, loaders, scaler, device)
    elif args.method == 'supervised':
        if args.dataset == 'stsb': train_supervised_text(args, loaders, scaler, device)
        else: train_supervised(args, loaders, scaler, device)


if __name__ == '__main__':
    main()