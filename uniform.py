"""
Uniform baseline: consistency training with r=1 (no filtering).
Isolates backbone contribution from probe filtering contribution.

Usage same as train.py but no probe is trained or used:
    python uniform.py -dataset utkface_official \
        --data_dir Heteroscedastic-Pseudo-Labels-main/utkface/data \
        --labeled_ratio 0.05 --epochs 30 --batch_size 16 \
        --backbone dinov2 --dino s \
        --save checkpoints/uniform_official_dinov2s_5.pt
"""

import argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone import ResNet50Regressor
from dinov2_backbone import DINOv2Regressor
from train import make_data, cycle, eval_mae, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', default='utkface', choices=['utkface', 'imdb_wiki', 'stsb', 'utkface_official'])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--labeled_ratio', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--fc_lr', type=float, default=1e-3)
    p.add_argument('--lambda_u', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', type=str, default=None)
    p.add_argument('--backbone', default='resnet50', choices=['resnet50', 'dinov2'])
    p.add_argument('--dino', default='s', choices=['s', 'b', 'l'])
    p.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    p.set_defaults(pretrained=True)
    # dummy args needed by make_data
    p.add_argument('--method', default='probe')
    p.add_argument('--unc_lr', type=float, default=1e-4)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = make_data(args)
    mean, std = loaders[-2], loaders[-1]
    lab, unlab, val, test = loaders[:4]

    # Model
    if args.backbone == 'dinov2':
        model = DINOv2Regressor(size={'s': 'small', 'b': 'base', 'l': 'large'}[args.dino]).to(device)
        opt = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.head.parameters(), 'lr': 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        model = ResNet50Regressor(pretrained=args.pretrained).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    lit, uit = cycle(lab), cycle(unlab)
    best_val, best_state = float('inf'), None

    for ep in range(args.epochs):
        model.train()
        for _ in range(max(len(lab), len(unlab))):
            x_l, y_l = next(lit)
            x_u_w, x_u_s = next(uit)
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

            loss_sup = F.mse_loss(model(x_l), y_l)

            with torch.no_grad():
                pseudo = model(x_u_w).detach()
            pred_s = model(x_u_s)
            loss_u = (pred_s - pseudo).pow(2).mean()  # r=1, no filtering

            loss = loss_sup + args.lambda_u * loss_u
            opt.zero_grad(); loss.backward(); opt.step()

        scheduler.step()
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST uniform: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')

    if args.save:
        import os; os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({'model': model.state_dict(), 'mean': mean, 'std': std}, args.save)
        print(f'Saved to {args.save}')


if __name__ == '__main__':
    main()