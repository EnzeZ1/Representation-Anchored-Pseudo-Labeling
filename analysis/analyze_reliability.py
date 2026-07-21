"""
Probe reliability curve comparison: R50/R50 vs DINOv2-S/DINOv2-S

Usage:
    python analysis/analyze_reliability.py <r50_ckpt> <dinov2_ckpt> <data_dir> [dataset]

Example:
    python analysis/analyze_reliability.py \
        checkpoints/probe_official_5.pt \
        checkpoints/probe_official_dinov2_both_5.pt \
        Heteroscedastic-Pseudo-Labels-main/utkface/data \
        utkface_official
"""

import sys, argparse, random
from pathlib import Path

# Keep repository-root imports working when invoked as
# ``python analysis/analyze_reliability.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rapl(ckpt_path, device):
    from backbone import ResNet50Regressor

    pc = torch.load(ckpt_path, map_location=device, weights_only=False)
    probe_dim = pc['probe']['weight'].shape[1]

    if probe_dim == 384:
        from dinov2_backbone import DINOv2Regressor
        model = DINOv2Regressor(size='small').to(device)
        frozen = DINOv2Regressor(size='small').to(device)
        label = 'DINOv2-S'
    elif probe_dim == 768:
        from dinov2_backbone import DINOv2Regressor
        model = DINOv2Regressor(size='base').to(device)
        frozen = DINOv2Regressor(size='base').to(device)
        label = 'DINOv2-B'
    else:
        model = ResNet50Regressor(pretrained=True).to(device)
        frozen = ResNet50Regressor(pretrained=True).to(device)
        label = 'ResNet50'

    model.load_state_dict(pc['model'])
    model.eval()
    frozen.backbone.load_state_dict(pc['frozen_backbone'])
    frozen.eval()

    probe = nn.Linear(probe_dim, 1).to(device)
    probe.load_state_dict(pc['probe'])
    probe.eval()

    return model, frozen, probe, pc['mean'], pc['std'], label


@torch.no_grad()
def get_reliability(model, frozen, probe, loader, y_mean, y_std, device):
    all_r, all_err = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        true_age = y * y_std + y_mean
        pred = model(x)
        pred_age = pred * y_std + y_mean
        z = frozen.backbone(x)
        probe_est = probe(z).squeeze(-1)
        d = (probe_est - pred).abs()
        r = 1.0 / (1.0 + d)
        err = (pred_age - true_age).abs()
        all_r.append(r.cpu().numpy())
        all_err.append(err.cpu().numpy())
    return np.concatenate(all_r), np.concatenate(all_err)


def main():
    r50_path = sys.argv[1]
    dinov2_path = sys.argv[2]
    data_dir = sys.argv[3]
    dataset = sys.argv[4] if len(sys.argv) > 4 else 'utkface'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load both models
    print('Loading R50/R50...')
    m1, f1, p1, mean1, std1, lab1 = load_rapl(r50_path, device)
    print('Loading DINOv2/DINOv2...')
    m2, f2, p2, mean2, std2, lab2 = load_rapl(dinov2_path, device)

    # Load data
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from train import make_data
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset=dataset,
        seed=0, pretrained=True, backbone='resnet50',
        probe_backbone=None, dino='s', unc_lr=1e-4,
    )
    loaders = make_data(args)
    val_loader = loaders[2]

    print('Computing reliability...')
    r1, err1 = get_reliability(m1, f1, p1, val_loader, mean1, std1, device)
    r2, err2 = get_reliability(m2, f2, p2, val_loader, mean2, std2, device)

    # Bin edges: 0.0, 0.1, 0.2, ..., 1.0
    bin_edges = np.arange(0.0, 1.05, 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, r, err, label, color in [
        (ax1, r1, err1, lab1, '#4C72B0'),
        (ax2, r2, err2, lab2, '#55A868'),
    ]:
        centers, means, counts = [], [], []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i+1]
            mask = (r >= lo) & (r < hi + (1e-6 if i == len(bin_edges)-2 else 0))
            n = mask.sum()
            if n == 0: continue
            centers.append((lo + hi) / 2)
            means.append(err[mask].mean())
            counts.append(n)

        bars = ax.bar(centers, means, width=0.08, color=color, alpha=0.8, edgecolor='white')
        for c, m, n in zip(centers, means, counts):
            ax.text(c, m + 0.15, str(n), ha='center', fontsize=8, color='gray')

        ax.set_xlabel('Probe trust r(x)', fontsize=12)
        ax.set_title(f'RAPL {label}', fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        # Spearman
        from scipy.stats import spearmanr
        rho, pval = spearmanr(r, err)
        ax.text(0.95, 0.95, f'ρ={rho:.3f}\np={pval:.1e}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_ylabel('Mean MAE (years)', fontsize=12)
    fig.suptitle('Probe Reliability: Higher Trust → Lower Error?', fontsize=14, y=1.02)
    plt.tight_layout()

    suffix = 'official' if 'official' in dataset else 'ours'
    out = f'reliability_{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')

    # Print summary
    for label, r, err in [(lab1, r1, err1), (lab2, r2, err2)]:
        rho, pval = spearmanr(r, err)
        print(f'\n{label}: ρ={rho:.3f} (p={pval:.1e})')
        for lo in np.arange(0, 1.0, 0.1):
            hi = lo + 0.1
            mask = (r >= lo) & (r < hi + 1e-6)
            if mask.sum() > 0:
                print(f'  r ∈ [{lo:.1f}, {hi:.1f}]: MAE={err[mask].mean():.2f}, n={mask.sum()}')


if __name__ == '__main__':
    main()
