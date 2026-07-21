"""
Compare probe vs HPL filtering: why does probe outperform?

Loads both checkpoints, evaluates on labeled data where we know truth,
and compares how well each method identifies bad pseudo-labels.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_checkpoint(path, device):
    """Load a checkpoint dict."""
    return torch.load(path, map_location=device)


def analyze(probe_path, hpl_path, data_dir, device='cuda'):
    from backbone import ResNet50Regressor
    from hpl import UncertaintyLearner
    from train import make_data_utkface, cycle
    import argparse

    # Minimal args for data loading
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset='utkface',
        seed=0, pretrained=True
    )

    import random; random.seed(0)
    np.random.seed(0); torch.manual_seed(0)

    loaders = make_data_utkface(args)
    lab, unlab, val, test = loaders[:4]

    # === Load probe model ===
    probe_ckpt = load_checkpoint(probe_path, device)
    model_probe = ResNet50Regressor(pretrained=True).to(device)
    model_probe.load_state_dict(probe_ckpt['model'])
    model_probe.eval()

    # Frozen backbone + probe from checkpoint
    frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.backbone.load_state_dict(probe_ckpt['frozen_backbone'])
    frozen.eval()

    probe = nn.Linear(frozen.feature_dim, 1).to(device)
    probe.load_state_dict(probe_ckpt['probe'])
    probe.eval()

    y_mean = probe_ckpt['mean']
    y_std = probe_ckpt['std']

    # === Load HPL model ===
    hpl_ckpt = load_checkpoint(hpl_path, device)
    model_hpl = ResNet50Regressor(pretrained=True).to(device)
    model_hpl.load_state_dict(hpl_ckpt['model'])
    model_hpl.eval()

    unc = UncertaintyLearner().to(device)
    unc.load_state_dict(hpl_ckpt['uncertainty'])
    unc.eval()

    # === Evaluate on validation set (known labels) ===
    all_probe_r = []
    all_hpl_w = []
    all_pseudo_err_probe = []
    all_pseudo_err_hpl = []
    all_pred_probe = []
    all_pred_hpl = []
    all_true = []

    with torch.no_grad():
        for x, y in val:
            x, y = x.to(device), y.to(device)
            y_real = y * y_std + y_mean  # denormalize

            # Probe method: pseudo-label + probe filter
            pseudo_p = model_probe(x)
            z = frozen.backbone(x)
            probe_est = probe(z).squeeze(-1)
            d = (probe_est - pseudo_p).abs()
            r = 1.0 / (1.0 + d)

            # HPL: pseudo-label + uncertainty filter
            pseudo_h = model_hpl(x)
            unc_in = torch.stack([pseudo_h - pseudo_h, pseudo_h], dim=-1)  # weak=strong on eval
            w_raw = unc(unc_in)
            w = (torch.exp(-w_raw) / 2).squeeze(-1)

            # Actual errors
            err_p = ((pseudo_p * y_std + y_mean) - y_real).abs()
            err_h = ((pseudo_h * y_std + y_mean) - y_real).abs()

            all_probe_r.append(r.cpu())
            all_hpl_w.append(w.cpu())
            all_pseudo_err_probe.append(err_p.cpu())
            all_pseudo_err_hpl.append(err_h.cpu())
            all_pred_probe.append((pseudo_p * y_std + y_mean).cpu())
            all_pred_hpl.append((pseudo_h * y_std + y_mean).cpu())
            all_true.append(y_real.cpu())

    probe_r = torch.cat(all_probe_r).numpy()
    hpl_w = torch.cat(all_hpl_w).numpy()
    err_p = torch.cat(all_pseudo_err_probe).numpy()
    err_h = torch.cat(all_pseudo_err_hpl).numpy()
    pred_p = torch.cat(all_pred_probe).numpy()
    pred_h = torch.cat(all_pred_hpl).numpy()
    true = torch.cat(all_true).numpy()

    # === Analysis ===
    print(f'=== Filter Calibration Analysis ===')
    print(f'Probe: mean_r={probe_r.mean():.3f}, std_r={probe_r.std():.3f}')
    print(f'HPL:   mean_w={hpl_w.mean():.3f}, std_w={hpl_w.std():.3f}')
    print()

    # Correlation between weight and actual error (should be negative)
    corr_probe = np.corrcoef(probe_r, err_p)[0, 1]
    corr_hpl = np.corrcoef(hpl_w, err_h)[0, 1]
    print(f'Correlation(probe_r, error): {corr_probe:.3f}  (more negative = better filter)')
    print(f'Correlation(hpl_w, error):   {corr_hpl:.3f}')
    print()

    # Binned analysis: average error in each weight bin
    print(f'{"Weight bin":<15} {"Probe avg err":<15} {"HPL avg err":<15} {"Probe n":<10} {"HPL n":<10}')
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask_p = (probe_r >= lo) & (probe_r < hi)
        mask_h = (hpl_w >= lo) & (hpl_w < hi)
        avg_p = err_p[mask_p].mean() if mask_p.sum() > 0 else float('nan')
        avg_h = err_h[mask_h].mean() if mask_h.sum() > 0 else float('nan')
        print(f'[{lo:.1f}, {hi:.1f})    {avg_p:<15.2f} {avg_h:<15.2f} {mask_p.sum():<10} {mask_h.sum():<10}')

    # === Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Probe vs HPL: Filter Calibration Analysis', fontsize=14)

    # 1. Weight vs error scatter
    ax = axes[0, 0]
    ax.scatter(probe_r, err_p, alpha=0.1, s=5, label='Probe')
    ax.set_xlabel('Trust weight r')
    ax.set_ylabel('Actual pseudo-label error (years)')
    ax.set_title(f'Probe: r vs error (corr={corr_probe:.3f})')
    ax.set_ylim(0, 40)

    ax = axes[0, 1]
    ax.scatter(hpl_w, err_h, alpha=0.1, s=5, color='orange', label='HPL')
    ax.set_xlabel('Uncertainty weight w')
    ax.set_ylabel('Actual pseudo-label error (years)')
    ax.set_title(f'HPL: w vs error (corr={corr_hpl:.3f})')
    ax.set_ylim(0, 40)

    # 2. Weight distributions
    ax = axes[0, 2]
    ax.hist(probe_r, bins=50, alpha=0.6, label='Probe r', density=True)
    ax.hist(hpl_w, bins=50, alpha=0.6, label='HPL w', density=True)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Density')
    ax.set_title('Weight distributions')
    ax.legend()

    # 3. Prediction error by age
    ax = axes[1, 0]
    ax.scatter(true, err_p, alpha=0.1, s=5, label='Probe')
    ax.scatter(true, err_h, alpha=0.1, s=5, label='HPL')
    ax.set_xlabel('True age')
    ax.set_ylabel('Pseudo-label error (years)')
    ax.set_title('Error by age')
    ax.legend()

    # 4. Weight by age
    ax = axes[1, 1]
    ax.scatter(true, probe_r, alpha=0.1, s=5, label='Probe r')
    ax.scatter(true, hpl_w, alpha=0.1, s=5, label='HPL w')
    ax.set_xlabel('True age')
    ax.set_ylabel('Weight')
    ax.set_title('Weight by age')
    ax.legend()

    # 5. Where they disagree most
    ax = axes[1, 2]
    diff = probe_r - hpl_w
    ax.scatter(true, diff, alpha=0.1, s=5, c=err_p, cmap='RdYlGn_r')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('True age')
    ax.set_ylabel('probe_r - hpl_w')
    ax.set_title('Filter disagreement (color=error)')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Error (years)')

    plt.tight_layout()
    plt.savefig('probe_vs_hpl_analysis.png', dpi=150)
    print(f'\nSaved probe_vs_hpl_analysis.png')


if __name__ == '__main__':
    probe_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/utk_hpl/probe_5_seed0.pt'
    hpl_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/utk_hpl/hpl_5_seed0.pt'
    data_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/utkface_all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze(probe_path, hpl_path, data_dir, device)