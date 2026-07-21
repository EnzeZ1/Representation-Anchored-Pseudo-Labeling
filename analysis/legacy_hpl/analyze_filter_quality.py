"""
Filter Quality Analysis.

The probe helps by choosing WHICH unlabeled samples the model learns from.
This script evaluates: does the filter let good pseudo-labels through
and block bad ones?

On validation data (where we know truth), for each sample:
- Compute the pseudo-label the model would generate
- Check how wrong it is (|pseudo - truth|)
- Check what weight the filter assigns
- A good filter: high weight on low-error samples, low weight on high-error samples
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    probe_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/utk_probe_5_seed0_b32.pt'
    hpl_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/utk_hpl_5_seed0_officialhp.pt'
    data_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/utkface_all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from backbone import ResNet50Regressor
    from hpl import UncertaintyLearner

    # Load models
    pc = torch.load(probe_path, map_location=device)
    model_p = ResNet50Regressor(pretrained=True).to(device)
    model_p.load_state_dict(pc['model']); model_p.eval()

    frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.backbone.load_state_dict(pc['frozen_backbone']); frozen.eval()

    probe = nn.Linear(frozen.feature_dim, 1).to(device)
    probe.load_state_dict(pc['probe']); probe.eval()

    y_mean, y_std = pc['mean'], pc['std']

    hc = torch.load(hpl_path, map_location=device)
    model_h = ResNet50Regressor(pretrained=True).to(device)
    model_h.load_state_dict(hc['model']); model_h.eval()

    unc = UncertaintyLearner().to(device)
    unc.load_state_dict(hc['uncertainty']); unc.eval()

    # Load data
    import random; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from train import make_data_utkface
    import argparse
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset='utkface',
        seed=0, pretrained=True
    )
    val_loader = make_data_utkface(args)[2]

    # Collect data
    all_true, all_pseudo_p, all_pseudo_h = [], [], []
    all_probe_r, all_hpl_w = [], []
    all_probe_est = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            true_age = y * y_std + y_mean

            # Probe method
            pseudo_p = model_p(x)  # normalized
            z = frozen.backbone(x)
            probe_est = probe(z).squeeze(-1)  # normalized
            d = (probe_est - pseudo_p).abs()
            r = 1.0 / (1.0 + d)

            # HPL method
            pseudo_h = model_h(x)
            unc_in = torch.stack([pseudo_h - pseudo_h, pseudo_h], dim=-1)
            w = (torch.exp(-unc(unc_in).squeeze(-1)) / 2)

            # Denormalize
            all_true.append(true_age)
            all_pseudo_p.append((pseudo_p * y_std + y_mean).cpu())
            all_pseudo_h.append((pseudo_h * y_std + y_mean).cpu())
            all_probe_est.append((probe_est * y_std + y_mean).cpu())
            all_probe_r.append(r.cpu())
            all_hpl_w.append(w.cpu())

    true = torch.cat(all_true).numpy()
    pseudo_p = torch.cat(all_pseudo_p).numpy()
    pseudo_h = torch.cat(all_pseudo_h).numpy()
    probe_est = torch.cat(all_probe_est).numpy()
    probe_r = torch.cat(all_probe_r).numpy()
    hpl_w = torch.cat(all_hpl_w).numpy()

    # Pseudo-label errors
    err_p = np.abs(pseudo_p - true)
    err_h = np.abs(pseudo_h - true)

    # === Print Analysis ===
    print('=== Filter Quality Analysis ===')
    print(f'\nPseudo-label quality:')
    print(f'  Probe model pseudo-label MAE: {err_p.mean():.2f} years')
    print(f'  HPL model pseudo-label MAE:   {err_h.mean():.2f} years')

    print(f'\nFilter statistics:')
    print(f'  Probe: mean_r={probe_r.mean():.3f}, min={probe_r.min():.3f}, max={probe_r.max():.3f}')
    print(f'  HPL:   mean_w={hpl_w.mean():.6f}, min={hpl_w.min():.6f}, max={hpl_w.max():.6f}')

    # Key metric: correlation between filter weight and pseudo-label quality
    corr_p, pval_p = spearmanr(probe_r, err_p)
    corr_h, pval_h = spearmanr(hpl_w, err_h)
    print(f'\nFilter calibration (weight vs pseudo-label error):')
    print(f'  Probe: Spearman rho={corr_p:.4f} (p={pval_p:.2e})')
    print(f'  HPL:   Spearman rho={corr_h:.4f} (p={pval_h:.2e})')
    print(f'  (Negative = good: low weight on high-error samples)')

    # What the filter actually does: split into kept vs suppressed
    print(f'\n--- What the probe filter does ---')
    for threshold_name, threshold in [('Low trust (r<0.5)', 0.5),
                                       ('Medium (0.5≤r<0.8)', 0.8),
                                       ('High trust (r≥0.8)', 1.01)]:
        if threshold == 0.5:
            mask = probe_r < 0.5
        elif threshold == 0.8:
            mask = (probe_r >= 0.5) & (probe_r < 0.8)
        else:
            mask = probe_r >= 0.8
        if mask.sum() == 0: continue
        print(f'  {threshold_name}: {mask.sum()} samples ({mask.mean()*100:.1f}%)')
        print(f'    Avg pseudo-label error: {err_p[mask].mean():.2f} years')
        print(f'    Avg weight applied: {probe_r[mask].mean():.3f}')

    # Weighted vs unweighted pseudo-label quality
    weighted_err_p = (probe_r * err_p).sum() / probe_r.sum()
    weighted_err_h = (hpl_w * err_h).sum() / max(hpl_w.sum(), 1e-10)
    print(f'\nEffective pseudo-label error (weight-averaged):')
    print(f'  Probe: {weighted_err_p:.2f} years (vs unweighted {err_p.mean():.2f})')
    print(f'  HPL:   {weighted_err_h:.2f} years (vs unweighted {err_h.mean():.2f})')
    print(f'  Probe reduces effective error by {err_p.mean() - weighted_err_p:.2f} years')

    # === Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Filter Quality: Which pseudo-labels does each method trust?', fontsize=14)

    # 1. Pseudo-label error vs true age
    ax = axes[0, 0]
    ax.scatter(true, err_p, alpha=0.15, s=5, c='blue', label=f'Probe (MAE={err_p.mean():.1f})')
    ax.scatter(true, err_h, alpha=0.15, s=5, c='orange', label=f'HPL (MAE={err_h.mean():.1f})')
    ax.set_xlabel('True Age'); ax.set_ylabel('|pseudo - true| (years)')
    ax.set_title('Pseudo-Label Error by Age'); ax.legend(fontsize=8)
    ax.set_ylim(0, 25)

    # 2. Probe weight vs pseudo-label error (THE key plot)
    ax = axes[0, 1]
    sc = ax.scatter(err_p, probe_r, alpha=0.15, s=5, c=true, cmap='viridis')
    ax.set_xlabel('Pseudo-label error (years)'); ax.set_ylabel('Probe trust r(x)')
    ax.set_title(f'Probe: Does it catch bad pseudo-labels?\n(ρ={corr_p:.3f})')
    plt.colorbar(sc, ax=ax, label='True age')

    # 3. HPL weight vs pseudo-label error
    ax = axes[0, 2]
    sc = ax.scatter(err_h, hpl_w, alpha=0.15, s=5, c=true, cmap='viridis')
    ax.set_xlabel('Pseudo-label error (years)'); ax.set_ylabel('HPL weight w(x)')
    ax.set_title(f'HPL: Does it catch bad pseudo-labels?\n(ρ={corr_h:.3f})')
    plt.colorbar(sc, ax=ax, label='True age')

    # 4. Reliability curve: binned error by weight level
    ax = axes[1, 0]
    bins = np.linspace(probe_r.min(), probe_r.max(), 11)
    centers, avg_errs, counts = [], [], []
    for i in range(len(bins)-1):
        mask = (probe_r >= bins[i]) & (probe_r < bins[i+1])
        if mask.sum() < 5: continue
        centers.append((bins[i]+bins[i+1])/2)
        avg_errs.append(err_p[mask].mean())
        counts.append(mask.sum())
    ax.bar(centers, avg_errs, width=(bins[1]-bins[0])*0.8, color='steelblue', alpha=0.8)
    for c, e, n in zip(centers, avg_errs, counts):
        ax.text(c, e+0.1, str(n), ha='center', fontsize=7, color='gray')
    ax.set_xlabel('Probe weight r(x)'); ax.set_ylabel('Mean pseudo-label error (years)')
    ax.set_title('Probe Reliability: weight → error')

    # 5. The tutor analogy: which samples does each method learn from?
    ax = axes[1, 1]
    effective_lr_p = probe_r / probe_r.max()
    effective_lr_h = hpl_w / max(hpl_w.max(), 1e-10)

    sorted_idx_p = np.argsort(err_p)
    cumulative_weight_p = np.cumsum(probe_r[sorted_idx_p]) / probe_r.sum()

    ax.plot(err_p[sorted_idx_p], cumulative_weight_p * 100, 'b-', linewidth=2, label='Probe')
    ax.axvline(5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Pseudo-label error threshold (years)')
    ax.set_ylabel('Cumulative weight (%)')
    ax.set_title('Where does training effort go?')
    ax.legend(fontsize=8)

    # 6. Probe estimate vs pseudo-label vs truth (3-way comparison)
    ax = axes[1, 2]
    sample_idx = np.random.choice(len(true), min(200, len(true)), replace=False)
    sample_idx = sample_idx[np.argsort(true[sample_idx])]

    ax.scatter(range(len(sample_idx)), true[sample_idx], s=8, c='green',
              label='Truth', zorder=3)
    ax.scatter(range(len(sample_idx)), pseudo_p[sample_idx], s=8, c='blue',
              alpha=0.5, label='Model pseudo-label')
    ax.scatter(range(len(sample_idx)), probe_est[sample_idx], s=8, c='red',
              alpha=0.5, label='Probe estimate')
    ax.set_xlabel('Samples (sorted by age)'); ax.set_ylabel('Age')
    ax.set_title('Three-way: truth vs model vs probe')
    ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig('filter_quality_analysis.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved filter_quality_analysis.png')


if __name__ == '__main__':
    main()