"""
Label-space drift analysis.

For each sample, compare:
- ỹ: what the frozen probe says (anchored to labeled data)  
- ŷ_probe: what the probe-trained model predicts
- ŷ_hpl: what the HPL-trained model predicts
- y: true label

Questions:
- Does the model drift away from the probe's estimate?
- When it drifts, does it get closer to truth (good) or further (bad)?
- Does HPL drift more than Probe?
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    probe_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/utk_probe_5_seed0_b32.pt'
    hpl_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/utk_hpl_5_seed0_officialhp.pt'
    data_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/utkface_all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from models.backbone import ResNet50Regressor
    # Load probe checkpoint
    pc = torch.load(probe_path, map_location=device)
    model_p = ResNet50Regressor(pretrained=True).to(device)
    model_p.load_state_dict(pc['model']); model_p.eval()

    frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.backbone.load_state_dict(pc['frozen_backbone']); frozen.eval()

    probe = nn.Linear(frozen.feature_dim, 1).to(device)
    probe.load_state_dict(pc['probe']); probe.eval()

    y_mean, y_std = pc['mean'], pc['std']

    # Load HPL checkpoint
    hc = torch.load(hpl_path, map_location=device)
    model_h = ResNet50Regressor(pretrained=True).to(device)
    model_h.load_state_dict(hc['model']); model_h.eval()

    # Load val data
    import random; random.seed(0)
    np.random.seed(0); torch.manual_seed(0)
    from training.train import make_data_utkface
    import argparse
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset='utkface',
        seed=0, pretrained=True
    )
    val_loader = make_data_utkface(args)[2]

    # Collect predictions
    true_ages, probe_ests, pred_probes, pred_hpls = [], [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            true_age = y * y_std + y_mean

            # Probe estimate (frozen, anchored to labeled data)
            z = frozen.backbone(x)
            est = (probe(z).squeeze(-1) * y_std + y_mean)

            # Model predictions
            pp = model_p(x) * y_std + y_mean
            ph = model_h(x) * y_std + y_mean

            true_ages.append(true_age)
            probe_ests.append(est.cpu())
            pred_probes.append(pp.cpu())
            pred_hpls.append(ph.cpu())

    true = torch.cat(true_ages).numpy()
    est = torch.cat(probe_ests).numpy()
    pp = torch.cat(pred_probes).numpy()
    ph = torch.cat(pred_hpls).numpy()

    # Compute drifts
    drift_probe = pp - est  # how far probe model moved from probe estimate
    drift_hpl = ph - est    # how far HPL model moved from probe estimate

    err_est = np.abs(est - true)     # probe estimate error
    err_pp = np.abs(pp - true)       # probe model error
    err_ph = np.abs(ph - true)       # HPL model error

    # Did drift help or hurt?
    helped_probe = err_pp < err_est  # model got closer to truth than probe
    helped_hpl = err_ph < err_est

    print('=== Label-Space Drift Analysis ===')
    print(f'\nProbe estimate MAE: {err_est.mean():.2f} years')
    print(f'Probe model MAE:    {err_pp.mean():.2f} years')
    print(f'HPL model MAE:      {err_ph.mean():.2f} years')
    print(f'\nMean drift from probe estimate:')
    print(f'  Probe model: {np.abs(drift_probe).mean():.2f} years (|ŷ - ỹ|)')
    print(f'  HPL model:   {np.abs(drift_hpl).mean():.2f} years (|ŷ - ỹ|)')
    print(f'\nDrift helped (model closer to truth than probe):')
    print(f'  Probe: {helped_probe.mean()*100:.1f}%')
    print(f'  HPL:   {helped_hpl.mean()*100:.1f}%')
    print(f'\nWhen drift helped, avg improvement:')
    imp_p = (err_est[helped_probe] - err_pp[helped_probe]).mean()
    imp_h = (err_est[helped_hpl] - err_ph[helped_hpl]).mean()
    print(f'  Probe: {imp_p:.2f} years closer to truth')
    print(f'  HPL:   {imp_h:.2f} years closer to truth')
    print(f'\nWhen drift hurt, avg damage:')
    hurt_p = ~helped_probe
    hurt_h = ~helped_hpl
    dmg_p = (err_pp[hurt_p] - err_est[hurt_p]).mean()
    dmg_h = (err_ph[hurt_h] - err_est[hurt_h]).mean()
    print(f'  Probe: {dmg_p:.2f} years further from truth')
    print(f'  HPL:   {dmg_h:.2f} years further from truth')

    # By age group
    print(f'\n=== Drift by Age Group ===')
    print(f'{"Group":<15} {"Probe drift":<15} {"HPL drift":<15} {"Probe helped%":<15} {"HPL helped%":<15}')
    for lo, hi, name in [(0,10,'Children'),(10,20,'Teens'),(20,40,'Young Adult'),
                          (40,60,'Middle Age'),(60,80,'Seniors'),(80,120,'Elderly')]:
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0: continue
        print(f'{name:<15} {np.abs(drift_probe[mask]).mean():<15.2f} '
              f'{np.abs(drift_hpl[mask]).mean():<15.2f} '
              f'{helped_probe[mask].mean()*100:<15.1f} '
              f'{helped_hpl[mask].mean()*100:<15.1f}')

    # === Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Label-Space Drift Analysis', fontsize=14)

    # 1. Probe estimate vs true (baseline)
    ax = axes[0, 0]
    ax.scatter(true, est, alpha=0.15, s=5, c='gray')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Probe Estimate ỹ')
    ax.set_title(f'Probe Estimate vs Truth\n(MAE={err_est.mean():.1f})')

    # 2. Probe model vs true
    ax = axes[0, 1]
    ax.scatter(true, pp, alpha=0.15, s=5, c='blue')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Probe Model ŷ')
    ax.set_title(f'Probe Model vs Truth\n(MAE={err_pp.mean():.1f})')

    # 3. HPL model vs true
    ax = axes[0, 2]
    ax.scatter(true, ph, alpha=0.15, s=5, c='orange')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('HPL Model ŷ')
    ax.set_title(f'HPL Model vs Truth\n(MAE={err_ph.mean():.1f})')

    # 4. Drift magnitude vs true age
    ax = axes[1, 0]
    ax.scatter(true, np.abs(drift_probe), alpha=0.15, s=5, label='Probe drift', c='blue')
    ax.scatter(true, np.abs(drift_hpl), alpha=0.15, s=5, label='HPL drift', c='orange')
    ax.set_xlabel('True Age'); ax.set_ylabel('|ŷ - ỹ| (drift from probe)')
    ax.set_title('Drift Magnitude by Age')
    ax.legend(fontsize=8)

    # 5. Helpful vs harmful drift
    ax = axes[1, 1]
    improvement_p = err_est - err_pp  # positive = model improved over probe
    improvement_h = err_est - err_ph
    ax.scatter(true, improvement_p, alpha=0.15, s=5, label='Probe', c='blue')
    ax.scatter(true, improvement_h, alpha=0.15, s=5, label='HPL', c='orange')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Improvement over probe (years)\n(positive = model better)')
    ax.set_title('Helpful vs Harmful Drift')
    ax.legend(fontsize=8)

    # 6. Scatter: drift magnitude vs whether it helped
    ax = axes[1, 2]
    bins = np.linspace(0, 20, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    helped_rate_p, helped_rate_h = [], []
    for i in range(len(bins)-1):
        mask_p = (np.abs(drift_probe) >= bins[i]) & (np.abs(drift_probe) < bins[i+1])
        mask_h = (np.abs(drift_hpl) >= bins[i]) & (np.abs(drift_hpl) < bins[i+1])
        helped_rate_p.append(helped_probe[mask_p].mean() if mask_p.sum() > 5 else np.nan)
        helped_rate_h.append(helped_hpl[mask_h].mean() if mask_h.sum() > 5 else np.nan)
    ax.plot(centers, np.array(helped_rate_p)*100, 'b-o', markersize=4, label='Probe')
    ax.plot(centers, np.array(helped_rate_h)*100, 'o-', color='orange', markersize=4, label='HPL')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Drift magnitude |ŷ - ỹ| (years)')
    ax.set_ylabel('% of drifts that helped')
    ax.set_title('Drift Magnitude → Helpful?')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('label_drift_analysis.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved label_drift_analysis.png')


if __name__ == '__main__':
    main()
