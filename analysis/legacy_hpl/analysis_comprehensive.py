"""
Comprehensive analysis: Why does the probe outperform HPL?

Analyses:
1. Distribution tails — error by age group (children, adults, elderly)
2. Representation drift — frozen backbone vs trained backbone alignment
3. d(x) vs |ŷ - y| — does probe disagreement predict actual error?
4. Confirmation bias — HPL's self-referential loop vs probe's independence
5. Reliability curve — weight vs actual pseudo-label quality
6. τ bound — does |ỹ - y| ≤ τ hold in practice?
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def load_models(probe_path, hpl_path, device):
    from backbone import ResNet50Regressor
    from hpl import UncertaintyLearner

    # Probe
    pc = torch.load(probe_path, map_location=device)
    model_p = ResNet50Regressor(pretrained=True).to(device)
    model_p.load_state_dict(pc['model'])
    model_p.eval()

    frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.backbone.load_state_dict(pc['frozen_backbone'])
    frozen.eval()

    probe = nn.Linear(frozen.feature_dim, 1).to(device)
    probe.load_state_dict(pc['probe'])
    probe.eval()

    y_mean, y_std = pc['mean'], pc['std']

    # HPL
    hc = torch.load(hpl_path, map_location=device)
    model_h = ResNet50Regressor(pretrained=True).to(device)
    model_h.load_state_dict(hc['model'])
    model_h.eval()

    unc = UncertaintyLearner().to(device)
    unc.load_state_dict(hc['uncertainty'])
    unc.eval()

    return model_p, frozen, probe, model_h, unc, y_mean, y_std


def collect_data(model_p, frozen, probe, model_h, unc, loader, y_mean, y_std, device):
    """Run both models on data, collect everything needed for analysis."""
    results = {k: [] for k in [
        'true_age', 'pred_probe', 'pred_hpl',
        'probe_est', 'probe_r', 'probe_d',
        'hpl_w', 'hpl_unc_raw',
        'feat_frozen', 'feat_probe_bb', 'feat_hpl_bb',
    ]}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            true_age = y * y_std + y_mean

            # Probe method
            pred_p = model_p(x)
            z_frozen = frozen.backbone(x)
            z_probe_bb = model_p.backbone(x)
            probe_est = probe(z_frozen).squeeze(-1)
            d = (probe_est - pred_p).abs()
            r = 1.0 / (1.0 + d)

            # HPL method
            pred_h = model_h(x)
            z_hpl_bb = model_h.backbone(x)
            unc_in = torch.stack([pred_h - pred_h, pred_h], dim=-1)  # eval: weak=strong
            unc_raw = unc(unc_in).squeeze(-1)
            w = torch.exp(-unc_raw) / 2

            results['true_age'].append(true_age.cpu())
            results['pred_probe'].append((pred_p * y_std + y_mean).cpu())
            results['pred_hpl'].append((pred_h * y_std + y_mean).cpu())
            results['probe_est'].append((probe_est * y_std + y_mean).cpu())
            results['probe_r'].append(r.cpu())
            results['probe_d'].append((d * y_std).cpu())  # in years
            results['hpl_w'].append(w.cpu())
            results['hpl_unc_raw'].append(unc_raw.cpu())
            results['feat_frozen'].append(z_frozen.cpu())
            results['feat_probe_bb'].append(z_probe_bb.cpu())
            results['feat_hpl_bb'].append(z_hpl_bb.cpu())

    return {k: torch.cat(v).numpy() if 'feat' not in k else torch.cat(v)
            for k, v in results.items()}


def analysis_1_tails(data, axes):
    """Error by age group — are tails (children/elderly) worse?"""
    ax1, ax2 = axes
    true = data['true_age']
    err_p = np.abs(data['pred_probe'] - true)
    err_h = np.abs(data['pred_hpl'] - true)

    groups = [(0, 10, 'Children'), (10, 20, 'Teens'), (20, 40, 'Young Adults'),
              (40, 60, 'Middle Age'), (60, 80, 'Seniors'), (80, 120, 'Elderly')]

    labels, mae_p, mae_h, counts = [], [], [], []
    for lo, hi, name in groups:
        mask = (true >= lo) & (true < hi)
        if mask.sum() == 0: continue
        labels.append(f'{name}\n({mask.sum()})')
        mae_p.append(err_p[mask].mean())
        mae_h.append(err_h[mask].mean())

    x = np.arange(len(labels))
    ax1.bar(x - 0.15, mae_p, 0.3, label='Probe', alpha=0.8)
    ax1.bar(x + 0.15, mae_h, 0.3, label='HPL', alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel('MAE (years)'); ax1.set_title('1. Error by Age Group')
    ax1.legend(fontsize=8)

    # Per-age scatter
    ax2.scatter(true, err_p, alpha=0.1, s=3, label='Probe')
    ax2.scatter(true, err_h, alpha=0.1, s=3, label='HPL')
    ax2.set_xlabel('True Age'); ax2.set_ylabel('|pred - true|')
    ax2.set_title('Error vs Age'); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 30)


def analysis_2_drift(data, ax):
    """Representation drift: cosine similarity between frozen and trained backbone features."""
    feat_frozen = data['feat_frozen']
    feat_probe = data['feat_probe_bb']
    feat_hpl = data['feat_hpl_bb']

    # Per-sample cosine similarity
    def cos_sim(a, b):
        a_n = F.normalize(a, dim=1)
        b_n = F.normalize(b, dim=1)
        return (a_n * b_n).sum(dim=1).numpy()

    sim_probe = cos_sim(feat_frozen, feat_probe)
    sim_hpl = cos_sim(feat_frozen, feat_hpl)

    ax.hist(sim_probe, bins=50, alpha=0.6, label=f'Probe bb (mean={sim_probe.mean():.3f})')
    ax.hist(sim_hpl, bins=50, alpha=0.6, label=f'HPL bb (mean={sim_hpl.mean():.3f})')
    ax.set_xlabel('Cosine similarity to frozen backbone')
    ax.set_ylabel('Count')
    ax.set_title('2. Representation Drift from Pretrained')
    ax.legend(fontsize=8)


def analysis_3_d_vs_error(data, ax):
    """Plot d(x) vs |ŷ - y| — does probe disagreement predict actual pseudo-label error?"""
    d = data['probe_d']  # in years
    true = data['true_age']
    err = np.abs(data['pred_probe'] - true)

    corr, pval = spearmanr(d, err)
    ax.scatter(d, err, alpha=0.1, s=3)
    ax.set_xlabel('Probe disagreement d(x) (years)')
    ax.set_ylabel('Actual pseudo-label error |ŷ - y| (years)')
    ax.set_title(f'3. d(x) vs Error (Spearman ρ={corr:.3f}, p={pval:.1e})')
    ax.set_xlim(0, 30); ax.set_ylim(0, 30)
    ax.plot([0, 30], [0, 30], 'r--', alpha=0.3, label='d = error')
    ax.legend(fontsize=8)


def analysis_4_confirmation_bias(data, axes):
    """Confirmation bias: does each method detect its own errors?"""
    ax1, ax2 = axes
    true = data['true_age']

    # Probe: model error vs probe trust
    err_p = np.abs(data['pred_probe'] - true)
    r = data['probe_r']
    corr_p, _ = spearmanr(r, err_p)
    ax1.scatter(r, err_p, alpha=0.1, s=3, c='blue')
    ax1.set_xlabel('Probe trust r(x)')
    ax1.set_ylabel('Actual error (years)')
    ax1.set_title(f'4a. Probe: Trust vs Error\n(ρ={corr_p:.3f}, independent view)')
    ax1.set_ylim(0, 30)

    # HPL: model error vs HPL weight
    err_h = np.abs(data['pred_hpl'] - true)
    w = data['hpl_w']
    corr_h, _ = spearmanr(w, err_h)
    ax2.scatter(w, err_h, alpha=0.1, s=3, c='orange')
    ax2.set_xlabel('HPL weight w(x)')
    ax2.set_ylabel('Actual error (years)')
    ax2.set_title(f'4b. HPL: Weight vs Error\n(ρ={corr_h:.3f}, self-referential)')
    ax2.set_ylim(0, 30)


def analysis_5_reliability_curve(data, axes):
    """Reliability curve: binned weight → mean error → does higher weight mean lower error?"""
    ax1, ax2 = axes
    true = data['true_age']

    for ax, weights, errors, name, color in [
        (ax1, data['probe_r'], np.abs(data['pred_probe'] - true), 'Probe', 'blue'),
        (ax2, data['hpl_w'], np.abs(data['pred_hpl'] - true), 'HPL', 'orange'),
    ]:
        bins = np.linspace(weights.min() - 1e-6, weights.max() + 1e-6, 21)
        bin_centers, bin_errors, bin_counts = [], [], []
        for i in range(len(bins) - 1):
            mask = (weights >= bins[i]) & (weights < bins[i+1])
            if mask.sum() < 5: continue
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_errors.append(errors[mask].mean())
            bin_counts.append(mask.sum())

        ax.bar(bin_centers, bin_errors, width=(bins[1]-bins[0])*0.8,
               color=color, alpha=0.7)
        ax.set_xlabel(f'{name} weight')
        ax.set_ylabel('Mean error (years)')
        ax.set_title(f'5. {name} Reliability Curve')

        # Add count labels
        for bc, be, bn in zip(bin_centers, bin_errors, bin_counts):
            ax.text(bc, be + 0.2, str(bn), ha='center', fontsize=5, color='gray')


def analysis_6_tau_bound(data, ax):
    """Test: does |ỹ - y| ≤ τ hold? What fraction of probe estimates are within τ of truth?"""
    true = data['true_age']
    probe_est = data['probe_est']  # probe's age estimate
    probe_err = np.abs(probe_est - true)

    taus = np.arange(1, 31)
    fractions = [(probe_err <= tau).mean() * 100 for tau in taus]

    ax.plot(taus, fractions, 'b-o', markersize=3)
    ax.set_xlabel('τ (years)')
    ax.set_ylabel('% samples with |ỹ - y| ≤ τ')
    ax.set_title('6. Probe Error Bound')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.3)

    # Annotate key thresholds
    for target in [50, 75, 90]:
        idx = np.searchsorted(fractions, target)
        if idx < len(taus):
            ax.annotate(f'{target}% at τ={taus[idx]}',
                       xy=(taus[idx], fractions[idx]),
                       fontsize=8, color='red')

    # Print statistics
    print(f'\n=== Probe Error Bound Analysis ===')
    print(f'Probe MAE: {probe_err.mean():.1f} years')
    print(f'Probe median error: {np.median(probe_err):.1f} years')
    for tau in [3, 5, 7, 10, 15]:
        pct = (probe_err <= tau).mean() * 100
        print(f'  |ỹ - y| ≤ {tau}: {pct:.1f}%')


def main():
    probe_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/utk_probe_5_seed0_b32.pt'
    hpl_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/utk_hpl_5_seed0_officialhp.pt'
    data_dir = sys.argv[3] if len(sys.argv) > 3 else 'data/utkface_all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    print('Loading models...')
    model_p, frozen, probe_linear, model_h, unc, y_mean, y_std = \
        load_models(probe_path, hpl_path, device)

    # Load data (same seed=0 split)
    import random; random.seed(0)
    np.random.seed(0); torch.manual_seed(0)

    from train import make_data_utkface
    import argparse
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset='utkface',
        seed=0, pretrained=True
    )
    loaders = make_data_utkface(args)
    val_loader = loaders[2]

    # Collect data
    print('Running inference on validation set...')
    data = collect_data(model_p, frozen, probe_linear, model_h, unc,
                        val_loader, y_mean, y_std, device)

    # Summary stats
    true = data['true_age']
    err_p = np.abs(data['pred_probe'] - true)
    err_h = np.abs(data['pred_hpl'] - true)
    print(f'\n=== Summary ===')
    print(f'Probe MAE: {err_p.mean():.2f}, HPL MAE: {err_h.mean():.2f}')
    print(f'Probe mean_r: {data["probe_r"].mean():.3f}, std: {data["probe_r"].std():.3f}')
    print(f'HPL mean_w: {data["hpl_w"].mean():.4f}, std: {data["hpl_w"].std():.4f}')
    print(f'HPL unc_raw mean: {data["hpl_unc_raw"].mean():.2f}')

    # Create figure
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    ax1a = fig.add_subplot(gs[0, 0:2])
    ax1b = fig.add_subplot(gs[0, 2:4])
    analysis_1_tails(data, (ax1a, ax1b))

    ax2 = fig.add_subplot(gs[1, 0])
    analysis_2_drift(data, ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    analysis_3_d_vs_error(data, ax3)

    ax4a = fig.add_subplot(gs[1, 2])
    ax4b = fig.add_subplot(gs[1, 3])
    analysis_4_confirmation_bias(data, (ax4a, ax4b))

    ax5a = fig.add_subplot(gs[2, 0])
    ax5b = fig.add_subplot(gs[2, 1])
    analysis_5_reliability_curve(data, (ax5a, ax5b))

    ax6 = fig.add_subplot(gs[2, 2:4])
    analysis_6_tau_bound(data, ax6)

    fig.suptitle('Why Probe Outperforms HPL: Comprehensive Analysis', fontsize=16, y=1.01)
    plt.savefig('probe_vs_hpl_comprehensive.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved probe_vs_hpl_comprehensive.png')


if __name__ == '__main__':
    main()