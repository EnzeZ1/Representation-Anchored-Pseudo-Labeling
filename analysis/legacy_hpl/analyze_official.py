"""
Why Probe Outperforms HPL: Comprehensive Analysis

Usage:
    python analysis/legacy_hpl/analyze_official.py <probe_ckpt> <hpl_ckpt> <data_dir> [dataset]

Examples:
    # Official splits
    python analysis/legacy_hpl/analyze_official.py checkpoints/probe_official_5.pt \
        checkpoints/hpl_official_r50_5.pt \
        Heteroscedastic-Pseudo-Labels-main/utkface/data utkface_official

    # Your splits (default)
    python analysis/legacy_hpl/analyze_official.py checkpoints/utk_probe_5_seed0_b32.pt \
        checkpoints/utk_hpl_5_seed0_officialhp.pt \
        data/utkface_all
"""

import sys, argparse, random
from pathlib import Path

# Keep repository-root imports working when invoked as
# ``python analysis/legacy_hpl/analyze_official.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def _make_model(probe_dim, device):
    """Auto-detect backbone from feature dim: 384=DINOv2-S, 768=DINOv2-B, 2048=ResNet50."""
    if probe_dim == 384:
        from dinov2_backbone import DINOv2Regressor
        print(f'  Detected DINOv2-S (dim={probe_dim})')
        return DINOv2Regressor(size='small').to(device)
    elif probe_dim == 768:
        from dinov2_backbone import DINOv2Regressor
        print(f'  Detected DINOv2-B (dim={probe_dim})')
        return DINOv2Regressor(size='base').to(device)
    else:
        from backbone import ResNet50Regressor
        print(f'  Detected ResNet50 (dim={probe_dim})')
        return ResNet50Regressor(pretrained=True).to(device)


def load_models(probe_path, hpl_path, device):
    from backbone import ResNet50Regressor
    from hpl import UncertaintyLearner

    pc = torch.load(probe_path, map_location=device, weights_only=False)
    probe_dim = pc['probe']['weight'].shape[1]

    # RAPL: auto-detect backbone from probe dim
    print('Loading RAPL model...')
    model_p = _make_model(probe_dim, device)
    model_p.load_state_dict(pc['model'])
    model_p.eval()

    frozen = _make_model(probe_dim, device)
    frozen.backbone.load_state_dict(pc['frozen_backbone'])
    frozen.eval()

    probe = nn.Linear(probe_dim, 1).to(device)
    probe.load_state_dict(pc['probe'])
    probe.eval()

    y_mean, y_std = pc['mean'], pc['std']

    # HPL: always ResNet50 (train_hpl is hardcoded R50)
    print('Loading HPL model...')
    hc = torch.load(hpl_path, map_location=device, weights_only=False)
    model_h = ResNet50Regressor(pretrained=True).to(device)
    model_h.load_state_dict(hc['model'])
    model_h.eval()

    # Fresh pretrained R50 for HPL drift comparison
    hpl_init = ResNet50Regressor(pretrained=True).to(device)
    hpl_init.eval()

    unc = UncertaintyLearner().to(device)
    unc.load_state_dict(hc['uncertainty'])
    unc.eval()

    return model_p, frozen, probe, model_h, hpl_init, unc, y_mean, y_std


@torch.no_grad()
def collect_data(model_p, frozen, probe, model_h, hpl_init, unc, loader, y_mean, y_std, device):
    results = {k: [] for k in [
        'true_age', 'pred_probe', 'pred_hpl',
        'probe_est', 'probe_r', 'probe_d',
        'hpl_w', 'hpl_unc_raw',
        'feat_frozen', 'feat_probe_bb', 'feat_hpl_bb', 'feat_hpl_init',
    ]}

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
        z_hpl_init = hpl_init.backbone(x)
        unc_in = torch.stack([pred_h - pred_h, pred_h], dim=-1)
        unc_raw = unc(unc_in).squeeze(-1)
        w = torch.exp(-unc_raw) / 2

        results['true_age'].append(true_age.cpu())
        results['pred_probe'].append((pred_p * y_std + y_mean).cpu())
        results['pred_hpl'].append((pred_h * y_std + y_mean).cpu())
        results['probe_est'].append((probe_est * y_std + y_mean).cpu())
        results['probe_r'].append(r.cpu())
        results['probe_d'].append((d * y_std).cpu())
        results['hpl_w'].append(w.cpu())
        results['hpl_unc_raw'].append(unc_raw.cpu())
        results['feat_frozen'].append(z_frozen.cpu())
        results['feat_probe_bb'].append(z_probe_bb.cpu())
        results['feat_hpl_bb'].append(z_hpl_bb.cpu())
        results['feat_hpl_init'].append(z_hpl_init.cpu())

    out = {}
    for k, v in results.items():
        t = torch.cat(v)
        out[k] = t if 'feat' in k else t.numpy()
    return out


def analysis_1_tails(data, axes):
    ax1, ax2 = axes
    true = data['true_age']
    err_p = np.abs(data['pred_probe'] - true)
    err_h = np.abs(data['pred_hpl'] - true)

    groups = [(0, 10, 'Children'), (10, 20, 'Teens'), (20, 40, 'Young Adults'),
              (40, 60, 'Middle Age'), (60, 80, 'Seniors'), (80, 120, 'Elderly')]

    labels, mae_p, mae_h = [], [], []
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

    ax2.scatter(true, err_p, alpha=0.1, s=3, label='Probe')
    ax2.scatter(true, err_h, alpha=0.1, s=3, label='HPL')
    ax2.set_xlabel('True Age'); ax2.set_ylabel('|pred - true|')
    ax2.set_title('Error vs Age'); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 30)


def analysis_2_drift(data, ax):
    feat_frozen = data['feat_frozen']       # RAPL's frozen backbone features
    feat_probe = data['feat_probe_bb']      # RAPL's trained backbone features
    feat_hpl = data['feat_hpl_bb']          # HPL's trained backbone features
    feat_hpl_init = data['feat_hpl_init']   # HPL's pretrained init features

    def cos_sim(a, b):
        a_n = F.normalize(a, dim=1)
        b_n = F.normalize(b, dim=1)
        return (a_n * b_n).sum(dim=1).numpy()

    # Each method compared against its own pretrained init
    sim_probe = cos_sim(feat_frozen, feat_probe)
    sim_hpl = cos_sim(feat_hpl_init, feat_hpl)

    ax.hist(sim_probe, bins=50, alpha=0.6, label=f'RAPL drift (mean={sim_probe.mean():.3f})')
    ax.hist(sim_hpl, bins=50, alpha=0.6, label=f'HPL drift (mean={sim_hpl.mean():.3f})')
    ax.set_xlabel('Cosine similarity to pretrained init')
    ax.set_ylabel('Count')
    ax.set_title('2. Representation Drift from Pretrained')
    ax.legend(fontsize=8)


def analysis_3_d_vs_error(data, ax):
    d = data['probe_d']
    true = data['true_age']
    err = np.abs(data['pred_probe'] - true)

    corr, pval = spearmanr(d, err)
    ax.scatter(d, err, alpha=0.1, s=3)
    ax.set_xlabel('Probe disagreement d(x) (years)')
    ax.set_ylabel('Actual error |ŷ - y| (years)')
    ax.set_title(f'3. d(x) vs Error (Spearman ρ={corr:.3f}, p={pval:.1e})')
    ax.set_xlim(0, 30); ax.set_ylim(0, 30)
    ax.plot([0, 30], [0, 30], 'r--', alpha=0.3, label='d = error')
    ax.legend(fontsize=8)


def analysis_4_confirmation_bias(data, axes):
    ax1, ax2 = axes
    true = data['true_age']

    err_p = np.abs(data['pred_probe'] - true)
    r = data['probe_r']
    corr_p, _ = spearmanr(r, err_p)
    ax1.scatter(r, err_p, alpha=0.1, s=3, c='blue')
    ax1.set_xlabel('Probe trust r(x)')
    ax1.set_ylabel('Actual error (years)')
    ax1.set_title(f'4a. Probe: Trust vs Error\n(ρ={corr_p:.3f}, independent view)')
    ax1.set_ylim(0, 30)

    err_h = np.abs(data['pred_hpl'] - true)
    w = data['hpl_w']
    if w.std() > 1e-10:
        corr_h, _ = spearmanr(w, err_h)
        ax2.scatter(w, err_h, alpha=0.1, s=3, c='orange')
        ax2.set_title(f'4b. HPL: Weight vs Error\n(ρ={corr_h:.3f}, self-referential)')
    else:
        ax2.text(0.5, 0.5, f'COLLAPSED\nmean_w={w.mean():.2e}\nraw={data["hpl_unc_raw"].mean():.1f}',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='red')
        ax2.set_title('4b. HPL Weight Collapse')
    ax2.set_xlabel('HPL weight w(x)')
    ax2.set_ylabel('Actual error (years)')
    ax2.set_ylim(0, 30)


def analysis_5_reliability_curve(data, axes):
    ax1, ax2 = axes
    true = data['true_age']

    for ax, weights, errors, name, color in [
        (ax1, data['probe_r'], np.abs(data['pred_probe'] - true), 'Probe', 'blue'),
        (ax2, data['hpl_w'], np.abs(data['pred_hpl'] - true), 'HPL', 'orange'),
    ]:
        if weights.std() < 1e-10:
            ax.text(0.5, 0.5, f'{name} weights collapsed\nNo reliability curve',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'5. {name} Reliability (Collapsed)')
            continue

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

        for bc, be, bn in zip(bin_centers, bin_errors, bin_counts):
            ax.text(bc, be + 0.2, str(bn), ha='center', fontsize=5, color='gray')


def analysis_6_tau_bound(data, ax):
    true = data['true_age']
    probe_est = data['probe_est']
    probe_err = np.abs(probe_est - true)

    taus = np.arange(1, 31)
    fractions = [(probe_err <= tau).mean() * 100 for tau in taus]

    ax.plot(taus, fractions, 'b-o', markersize=3)
    ax.set_xlabel('τ (years)')
    ax.set_ylabel('% samples with |ỹ - y| ≤ τ')
    ax.set_title('6. Probe Error Bound')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.3)

    for target in [50, 75, 90]:
        idx = np.searchsorted(fractions, target)
        if idx < len(taus):
            ax.annotate(f'{target}% at τ={taus[idx]}',
                       xy=(taus[idx], fractions[idx]), fontsize=8, color='red')

    print(f'\n=== Probe Error Bound Analysis ===')
    print(f'Probe MAE: {probe_err.mean():.1f} years')
    print(f'Probe median error: {np.median(probe_err):.1f} years')
    for tau in [3, 5, 7, 10, 15]:
        pct = (probe_err <= tau).mean() * 100
        print(f'  |ỹ - y| ≤ {tau}: {pct:.1f}%')


def main():
    probe_path = sys.argv[1]
    hpl_path = sys.argv[2]
    data_dir = sys.argv[3]
    dataset = sys.argv[4] if len(sys.argv) > 4 else 'utkface'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading models...')
    model_p, frozen, probe_linear, model_h, hpl_init, unc, y_mean, y_std = \
        load_models(probe_path, hpl_path, device)

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

    print('Running inference on validation set...')
    data = collect_data(model_p, frozen, probe_linear, model_h, hpl_init, unc,
                        val_loader, y_mean, y_std, device)

    true = data['true_age']
    err_p = np.abs(data['pred_probe'] - true)
    err_h = np.abs(data['pred_hpl'] - true)
    print(f'\n=== Summary ===')
    print(f'Probe MAE: {err_p.mean():.2f}, HPL MAE: {err_h.mean():.2f}')
    print(f'Probe mean_r: {data["probe_r"].mean():.3f}, std: {data["probe_r"].std():.3f}')
    print(f'HPL mean_w: {data["hpl_w"].mean():.4e}, std: {data["hpl_w"].std():.4e}')
    print(f'HPL unc_raw mean: {data["hpl_unc_raw"].mean():.2f}')

    r = data['probe_r']
    rho_p, p_p = spearmanr(r, err_p)
    print(f'\n=== Confirmation Bias ===')
    print(f'Probe: r-error Spearman ρ = {rho_p:.3f} (p={p_p:.1e})')
    w = data['hpl_w']
    if w.std() > 1e-10:
        rho_h, p_h = spearmanr(w, err_h)
        print(f'HPL:   w-error Spearman ρ = {rho_h:.3f} (p={p_h:.1e})')
    else:
        print(f'HPL:   COLLAPSED (mean_w={w.mean():.2e}, raw={data["hpl_unc_raw"].mean():.1f})')

    print(f'\n=== Probe Reliability ===')
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 1.0)]:
        mask = (r >= lo) & (r < hi + 0.001)
        if mask.sum() > 0:
            print(f'  r ∈ [{lo:.1f}, {hi:.1f}]: mean_err={err_p[mask].mean():.2f}yr, n={mask.sum()}')

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    analysis_1_tails(data, (fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])))
    analysis_2_drift(data, fig.add_subplot(gs[1, 0]))
    analysis_3_d_vs_error(data, fig.add_subplot(gs[1, 1]))
    analysis_4_confirmation_bias(data, (fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])))
    analysis_5_reliability_curve(data, (fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])))
    analysis_6_tau_bound(data, fig.add_subplot(gs[2, 2:4]))

    suffix = 'official' if 'official' in dataset else 'ours'
    fig.suptitle(f'Why Probe Outperforms HPL ({suffix} splits)', fontsize=16, y=1.01)
    out_name = f'probe_vs_hpl_{suffix}.png'
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f'\nSaved {out_name}')


if __name__ == '__main__':
    main()
