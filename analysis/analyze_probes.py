"""
Compare linear vs kernel probe: how features relate to ages.

Linear: ŷ = w · z + b  → w tells you which features predict age
Kernel: ŷ = Σ αᵢ K(z, zᵢ) → αᵢ tells you which training samples matter
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/utkface_all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from models.backbone import ResNet50Regressor
    import random; random.seed(0)
    np.random.seed(0); torch.manual_seed(0)

    from training.train import make_data_utkface
    import argparse
    args = argparse.Namespace(
        data_dir=data_dir, labeled_ratio=0.05, batch_size=32,
        img_size=224, workers=4, method='probe', dataset='utkface',
        seed=0, pretrained=True
    )
    loaders = make_data_utkface(args)
    lab, _, val, test = loaders[:4]
    y_mean, y_std = loaders[4], loaders[5]

    # Extract features
    backbone = ResNet50Regressor(pretrained=True).to(device)
    backbone.eval()

    def extract(loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                f = backbone.backbone(x.to(device))
                feats.append(f.cpu()); labels.append(y)
        return torch.cat(feats).numpy(), torch.cat(labels).numpy()

    print('Extracting features...')
    X_lab, y_lab = extract(lab)
    X_val, y_val = extract(val)

    # Denormalize labels
    y_lab_age = y_lab * y_std + y_mean
    y_val_age = y_val * y_std + y_mean

    print(f'Labeled: {X_lab.shape[0]}, Val: {X_val.shape[0]}, Features: {X_lab.shape[1]}')

    # === Train both probes ===
    print('\nTraining linear probe...')
    linear = Ridge(alpha=1.0)
    linear.fit(X_lab, y_lab)
    pred_linear_val = linear.predict(X_val) * y_std + y_mean
    pred_linear_lab = linear.predict(X_lab) * y_std + y_mean
    mae_linear = np.abs(pred_linear_val - y_val_age).mean()
    print(f'  Linear val MAE: {mae_linear:.2f} years')

    print('Training kernel probe...')
    kernel = KernelRidge(alpha=1.0, kernel='rbf', gamma=1.0/X_lab.shape[1])
    kernel.fit(X_lab, y_lab)
    pred_kernel_val = kernel.predict(X_val) * y_std + y_mean
    pred_kernel_lab = kernel.predict(X_lab) * y_std + y_mean
    mae_kernel = np.abs(pred_kernel_val - y_val_age).mean()
    print(f'  Kernel val MAE: {mae_kernel:.2f} years')

    # === Analysis ===
    w = linear.coef_  # [2048] — the age direction in feature space

    print(f'\n=== Linear probe weight analysis ===')
    print(f'  w norm: {np.linalg.norm(w):.4f}')
    print(f'  w sparsity (|w|<0.001): {(np.abs(w) < 0.001).mean()*100:.1f}%')
    top_k = 20
    top_dims = np.argsort(np.abs(w))[::-1][:top_k]
    print(f'  Top {top_k} feature dims: {top_dims}')
    print(f'  Top {top_k} weights: {np.round(w[top_dims], 4)}')

    # How much variance does the top-k explain?
    for k in [10, 50, 100, 500]:
        topk = np.argsort(np.abs(w))[::-1][:k]
        w_sparse = np.zeros_like(w)
        w_sparse[topk] = w[topk]
        pred_sparse = X_val @ w_sparse + linear.intercept_
        pred_sparse_age = pred_sparse * y_std + y_mean
        mae_sparse = np.abs(pred_sparse_age - y_val_age).mean()
        print(f'  Top-{k} features only: MAE={mae_sparse:.2f} years '
              f'({np.abs(w[topk]).sum()/np.abs(w).sum()*100:.1f}% of total |w|)')

    # === Kernel dual coefficients ===
    alpha = kernel.dual_coef_.flatten()  # [n_labeled]
    print(f'\n=== Kernel probe analysis ===')
    print(f'  α range: [{alpha.min():.4f}, {alpha.max():.4f}]')
    print(f'  α norm: {np.linalg.norm(alpha):.4f}')
    top_support = np.argsort(np.abs(alpha))[::-1][:10]
    print(f'  Top 10 support samples:')
    for idx in top_support:
        print(f'    sample {idx}: age={y_lab_age[idx]:.0f}, α={alpha[idx]:.4f}')

    # === Where does kernel help vs linear? ===
    improvement = np.abs(pred_linear_val - y_val_age) - np.abs(pred_kernel_val - y_val_age)
    print(f'\n=== Where kernel beats linear ===')
    print(f'  Kernel better: {(improvement > 0).mean()*100:.1f}% of samples')
    print(f'  Mean improvement: {improvement.mean():.2f} years')

    for lo, hi, name in [(0,10,'Children'),(10,20,'Teens'),(20,40,'Young Adult'),
                          (40,60,'Middle Age'),(60,80,'Seniors'),(80,120,'Elderly')]:
        mask = (y_val_age >= lo) & (y_val_age < hi)
        if mask.sum() == 0: continue
        imp = improvement[mask].mean()
        ml = np.abs(pred_linear_val[mask] - y_val_age[mask]).mean()
        mk = np.abs(pred_kernel_val[mask] - y_val_age[mask]).mean()
        print(f'  {name}: linear MAE={ml:.1f}, kernel MAE={mk:.1f}, '
              f'improvement={imp:+.1f} years')

    # === Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Linear vs Kernel Probe: How Features Relate to Age', fontsize=14)

    # 1. Linear prediction vs truth
    ax = axes[0, 0]
    ax.scatter(y_val_age, pred_linear_val, alpha=0.15, s=5, c='blue')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Linear Probe ŷ')
    ax.set_title(f'Linear Probe (MAE={mae_linear:.1f})')
    ax.set_xlim(0, 100); ax.set_ylim(-20, 120)

    # 2. Kernel prediction vs truth
    ax = axes[0, 1]
    ax.scatter(y_val_age, pred_kernel_val, alpha=0.15, s=5, c='red')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Kernel Probe ŷ')
    ax.set_title(f'Kernel Probe (MAE={mae_kernel:.1f})')
    ax.set_xlim(0, 100); ax.set_ylim(-20, 120)

    # 3. Residual comparison
    ax = axes[0, 2]
    res_linear = pred_linear_val - y_val_age
    res_kernel = pred_kernel_val - y_val_age
    ax.scatter(y_val_age, res_linear, alpha=0.1, s=5, label='Linear', c='blue')
    ax.scatter(y_val_age, res_kernel, alpha=0.1, s=5, label='Kernel', c='red')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Residual (pred - true)')
    ax.set_title('Residuals by Age')
    ax.legend(fontsize=8); ax.set_ylim(-40, 40)

    # 4. Weight vector magnitude by dimension
    ax = axes[1, 0]
    sorted_w = np.sort(np.abs(w))[::-1]
    ax.plot(sorted_w, 'b-', linewidth=1)
    ax.set_xlabel('Feature dimension (sorted by |w|)')
    ax.set_ylabel('|w|')
    ax.set_title('Linear Probe: Feature Importance')
    ax.set_xlim(0, 500)

    # 5. Kernel dual coefficients vs age
    ax = axes[1, 1]
    ax.scatter(y_lab_age, alpha, alpha=0.5, s=10, c='red')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Sample Age')
    ax.set_ylabel('Dual Coefficient α')
    ax.set_title('Kernel: Which Training Samples Matter?')

    # 6. Where kernel helps
    ax = axes[1, 2]
    ax.scatter(y_val_age, improvement, alpha=0.15, s=5, c='green')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('True Age')
    ax.set_ylabel('Improvement (linear_err - kernel_err)')
    ax.set_title('Where Kernel Beats Linear (positive = kernel better)')

    plt.tight_layout()
    plt.savefig('linear_vs_kernel_probe.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved linear_vs_kernel_probe.png')


if __name__ == '__main__':
    main()
