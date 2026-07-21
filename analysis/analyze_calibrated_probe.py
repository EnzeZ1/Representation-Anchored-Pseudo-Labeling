"""
Calibrated probe: linear projection finds the age direction,
tiny MLP bends the output to fix nonlinearity.

Step 1: w·z + b → scalar score (linear, 2048→1)
Step 2: MLP(score) → calibrated age (1→64→1, fixes the camel curve)
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

    # Step 1: linear probe → scalar score
    print('Training linear probe...')
    linear = Ridge(alpha=1.0)
    linear.fit(X_lab, y_lab)

    scores_lab = linear.predict(X_lab).astype(np.float32)  # [n_lab]
    scores_val = linear.predict(X_val).astype(np.float32)  # [n_val]

    pred_linear_val = scores_val * y_std + y_mean
    mae_linear = np.abs(pred_linear_val - (y_val * y_std + y_mean)).mean()
    print(f'  Linear probe val MAE: {mae_linear:.2f} years')

    # Step 2: tiny MLP to calibrate the scalar score
    print('Training calibration MLP (1→64→1)...')
    mlp = nn.Sequential(
        nn.Linear(1, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(device)

    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    s_train = torch.tensor(scores_lab, device=device).unsqueeze(-1)
    y_train = torch.tensor(y_lab, device=device)

    for ep in range(500):
        pred = mlp(s_train).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_train)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            print(f'  ep {ep+1}: loss={loss.item():.4f}')

    mlp.eval()
    with torch.no_grad():
        s_val = torch.tensor(scores_val, device=device).unsqueeze(-1)
        pred_calibrated = mlp(s_val).squeeze(-1).cpu().numpy()

    pred_cal_val = pred_calibrated * y_std + y_mean
    y_val_age = y_val * y_std + y_mean
    mae_calibrated = np.abs(pred_cal_val - y_val_age).mean()
    print(f'  Calibrated probe val MAE: {mae_calibrated:.2f} years')
    print(f'  Improvement: {mae_linear - mae_calibrated:.2f} years')

    # By age group
    print(f'\n=== By Age Group ===')
    print(f'{"Group":<15} {"Linear MAE":<15} {"Calibrated MAE":<15} {"Improvement":<15}')
    for lo, hi, name in [(0,10,'Children'),(10,20,'Teens'),(20,40,'Young Adult'),
                          (40,60,'Middle Age'),(60,80,'Seniors'),(80,120,'Elderly')]:
        mask = (y_val_age >= lo) & (y_val_age < hi)
        if mask.sum() == 0: continue
        ml = np.abs(pred_linear_val[mask] - y_val_age[mask]).mean()
        mc = np.abs(pred_cal_val[mask] - y_val_age[mask]).mean()
        print(f'{name:<15} {ml:<15.2f} {mc:<15.2f} {ml-mc:<+15.2f}')

    # === Plots ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Linear Probe → MLP Calibration', fontsize=14)

    # 1. The calibration curve: what MLP learned
    ax = axes[0]
    s_range = torch.linspace(scores_lab.min() - 0.5, scores_lab.max() + 0.5, 200).unsqueeze(-1).to(device)
    with torch.no_grad():
        cal_range = mlp(s_range).squeeze(-1).cpu().numpy()
    s_range_np = s_range.cpu().numpy().flatten()
    ax.plot(s_range_np * y_std + y_mean, cal_range * y_std + y_mean, 'r-', linewidth=2, label='MLP calibration')
    ax.plot(s_range_np * y_std + y_mean, s_range_np * y_std + y_mean, 'k--', alpha=0.3, label='Identity (linear)')
    ax.scatter(scores_lab * y_std + y_mean, y_lab * y_std + y_mean, alpha=0.2, s=5, c='blue', label='Training data')
    ax.set_xlabel('Linear Probe Score (age)')
    ax.set_ylabel('Calibrated Output / True Age')
    ax.set_title('Calibration Curve')
    ax.legend(fontsize=8)

    # 2. Before vs after calibration
    ax = axes[1]
    ax.scatter(y_val_age, pred_linear_val, alpha=0.1, s=5, c='blue', label=f'Linear (MAE={mae_linear:.1f})')
    ax.scatter(y_val_age, pred_cal_val, alpha=0.1, s=5, c='red', label=f'Calibrated (MAE={mae_calibrated:.1f})')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Predicted Age')
    ax.set_title('Predictions vs Truth')
    ax.legend(fontsize=8)

    # 3. Residuals
    ax = axes[2]
    res_linear = pred_linear_val - y_val_age
    res_cal = pred_cal_val - y_val_age
    ax.scatter(y_val_age, res_linear, alpha=0.1, s=5, c='blue', label='Linear')
    ax.scatter(y_val_age, res_cal, alpha=0.1, s=5, c='red', label='Calibrated')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Residual (pred - true)')
    ax.set_title('Residuals')
    ax.legend(fontsize=8)
    ax.set_ylim(-40, 40)

    plt.tight_layout()
    plt.savefig('calibrated_probe.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved calibrated_probe.png')


if __name__ == '__main__':
    main()
