"""
Feature representation analysis by age group.

Extract ResNet50 features for all images, average by age decade,
analyze directions, distances, and separability.
"""

import sys
from pathlib import Path

# Keep repository-root imports working when invoked as
# ``python analysis/analyze_age_features.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity


def main():
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('data_dir', default='data/utkface_all')
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'dinov2'])
    a = parser.parse_args()
    data_dir = a.data_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from models.backbone import ResNet50Regressor
    import random; random.seed(0)
    np.random.seed(0); torch.manual_seed(0)

    if a.backbone == 'dinov2':
        from models.dinov2_backbone import DINOv2Regressor
        backbone = DINOv2Regressor(size='small').to(device)
    else:
        backbone = ResNet50Regressor(pretrained=True).to(device)
    backbone.eval()
    print(f'Using backbone: {a.backbone}, feature_dim={backbone.feature_dim}')

    from PIL import Image
    from torchvision import transforms
    root = Path(data_dir)
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print('Extracting features...')
    files = list(root.glob('*.jpg')) + list(root.glob('*.png'))
    
    # Parse ages from filenames
    valid_files, valid_ages = [], []
    for p in files:
        try:
            age = float(p.name.split('_')[0])
            if 0 <= age <= 120:
                valid_files.append(p)
                valid_ages.append(age)
        except: continue
    
    # Sample 5K for speed
    import random as rng
    if len(valid_files) > 5000:
        idx = rng.sample(range(len(valid_files)), 5000)
        valid_files = [valid_files[i] for i in idx]
        valid_ages = [valid_ages[i] for i in idx]
    
    from tqdm import tqdm
    all_feats = []
    for p in tqdm(valid_files):
        img = Image.open(p).convert('RGB')
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            all_feats.append(backbone.backbone(x).squeeze(0).cpu())

    feats = torch.stack(all_feats)  # [N, 2048]
    ages = np.array(valid_ages)
    print(f'Total: {len(ages)} images, features: {feats.shape}')

    # === Bin by age decades ===
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 100)]
    bin_labels = []
    bin_means = []
    bin_counts = []

    print(f'\n=== Average features by age group ===')
    for lo, hi in bins:
        mask = (ages >= lo) & (ages < hi)
        n = mask.sum()
        if n == 0: continue
        mean_feat = feats[mask].mean(dim=0)
        bin_labels.append(f'{lo}-{hi}')
        bin_means.append(mean_feat)
        bin_counts.append(n)
        print(f'  {lo:3d}-{hi:3d}: {n:5d} samples, '
              f'||mean_v||={mean_feat.norm():.2f}')

    bin_means = torch.stack(bin_means)  # [n_bins, 2048]
    n_bins = len(bin_labels)

    # === Cosine similarity between bin centroids ===
    print(f'\n=== Cosine similarity between age group centroids ===')
    cos_matrix = np.zeros((n_bins, n_bins))
    print(f'{"":>8}', end='')
    for l in bin_labels:
        print(f'{l:>8}', end='')
    print()
    for i in range(n_bins):
        print(f'{bin_labels[i]:>8}', end='')
        for j in range(n_bins):
            cos = cosine_similarity(
                bin_means[i].unsqueeze(0),
                bin_means[j].unsqueeze(0)).item()
            cos_matrix[i, j] = cos
            print(f'{cos:8.3f}', end='')
        print()

    # === L2 distance between consecutive bins ===
    print(f'\n=== L2 distance between consecutive age group centroids ===')
    for i in range(n_bins - 1):
        dist = (bin_means[i] - bin_means[i+1]).norm().item()
        cos = cos_matrix[i, i+1]
        print(f'  {bin_labels[i]} → {bin_labels[i+1]}: '
              f'L2={dist:.2f}, cos={cos:.3f}')

    # === Direction analysis: is there a consistent "aging direction"? ===
    print(f'\n=== Aging direction analysis ===')
    # Compute pairwise difference vectors between consecutive bins
    diff_vectors = []
    for i in range(n_bins - 1):
        diff = bin_means[i+1] - bin_means[i]
        diff = diff / diff.norm()  # normalize
        diff_vectors.append(diff)

    # How consistent are the aging directions?
    print('Cosine similarity between consecutive aging directions:')
    for i in range(len(diff_vectors) - 1):
        cos = cosine_similarity(
            diff_vectors[i].unsqueeze(0),
            diff_vectors[i+1].unsqueeze(0)).item()
        print(f'  ({bin_labels[i]}→{bin_labels[i+1]}) vs '
              f'({bin_labels[i+1]}→{bin_labels[i+2]}): cos={cos:.3f}')

    # Global aging direction: young centroid → old centroid
    global_dir = bin_means[-1] - bin_means[0]
    global_dir = global_dir / global_dir.norm()
    print(f'\nGlobal aging direction (youngest → oldest):')
    for i in range(len(diff_vectors)):
        cos = cosine_similarity(
            diff_vectors[i].unsqueeze(0),
            global_dir.unsqueeze(0)).item()
        print(f'  {bin_labels[i]}→{bin_labels[i+1]} alignment with global: cos={cos:.3f}')

    # === Project all samples onto the global aging direction ===
    projections = (feats @ global_dir).numpy()

    # === Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('ResNet50 Feature Representations by Age', fontsize=14)

    # 1. Cosine similarity heatmap
    ax = axes[0, 0]
    im = ax.imshow(cos_matrix, cmap='RdYlBu_r', vmin=0.8, vmax=1.0)
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
    ax.set_yticks(range(n_bins)); ax.set_yticklabels(bin_labels, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title('Cosine Similarity Between Age Centroids')

    # 2. L2 distance heatmap
    ax = axes[0, 1]
    l2_matrix = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            l2_matrix[i, j] = (bin_means[i] - bin_means[j]).norm().item()
    im = ax.imshow(l2_matrix, cmap='viridis')
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
    ax.set_yticks(range(n_bins)); ax.set_yticklabels(bin_labels, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title('L2 Distance Between Age Centroids')

    # 3. Projection onto aging direction vs true age
    ax = axes[0, 2]
    ax.scatter(ages, projections, alpha=0.05, s=3)
    ax.set_xlabel('True Age'); ax.set_ylabel('Projection onto aging direction')
    ax.set_title('Feature Projection onto Global Aging Direction')
    # Add bin means
    for i, (lo, hi) in enumerate(bins[:n_bins]):
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() > 0:
            ax.scatter(ages[mask].mean(), projections[mask].mean(),
                      s=100, c='red', zorder=5, edgecolors='black')

    # 4. Consecutive distance
    ax = axes[1, 0]
    dists = [(bin_means[i] - bin_means[i+1]).norm().item() for i in range(n_bins-1)]
    x_pos = range(len(dists))
    ax.bar(x_pos, dists, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{bin_labels[i]}→\n{bin_labels[i+1]}' for i in range(n_bins-1)],
                       fontsize=7)
    ax.set_ylabel('L2 Distance')
    ax.set_title('Distance Between Consecutive Age Groups')

    # 5. Aging direction consistency
    ax = axes[1, 1]
    cos_consecutive = [cosine_similarity(
        diff_vectors[i].unsqueeze(0), diff_vectors[i+1].unsqueeze(0)).item()
        for i in range(len(diff_vectors)-1)]
    cos_global = [cosine_similarity(
        diff_vectors[i].unsqueeze(0), global_dir.unsqueeze(0)).item()
        for i in range(len(diff_vectors))]
    ax.plot(cos_global, 'bo-', label='Alignment with global dir')
    ax.plot(cos_consecutive, 'rs-', label='Consecutive consistency')
    ax.set_xticks(range(len(cos_global)))
    ax.set_xticklabels([f'{bin_labels[i]}→{bin_labels[i+1]}' for i in range(n_bins-1)],
                       rotation=45, fontsize=7)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Aging Direction Consistency')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # 6. Within-bin spread vs between-bin distance
    ax = axes[1, 2]
    within_std = []
    for i, (lo, hi) in enumerate(bins[:n_bins]):
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() > 1:
            centered = feats[mask] - bin_means[i]
            within_std.append(centered.norm(dim=1).mean().item())
        else:
            within_std.append(0)
    between_dist = [(bin_means[i] - bin_means[i+1]).norm().item()
                    for i in range(n_bins-1)]
    x = range(n_bins)
    ax.bar(x, within_std, color='lightcoral', alpha=0.8, label='Within-group spread')
    ax.bar([i+0.3 for i in range(n_bins-1)], between_dist,
           width=0.3, color='steelblue', alpha=0.8, label='Between-group distance')
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_ylabel('Distance')
    ax.set_title('Within-group Spread vs Between-group Distance')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('feature_age_analysis.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved feature_age_analysis.png')


if __name__ == '__main__':
    main()
