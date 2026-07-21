"""Offline diagnosis of why RAPL may outperform HPL.

The diagnostic loader must expose the normally hidden unlabeled target as either
``((x_weak, x_strong), y)`` or ``(x_weak, x_strong, y)``. Targets must use the
same normalization as training. This script never updates either model.

Direct use (run from the repository root)::

    python analysis/legacy_hpl/analyze_rapl_vs_hpl.py --dataset utkface --data_dir DATA \
      --rapl_ckpt checkpoints/rapl.pt --hpl_ckpt checkpoints/hpl.pt
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unpack(batch):
    if len(batch) == 2:
        (x_w, x_s), y = batch
    else:
        x_w, x_s, y = batch
    return x_w, x_s, y


def _gradient(loss, model, retain_graph=False):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def _cosine(x, y):
    return F.cosine_similarity(x, y, dim=0, eps=1e-12).item()


def _spearman(x, y):
    # Sufficient for continuous diagnostics; ties are rare here.
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _report(name, error, aug_gap, proxy, weight, grad_cos,
            grad_norm, oracle_norm):
    q25, q50, q75, q90 = np.quantile(error, [.25, .50, .75, .90])
    good, bad = error <= q25, error >= q75
    stable_wrong = bad & (aug_gap <= np.quantile(aug_gap, .25))
    print(f"\n{name}")
    print(f"  |pseudo-y|: mean={error.mean():.3f}, median={q50:.3f}, "
          f"p75={q75:.3f}, p90={q90:.3f}")
    print(f"  grad cosine(pseudo, oracle): {np.nanmean(grad_cos):.6f}")
    print(f"  pseudo/oracle grad norm: {np.mean(grad_norm):.3e} / "
          f"{np.mean(oracle_norm):.3e} "
          f"(ratio={np.mean(grad_norm) / (np.mean(oracle_norm) + 1e-30):.3e})")
    print(f"  corr(proxy, true error): {_spearman(proxy, error):.3f}")
    print(f"  corr(weight, true error): {_spearman(weight, error):.3f}")
    print(f"  weight worst/best quartile: "
          f"{weight[bad].mean() / (weight[good].mean() + 1e-12):.3f}")
    print(f"  stable-but-wrong: {stable_wrong.mean():.1%} of samples, "
          f"mean weight={weight[stable_wrong].mean() if stable_wrong.any() else float('nan'):.3f}")


def _counterfactual_report(a):
    print("\nCounterfactual gradient test (same trained checkpoints)")
    print("  RAPL actual:   "
          f"cos={np.mean(a['rapl_grad_cos']): .6f}, "
          f"norm={np.mean(a['rapl_grad_norm']):.3e}")
    print("  RAPL uniform:  "
          f"cos={np.mean(a['rapl_uniform_grad_cos']): .6f}, "
          f"norm={np.mean(a['rapl_uniform_grad_norm']):.3e}")
    print("  RAPL shuffled: "
          f"cos={np.mean(a['rapl_shuffled_grad_cos']): .6f}, "
          f"norm={np.mean(a['rapl_shuffled_grad_norm']):.3e}")
    print("  RAPL oracle:   "
          f"cos={np.mean(a['rapl_oracle_weight_grad_cos']): .6f}, "
          f"norm={np.mean(a['rapl_oracle_weight_grad_norm']):.3e}")
    print("  HPL actual:    "
          f"cos={np.mean(a['hpl_grad_cos']): .6f}, "
          f"norm={np.mean(a['hpl_grad_norm']):.3e}")
    print("  HPL uniform:   "
          f"cos={np.mean(a['hpl_uniform_grad_cos']): .6f}, "
          f"norm={np.mean(a['hpl_uniform_grad_norm']):.3e}")


def _plot(a, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(9, 7))
    ax[0, 0].hist(a["y"], 35, density=True, alpha=.45, label="true")
    ax[0, 0].hist(a["rapl_pseudo"], 35, density=True, alpha=.45, label="RAPL")
    ax[0, 0].hist(a["hpl_pseudo"], 35, density=True, alpha=.45, label="HPL")
    ax[0, 0].set_title("Pseudo-label distribution")
    ax[0, 0].legend()

    ax[0, 1].hist(a["rapl_error"], 35, density=True, alpha=.55, label="RAPL")
    ax[0, 1].hist(a["hpl_error"], 35, density=True, alpha=.55, label="HPL")
    ax[0, 1].set_title("Absolute pseudo-label error")
    ax[0, 1].legend()

    for name, color in [("rapl", "C0"), ("hpl", "C1")]:
        error, weight = a[f"{name}_error"], a[f"{name}_weight"]
        bins = np.quantile(error, [0, .25, .50, .75, 1])
        group = np.clip(np.digitize(error, bins[1:-1]), 0, 3)
        normalized = weight / (weight.mean() + 1e-12)
        ax[1, 0].plot(range(1, 5),
                      [normalized[group == i].mean() for i in range(4)],
                      "o-", color=color, label=name.upper())
    ax[1, 0].set_xticks(range(1, 5), ["best", "Q2", "Q3", "worst"])
    ax[1, 0].set_title("Trust by true-error quartile")
    ax[1, 0].set_ylabel("weight / mean(weight)")
    ax[1, 0].legend()

    means = [np.nanmean(a["rapl_grad_cos"]), np.nanmean(a["hpl_grad_cos"])]
    ax[1, 1].bar(["RAPL", "HPL"], means, color=["C0", "C1"])
    ax[1, 1].axhline(0, color="black", linewidth=.8)
    ax[1, 1].set_ylim(-1, 1)
    ax[1, 1].set_title("Pseudo vs. oracle gradient")
    ax[1, 1].set_ylabel("cosine similarity")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(rapl_model, hpl_model, hpl_uncertainty, frozen_backbone, probe,
            diagnostic_loader, device, out_dir="diagnostics", max_batches=20,
            label_mean=0.0, label_std=1.0):
    """Compare each trained method with its own oracle gradient.

    ``label_mean`` and ``label_std`` only convert reported values back to the
    original target unit; gradients are computed in the training target unit.
    """
    modules = [rapl_model, hpl_model, hpl_uncertainty, frozen_backbone, probe]
    old_modes = [m.training for m in modules]
    for m in modules:
        m.eval()  # deterministic comparison; also avoids changing BN statistics

    values = {k: [] for k in [
        "y", "rapl_pseudo", "hpl_pseudo", "rapl_error", "hpl_error",
        "rapl_aug_gap", "hpl_aug_gap", "rapl_proxy", "hpl_proxy",
        "rapl_weight", "hpl_weight"]}
    values.update(rapl_grad_cos=[], hpl_grad_cos=[],
                  rapl_grad_norm=[], hpl_grad_norm=[],
                  rapl_oracle_norm=[], hpl_oracle_norm=[],
                  rapl_uniform_grad_cos=[], rapl_uniform_grad_norm=[],
                  rapl_shuffled_grad_cos=[], rapl_shuffled_grad_norm=[],
                  rapl_oracle_weight_grad_cos=[], rapl_oracle_weight_grad_norm=[],
                  hpl_uniform_grad_cos=[], hpl_uniform_grad_norm=[])

    try:
        for batch_id, batch in enumerate(diagnostic_loader):
            if max_batches is not None and batch_id >= max_batches:
                break
            x_w, x_s, y = _unpack(batch)
            x_w, x_s = x_w.to(device), x_s.to(device)
            y = y.to(device).reshape(-1)

            # RAPL: representation disagreement is the reliability proxy.
            r_w = rapl_model(x_w).detach().reshape(-1)
            r_s = rapl_model(x_s).reshape(-1)
            with torch.no_grad():
                anchor = probe(frozen_backbone(x_w)).reshape(-1)
                r_proxy = (r_w - anchor).abs()
                r_weight = 1.0 / (1.0 + r_proxy)
            r_error = (r_s - r_w).square()
            r_loss = (r_weight * r_error).mean()
            r_uniform_loss = r_error.mean()
            r_shuffled_loss = (r_weight[torch.randperm(
                len(r_weight), device=r_weight.device)] * r_error).mean()
            r_oracle_weight = 1.0 / (1.0 + (r_w - y).abs())
            r_oracle_weight_loss = (r_oracle_weight * r_error).mean()
            r_oracle = F.mse_loss(r_s, y)
            r_grad = _gradient(r_loss, rapl_model, True)
            r_uniform_grad = _gradient(r_uniform_loss, rapl_model, True)
            r_shuffled_grad = _gradient(r_shuffled_loss, rapl_model, True)
            r_oracle_weight_grad = _gradient(
                r_oracle_weight_loss, rapl_model, True)
            r_oracle_grad = _gradient(r_oracle, rapl_model)
            values["rapl_grad_cos"].append(_cosine(r_grad, r_oracle_grad))
            values["rapl_grad_norm"].append(r_grad.norm().item())
            values["rapl_oracle_norm"].append(r_oracle_grad.norm().item())
            values["rapl_uniform_grad_cos"].append(
                _cosine(r_uniform_grad, r_oracle_grad))
            values["rapl_uniform_grad_norm"].append(r_uniform_grad.norm().item())
            values["rapl_shuffled_grad_cos"].append(
                _cosine(r_shuffled_grad, r_oracle_grad))
            values["rapl_shuffled_grad_norm"].append(r_shuffled_grad.norm().item())
            values["rapl_oracle_weight_grad_cos"].append(
                _cosine(r_oracle_weight_grad, r_oracle_grad))
            values["rapl_oracle_weight_grad_norm"].append(
                r_oracle_weight_grad.norm().item())

            # HPL: detach uncertainty weights exactly as in the model update.
            h_w = hpl_model(x_w).detach().reshape(-1)
            h_s = hpl_model(x_s).reshape(-1)
            with torch.no_grad():
                h_in = torch.stack([h_s.detach() - h_w, h_s.detach()], dim=-1)
                h_proxy = hpl_uncertainty(h_in).reshape(-1)
                h_weight = torch.exp(-h_proxy) / 2.0
            h_loss = (h_weight * (h_s - h_w).square()).mean()
            h_uniform_loss = (h_s - h_w).square().mean()
            h_oracle = F.mse_loss(h_s, y)
            h_grad = _gradient(h_loss, hpl_model, True)
            h_uniform_grad = _gradient(h_uniform_loss, hpl_model, True)
            h_oracle_grad = _gradient(h_oracle, hpl_model)
            values["hpl_grad_cos"].append(_cosine(h_grad, h_oracle_grad))
            values["hpl_grad_norm"].append(h_grad.norm().item())
            values["hpl_oracle_norm"].append(h_oracle_grad.norm().item())
            values["hpl_uniform_grad_cos"].append(
                _cosine(h_uniform_grad, h_oracle_grad))
            values["hpl_uniform_grad_norm"].append(h_uniform_grad.norm().item())

            scale = abs(label_std)
            batch_values = {
                "y": y * label_std + label_mean,
                "rapl_pseudo": r_w * label_std + label_mean,
                "hpl_pseudo": h_w * label_std + label_mean,
                "rapl_error": (r_w - y).abs() * scale,
                "hpl_error": (h_w - y).abs() * scale,
                "rapl_aug_gap": (r_s.detach() - r_w).abs() * scale,
                "hpl_aug_gap": (h_s.detach() - h_w).abs() * scale,
                "rapl_proxy": r_proxy * scale,
                "hpl_proxy": h_proxy,
                "rapl_weight": r_weight,
                "hpl_weight": h_weight,
            }
            for key, value in batch_values.items():
                values[key].append(value.cpu().numpy())
    finally:
        for module, mode in zip(modules, old_modes):
            module.train(mode)

    arrays = {k: (np.asarray(v) if np.isscalar(v[0]) else np.concatenate(v))
              for k, v in values.items()}
    _report("RAPL", arrays["rapl_error"], arrays["rapl_aug_gap"],
            arrays["rapl_proxy"], arrays["rapl_weight"], arrays["rapl_grad_cos"],
            arrays["rapl_grad_norm"], arrays["rapl_oracle_norm"])
    _report("HPL", arrays["hpl_error"], arrays["hpl_aug_gap"],
            arrays["hpl_proxy"], arrays["hpl_weight"], arrays["hpl_grad_cos"],
            arrays["hpl_grad_norm"], arrays["hpl_oracle_norm"])
    _counterfactual_report(arrays)
    print("\nInterpretation: better alignment, a more negative weight/error "
          "correlation, and a lower worst/best ratio indicate a better filter.")
    print("Raw RAPL and HPL weights are on different scales; compare selectivity, "
          "not their absolute means.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rapl_vs_hpl.npz", **arrays)
    _plot(arrays, out / "rapl_vs_hpl.png")
    return arrays


# --------------------------- one-command CLI ---------------------------

class _ImagePairs(torch.utils.data.Dataset):
    def __init__(self, items, weak, strong, mean, std):
        self.items, self.weak, self.strong = items, weak, strong
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, target = self.items[i]
        image = Image.open(path).convert("RGB")
        y = torch.tensor((target - self.mean) / self.std, dtype=torch.float32)
        return self.weak(image), self.strong(image), y


def _image_loader(args, mean, std):
    from torch.utils.data import DataLoader
    from torchvision import transforms

    root = Path(args.data_dir)
    if args.dataset == "utkface":
        paths = list(root.glob("*.jpg")) + list(root.glob("*.png")) \
                + list(root.glob("*.jpeg"))
        items = []
        for path in paths:
            try:
                age = float(path.name.split("_")[0])
                if 0 <= age <= 120:
                    items.append((path, age))
            except (ValueError, IndexError):
                pass
    else:
        with open(root / "metadata.json") as f:
            metadata = json.load(f)
        items = [(Path(row["path"]), float(row["age"])) for row in metadata]
        items = [(p, y) for p, y in items if p.exists()]

    random.Random(args.seed).shuffle(items)
    n = len(items)
    n_test, n_val = int(.1 * n), int(.1 * n)
    train = items[n_test + n_val:]
    n_labeled = max(1, int(args.labeled_ratio * len(train)))
    unlabeled = train[n_labeled:]

    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    weak = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 10),
        transforms.ToTensor(), norm])
    return DataLoader(_ImagePairs(unlabeled, weak, strong, mean, std),
                      batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True)


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["utkface", "imdb_wiki"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rapl_ckpt", required=True)
    parser.add_argument("--hpl_ckpt", required=True)
    parser.add_argument("--labeled_ratio", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--out_dir", default="diagnostics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    rapl_ckpt, hpl_ckpt = _load(args.rapl_ckpt, device), _load(args.hpl_ckpt, device)

    from backbone import ResNet50Regressor
    from hpl import UncertaintyLearner

    rapl_model = ResNet50Regressor(pretrained=False).to(device)
    hpl_model = ResNet50Regressor(pretrained=False).to(device)
    uncertainty = UncertaintyLearner().to(device)
    frozen = ResNet50Regressor(pretrained=False).backbone.to(device)

    probe_state = rapl_ckpt["probe"]
    probe = nn.Linear(probe_state["weight"].shape[1], 1).to(device)
    rapl_model.load_state_dict(rapl_ckpt["model"])
    hpl_model.load_state_dict(hpl_ckpt["model"])
    uncertainty.load_state_dict(hpl_ckpt["uncertainty"])
    frozen.load_state_dict(rapl_ckpt["frozen_backbone"])
    probe.load_state_dict(probe_state)

    mean = float(rapl_ckpt["mean"])
    std = float(rapl_ckpt["std"])
    if not np.allclose([mean, std], [hpl_ckpt["mean"], hpl_ckpt["std"]]):
        raise ValueError("RAPL and HPL checkpoints used different label scalers/splits")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader = _image_loader(args, mean, std)
    analyze(rapl_model, hpl_model, uncertainty, frozen, probe, loader, device,
            args.out_dir, args.max_batches, mean, std)
    print(f"\nSaved: {Path(args.out_dir) / 'rapl_vs_hpl.png'}")
    print(f"Saved: {Path(args.out_dir) / 'rapl_vs_hpl.npz'}")


if __name__ == "__main__":
    main()
