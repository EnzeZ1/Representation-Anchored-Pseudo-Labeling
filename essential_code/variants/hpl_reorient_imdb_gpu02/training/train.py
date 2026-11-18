import argparse, csv, json, os, random, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.backbone import ResNet50Regressor
from models.dinov2_backbone import DINOv2Regressor
from models.dinov3_convnext_backbone import DINOv3ConvNextTinyRegressor
from training.probe_filter import ProbeFilteredTrainer, ProbeConfig, train_probe
from data_processing.stsb import make_data_stsb, TextRegressor
from data_processing.hpl_data import make_data_hpl_official
from data_processing.utkface_protocol import (
    IMAGE_SIZE,
    build_evaluation_transform,
    build_labeled_transform,
    build_strong_transform,
    build_weak_transform,
    dataloader_generator,
    loader_metadata,
    load_cohort,
    load_seed_manifest,
    manifest_items,
    runtime_metadata,
    seed_dataloader_worker,
    validate_cohort,
)
from baselines.benchmark_io import (
    checkpoint_size, write_history, write_json, write_npz,
)


def _progress(args, marker, **details):
    output = getattr(args, 'benchmark_output_dir', None)
    if not output:
        return
    path = Path(output) / 'progress.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps({'timestamp': time.time(), 'marker': marker, **details}, sort_keys=True) + '\n')


def _memory_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def _rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return None


def _start_resource_sampler(args, device):
    output = getattr(args, 'benchmark_output_dir', None)
    if not output:
        return None, None
    stop = threading.Event()
    path = Path(output) / 'torch_resource_trace.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    def sample():
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['timestamp','pytorch_allocated_bytes','pytorch_reserved_bytes','host_rss_bytes','host_available_bytes'])
            writer.writeheader(); handle.flush()
            while not stop.is_set():
                writer.writerow({'timestamp':time.time(),
                    'pytorch_allocated_bytes':torch.cuda.memory_allocated(device) if device.type=='cuda' else 0,
                    'pytorch_reserved_bytes':torch.cuda.memory_reserved(device) if device.type=='cuda' else 0,
                    'host_rss_bytes':_rss_bytes(),'host_available_bytes':_memory_available_bytes()})
                handle.flush(); stop.wait(2)
    thread = threading.Thread(target=sample, daemon=True); thread.start(); return stop, thread


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_model(args, device):
    """Create the appropriate backbone model."""
    if args.backbone == 'dinov2':
        return DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
    return ResNet50Regressor(pretrained=args.pretrained).to(device)


def age_from_name(p):
    try: return float(p.name.split('_')[0])
    except Exception: return None


class UTKFace(Dataset):
    def __init__(self, items, tfm, mean=None, std=None, unlabeled=False, strong_tfm=None):
        self.items, self.tfm, self.strong_tfm = items, tfm, strong_tfm
        self.mean, self.std, self.unlabeled = mean, std, unlabeled

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert('RGB')
        if self.unlabeled:
            if self.strong_tfm is None: return self.tfm(img)
            return self.tfm(img), self.strong_tfm(img)
        y = torch.tensor((y - self.mean) / self.std, dtype=torch.float32)
        return self.tfm(img), y


def make_data(args):
    if args.dataset == 'imdb_wiki':
        return make_data_imdb_wiki(args)
    if args.dataset == 'stsb':
        return make_data_stsb(args)
    if args.dataset == 'utkface_official':
        return make_data_hpl_official(args)
    if args.method == 'probe':
        return make_data_utkface(args)
    return make_data_utkface_legacy(args)


def make_data_imdb_wiki(args):
    if getattr(args, 'benchmark_manifest', None):
        from data_processing.imdb_wiki_protocol import (
            build_evaluation_transform as imdb_evaluation_transform,
            build_labeled_transform as imdb_labeled_transform,
            build_strong_transform as imdb_strong_transform,
            build_weak_transform as imdb_weak_transform,
            dataloader_generator as imdb_dataloader_generator,
            load_cohort as load_imdb_cohort,
            load_seed_manifest as load_imdb_manifest,
            loader_metadata as imdb_loader_metadata,
            manifest_items as imdb_manifest_items,
            runtime_metadata as imdb_runtime_metadata,
            seed_dataloader_worker as imdb_seed_worker,
            validate_cohort as validate_imdb_cohort,
        )
        cohort_path = Path(__file__).resolve().parents[1] / 'data_processing/splits/imdb_wiki_dir_cohort_v1.json'
        manifest_path = Path(args.benchmark_manifest)
        cohort = load_imdb_cohort(cohort_path)
        validate_imdb_cohort(cohort, args.data_dir)
        manifest = load_imdb_manifest(manifest_path, cohort)
        if manifest['seed'] != args.seed or not np.isclose(manifest['labeled_ratio'], args.labeled_ratio):
            raise ValueError('IMDB-WIKI manifest seed or labeled ratio does not match the command.')
        mean = float(manifest['label_scaler']['mean']); std = float(manifest['label_scaler']['std'])
        lab = UTKFace(imdb_manifest_items(cohort, manifest['labeled_indices'], args.data_dir),
                      imdb_labeled_transform(), mean, std)
        unlab = UTKFace(imdb_manifest_items(cohort, manifest['unlabeled_indices'], args.data_dir),
                        imdb_weak_transform(), unlabeled=True, strong_tfm=imdb_strong_transform())
        val = UTKFace(imdb_manifest_items(cohort, manifest['splits']['validation'], args.data_dir),
                      imdb_evaluation_transform(), mean, std)
        test = UTKFace(imdb_manifest_items(cohort, manifest['splits']['test'], args.data_dir),
                       imdb_evaluation_transform(), mean, std)
        specs = {'labeled': (lab, True, True), 'unlabeled': (unlab, True, True),
                 'validation': (val, False, False), 'test': (test, False, False)}
        loaders = {}; metadata = {}
        for role, (dataset, shuffle, drop_last) in specs.items():
            loaders[role] = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                num_workers=args.workers, pin_memory=True, drop_last=drop_last,
                worker_init_fn=imdb_seed_worker, generator=imdb_dataloader_generator(args.seed, role))
            metadata[role] = imdb_loader_metadata(seed=args.seed, role=role,
                batch_size=args.batch_size, num_workers=args.workers, shuffle=shuffle,
                drop_last=drop_last, sampler='RandomSampler' if shuffle else 'SequentialSampler',
                pin_memory=True)
        args.benchmark_metadata = {
            'dataset': 'IMDB-WIKI-DIR', 'protocol_version': manifest['protocol_version'],
            'transform_version': manifest['transform_version'], 'cohort_sha256': cohort['cohort_sha256'],
            'cohort_path': str(cohort_path.resolve()), 'manifest_path': str(manifest_path.resolve()),
            'manifest_seed': manifest['seed'], 'labeled_ratio': manifest['labeled_ratio'],
            'counts': manifest['counts'], 'label_scaler': manifest['label_scaler'],
            'dataloaders': metadata, 'runtime': imdb_runtime_metadata(),
        }
        return loaders['labeled'], loaders['unlabeled'], loaders['validation'], loaders['test'], mean, std
    import json
    meta_path = Path(args.data_dir) / 'metadata.json'
    if not meta_path.exists():
        raise RuntimeError(f'{meta_path} not found. Run preprocess_imdb_wiki.py first.')
    with open(meta_path) as f:
        all_items = json.load(f)
    items = [(Path(item['path']), float(item['age'])) for item in all_items]
    items = [(p, y) for p, y in items if p.exists()]
    if not items:
        raise RuntimeError('No valid images found in metadata.json')
    random.shuffle(items)

    n = len(items); n_test = int(0.1 * n); n_val = int(0.1 * n)
    test, val, train = items[:n_test], items[n_test:n_test+n_val], items[n_test+n_val:]
    n_lab = max(1, int(args.labeled_ratio * len(train)))
    labeled, unlabeled = train[:n_lab], train[n_lab:]

    y = np.array([v for _, v in labeled], dtype=np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-6)

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    weak = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 10), transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(args.img_size), transforms.ToTensor(), norm])

    use_strong = args.method in ('hpl', 'probe')
    lab = UTKFace(labeled, weak, y_mean, y_std)
    unlab = UTKFace(unlabeled, weak, unlabeled=True, strong_tfm=strong if use_strong else None)
    val_ds = UTKFace(val, eval_tfm, y_mean, y_std)
    test_ds = UTKFace(test, eval_tfm, y_mean, y_std)

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=drop)

    print(f'IMDB-WIKI: labeled={len(labeled)}, unlabeled={len(unlabeled)}, val={len(val)}, test={len(test)}')
    print(f'label scaler: mean={y_mean:.2f}, std={y_std:.2f}')
    return loader(lab), loader(unlab), loader(val_ds, False, False), loader(test_ds, False, False), y_mean, y_std


def make_data_utkface(args):
    """Create RAPL loaders from the persisted controlled-benchmark manifest."""
    splits_dir = Path(__file__).resolve().parents[1] / 'data_processing' / 'splits'
    cohort_path = splits_dir / 'utkface_cohort_v1.json'
    manifest_arg = getattr(args, 'benchmark_manifest', None)
    manifest_path = (Path(manifest_arg) if manifest_arg else
                     splits_dir / f'utkface_ratio_0.05_seed_{args.seed}.json')
    target_backbone = getattr(args, 'backbone', 'resnet50')
    probe_backbone = getattr(args, 'probe_backbone', None) or target_backbone
    if target_backbone not in ('resnet50', 'dinov2', 'dinov3_convnext_tiny') or probe_backbone != target_backbone:
        raise ValueError('The UTKFace benchmark requires matching supported target and probe backbones.')
    if not getattr(args, 'pretrained', True):
        raise ValueError('The UTKFace benchmark requires ImageNet-pretrained ResNet-50.')
    if args.img_size != IMAGE_SIZE:
        raise ValueError(f'The UTKFace benchmark requires --img_size {IMAGE_SIZE}.')

    cohort = load_cohort(cohort_path)
    validate_cohort(cohort, args.data_dir)
    manifest = load_seed_manifest(manifest_path, cohort)
    if manifest['seed'] != args.seed:
        raise ValueError(f"Manifest seed {manifest['seed']} does not match --seed {args.seed}.")
    if not np.isclose(manifest['labeled_ratio'], args.labeled_ratio):
        raise ValueError(
            f"Manifest ratio {manifest['labeled_ratio']} does not match "
            f"--labeled_ratio {args.labeled_ratio}."
        )

    labeled = manifest_items(cohort, manifest['labeled_indices'], args.data_dir)
    unlabeled = manifest_items(cohort, manifest['unlabeled_indices'], args.data_dir)
    val = manifest_items(cohort, manifest['splits']['validation'], args.data_dir)
    test = manifest_items(cohort, manifest['splits']['test'], args.data_dir)
    y_mean = float(manifest['label_scaler']['mean'])
    y_std = float(manifest['label_scaler']['std'])

    lab = UTKFace(labeled, build_labeled_transform(), y_mean, y_std)
    unlab = UTKFace(
        unlabeled, build_weak_transform(), unlabeled=True,
        strong_tfm=build_strong_transform(),
    )
    val_ds = UTKFace(val, build_evaluation_transform(), y_mean, y_std)
    test_ds = UTKFace(test, build_evaluation_transform(), y_mean, y_std)

    specifications = {
        'labeled': (lab, True, True),
        'unlabeled': (unlab, True, True),
        'validation': (val_ds, False, False),
        'test': (test_ds, False, False),
    }
    loaders, metadata = {}, {}
    for role, (dataset, shuffle, drop_last) in specifications.items():
        loaders[role] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=drop_last,
            worker_init_fn=seed_dataloader_worker,
            generator=dataloader_generator(args.seed, role),
        )
        metadata[role] = loader_metadata(
            seed=args.seed,
            role=role,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=shuffle,
            drop_last=drop_last,
            sampler=('RandomSampler' if shuffle else 'SequentialSampler'),
            pin_memory=True,
        )

    args.benchmark_metadata = {
        'protocol_version': manifest['protocol_version'],
        'transform_version': manifest['transform_version'],
        'cohort_sha256': cohort['cohort_sha256'],
        'cohort_path': str(cohort_path.resolve()),
        'manifest_path': str(manifest_path.resolve()),
        'manifest_seed': manifest['seed'],
        'labeled_ratio': manifest['labeled_ratio'],
        'counts': manifest['counts'],
        'label_scaler': manifest['label_scaler'],
        'dataloaders': metadata,
        'runtime': runtime_metadata(),
    }
    print(f"UTKFace benchmark manifest: {manifest_path}")
    print(f"UTKFace cohort: {cohort['cohort_sha256']}")
    print(f"UTKFace: labeled={len(labeled)}, unlabeled={len(unlabeled)}, "
          f"val={len(val)}, test={len(test)}")
    print(f'label scaler: mean={y_mean:.6f}, std={y_std:.6f}')
    print('DataLoader settings: ' + json.dumps(metadata, sort_keys=True))
    return (loaders['labeled'], loaders['unlabeled'], loaders['validation'],
            loaders['test'], y_mean, y_std)


def make_data_utkface_legacy(args):
    """Original non-benchmark UTKFace loader retained for non-RAPL methods."""
    root = Path(args.data_dir)
    files = list(root.glob('*.jpg')) + list(root.glob('*.png')) + list(root.glob('*.jpeg'))
    items = [(p, age_from_name(p)) for p in files]
    items = [(p, y) for p, y in items if y is not None and 0 <= y <= 120]
    if not items: raise RuntimeError('No UTKFace images found.')
    random.shuffle(items)

    n = len(items); n_test = int(0.1 * n); n_val = int(0.1 * n)
    test, val, train = items[:n_test], items[n_test:n_test+n_val], items[n_test+n_val:]
    n_lab = max(1, int(args.labeled_ratio * len(train)))
    labeled, unlabeled = train[:n_lab], train[n_lab:]

    y = np.array([v for _, v in labeled], dtype=np.float32)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-6)

    norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    weak = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)), transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 10), transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(args.img_size), transforms.ToTensor(), norm])

    use_strong = args.method in ('hpl', 'probe')
    lab = UTKFace(labeled, weak, y_mean, y_std)
    unlab = UTKFace(unlabeled, weak, unlabeled=True, strong_tfm=strong if use_strong else None)
    val_ds = UTKFace(val, eval_tfm, y_mean, y_std)
    test_ds = UTKFace(test, eval_tfm, y_mean, y_std)

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True, drop_last=drop)

    print(f'UTKFace: labeled={len(labeled)}, unlabeled={len(unlabeled)}, val={len(val)}, test={len(test)}')
    print(f'label scaler: mean={y_mean:.2f}, std={y_std:.2f}')
    return loader(lab), loader(unlab), loader(val_ds, False, False), loader(test_ds, False, False), y_mean, y_std


def cycle(loader):
    while True:
        for b in loader: yield b


@torch.no_grad()
def eval_mae(model, loader, mean, std, device, return_predictions=False):
    model.eval(); ps=[]; ys=[]
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        ps.append((p.cpu() * std + mean)); ys.append((y.cpu() * std + mean))
    p, y = torch.cat(ps), torch.cat(ys)
    mae = (p-y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    if return_predictions:
        return mae, r2, p.numpy(), y.numpy()
    return mae, r2


# ── Supervised ──

def train_supervised(args, loaders, scaler, device):
    lab, _, val, test = loaders
    mean, std = scaler
    model = ResNet50Regressor(pretrained=args.pretrained).to(device)
    opt = torch.optim.Adam([{'params': model.backbone.parameters(), 'lr': args.lr},
                            {'params': model.head.parameters(), 'lr': args.fc_lr}])
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for x, y in lab:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = F.mse_loss(model(x), y); loss.backward(); opt.step()
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST supervised: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')


def train_supervised_text(args, loaders, scaler, device):
    lab, _, val, test = loaders
    mean, std = scaler
    feat_dim = next(iter(lab))[0].shape[1]
    model = TextRegressor(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for x, y in lab:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = F.mse_loss(model(x), y); loss.backward(); opt.step()
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in val:
                ps.append(model(x.to(device)).cpu() * std + mean)
                ys.append(y.cpu() * std + mean)
        p, y = torch.cat(ps), torch.cat(ys)
        mae = (p - y).abs().mean().item()
        r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        if mae < best_val:
            best_val = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in test:
            ps.append(model(x.to(device)).cpu() * std + mean)
            ys.append(y.cpu() * std + mean)
    p, y = torch.cat(ps), torch.cat(ys)
    mae = (p - y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    print(f'TEST supervised: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')


# ── Probe (RAPL) ──

def train_probe_method(args, loaders, scaler, device):
    if args.dataset == 'stsb':
        return train_probe_text(args, loaders, scaler, device)

    lab, unlab, val, test = loaders
    mean, std = scaler
    is_image_benchmark = hasattr(args, 'benchmark_metadata')
    if is_image_benchmark and not args.save:
        raise ValueError('Benchmark training requires --save for best-checkpoint restoration.')
    if is_image_benchmark and not getattr(args, 'benchmark_output_dir', None):
        raise ValueError('Benchmark training requires --benchmark_output_dir.')

    started = time.time()
    sampler_stop, sampler_thread = _start_resource_sampler(args, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    _progress(args, 'model construction started')
    # Frozen backbone for probe — decoupled from trainable backbone
    _pb = args.probe_backbone or args.backbone
    if _pb == 'dinov2':
        frozen = DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
    elif _pb == 'dinov3_convnext_tiny':
        frozen = DINOv3ConvNextTinyRegressor().to(device)
    else:
        frozen = ResNet50Regressor(pretrained=True).to(device)
    frozen.eval()
    for parameter in frozen.backbone.parameters():
        parameter.requires_grad_(False)
    if _pb in ('dinov2', 'dinov3_convnext_tiny'):
        assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())
    _progress(args, 'frozen-probe feature extraction started')
    probe = train_probe(frozen.backbone, lab, device, progress_callback=lambda marker: _progress(args, marker))
    probe = probe.to(device)

    # Trainable model
    if args.backbone == 'dinov2':
        model = DINOv2Regressor(size={'s':'small','b':'base','l':'large'}[args.dino]).to(device)
        opt = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.head.parameters(), 'lr': 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.backbone == 'dinov3_convnext_tiny':
        model = DINOv3ConvNextTinyRegressor().to(device)
        opt = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.head.parameters(), 'lr': 1e-4},
        ], weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        model = ResNet50Regressor(pretrained=args.pretrained).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    _progress(args, 'model construction completed')

    cfg = ProbeConfig(lambda_u=args.lambda_u)
    trainer = ProbeFilteredTrainer(model, frozen.backbone, probe, opt, cfg,
                                   progress_callback=lambda marker: _progress(args, marker))
    _progress(args, 'train DataLoader constructed')

    lit, uit = cycle(lab), cycle(unlab)
    best_val, best_epoch = float('inf'), None
    best_state = None
    history = []
    for ep in range(args.epochs):
        epoch_started = time.time()
        model.train()
        totals = {'loss_total': 0.0, 'loss_sup': 0.0, 'loss_u': 0.0}
        steps = max(len(lab), len(unlab))
        for _ in range(steps):
            x_l, y_l = next(lit); x_u_w, x_u_s = next(uit)
            if ep == 0 and not totals['loss_total']:
                _progress(args, 'first weak batch loaded')
                _progress(args, 'first strong batch loaded')
            step_metrics = trainer.step(x_l.to(device), y_l.to(device),
                                       x_u_w.to(device), x_u_s.to(device))
            for key in totals:
                totals[key] += step_metrics[key]
        scheduler.step()
        mae, r2 = eval_mae(model, val, mean, std, device)
        print(f'ep {ep+1}: val_mae={mae:.3f}, r2={r2:.4f}')
        _progress(args, 'first epoch completed' if ep == 0 else 'epoch completed', epoch=ep + 1)
        improved = mae < best_val
        history.append({
            'epoch': ep + 1,
            'loss_total': totals['loss_total'] / steps,
            'loss_supervised': totals['loss_sup'] / steps,
            'loss_unlabeled': totals['loss_u'] / steps,
            'validation_mae': mae,
            'validation_r2': r2,
            'learning_rate': opt.param_groups[0]['lr'],
            'best_so_far': int(improved),
            'elapsed_seconds': time.time() - epoch_started,
        })
        if improved:
            best_val = mae
            best_epoch = ep + 1
            if is_image_benchmark:
                import os
                os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
                torch.save({
                    'checkpoint_version': 'rapl-utkface-benchmark-v1',
                    'epoch': best_epoch,
                    'validation_mae_years': best_val,
                    'model': model.state_dict(),
                    'probe': probe.state_dict(),
                    'frozen_backbone': ({
                        'identifier': frozen.weight_identifier,
                        'model_name': frozen.model_name,
                        'url': frozen.weight_url,
                        'checksum_sha256': ('b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
                                            if args.dino == 's' else None),
                        'frozen': True,
                    } if _pb == 'dinov2' else ({
                        'identifier': frozen.weight_identifier,
                        'model_name': frozen.model_name,
                        'revision': frozen.weight_revision,
                        'checksum_sha256': frozen.weight_checksum_sha256,
                        'frozen': True,
                    } if _pb == 'dinov3_convnext_tiny' else {
                        'identifier': 'torchvision.ResNet50_Weights.IMAGENET1K_V1',
                        'url': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                        'checksum_identifier': 'filename-sha256-prefix:0676ba61',
                        'frozen': True,
                    })),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'mean': mean,
                    'std': std,
                    'probe_config': vars(cfg),
                    'run_config': vars(args),
                    'benchmark': args.benchmark_metadata,
                }, args.save)
                print(f'Saved new best checkpoint to {args.save}')
                _progress(args, 'best checkpoint written', epoch=best_epoch, validation_mae=best_val)
            else:
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_epoch is None:
        raise RuntimeError('No checkpoint was selected; --epochs must be at least 1.')
    if is_image_benchmark:
        checkpoint = torch.load(args.save, map_location=device, weights_only=False)
        assert checkpoint['epoch'] == best_epoch
        assert checkpoint['validation_mae_years'] == best_val
        model.load_state_dict(checkpoint['model'])
        probe.load_state_dict(checkpoint['probe'])
        print(f'Restored best checkpoint from epoch {best_epoch} '
              f'(val_mae={best_val:.6f} years)')
        _progress(args, 'best checkpoint reloaded', epoch=best_epoch, validation_mae=best_val)
        _progress(args, 'validation metadata verified', epoch=best_epoch, validation_mae=best_val)
    else:
        model.load_state_dict(best_state)

    # The shared test set is evaluated exactly once, after best-checkpoint restoration.
    if is_image_benchmark:
        mae, r2, predictions, targets = eval_mae(
            model, test, mean, std, device, return_predictions=True
        )
    else:
        mae, r2 = eval_mae(model, test, mean, std, device)
    print(f'TEST probe: mae={mae:.3f}, r2={r2:.4f} (best val epoch)')
    if is_image_benchmark:
        _progress(args, 'test evaluation completed', test_mae=mae, test_r2=r2)

    if is_image_benchmark:
        output_dir = Path(args.benchmark_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_history(output_dir / 'history.csv', history)
        if args.dataset == 'imdb_wiki':
            from data_processing.imdb_wiki_protocol import load_cohort as load_run_cohort, load_seed_manifest as load_run_manifest
        else:
            load_run_cohort, load_run_manifest = load_cohort, load_seed_manifest
        cohort = load_run_cohort(args.benchmark_metadata['cohort_path'])
        manifest = load_run_manifest(args.benchmark_metadata['manifest_path'], cohort)
        test_indices = np.asarray(manifest['splits']['test'], dtype=np.int64)
        relative_paths = np.asarray(
            [cohort['records'][idx]['path'] for idx in test_indices], dtype=str
        )
        write_npz(
            output_dir / 'test_predictions.npz',
            cohort_indices=test_indices,
            relative_paths=relative_paths,
            predictions_years=predictions,
            targets_years=targets,
        )

        # Post-hoc RAPL diagnostics; these never affect training or selection.
        diag_loader = DataLoader(
            unlab.dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False,
            worker_init_fn=seed_dataloader_worker,
            generator=dataloader_generator(args.seed, 'unlabeled'),
        )
        pseudo_values, probe_values, trust_values, strong_values = [], [], [], []
        with torch.no_grad():
            model.eval(); frozen.eval(); probe.eval()
            for weak, strong in diag_loader:
                weak, strong = weak.to(device), strong.to(device)
                pseudo = model(weak)
                probe_prediction = probe(frozen.backbone(weak)).squeeze(-1)
                strong_prediction = model(strong)
                disagreement = (probe_prediction - pseudo).abs()
                pseudo_values.append(pseudo.cpu().numpy() * std + mean)
                probe_values.append(probe_prediction.cpu().numpy() * std + mean)
                trust_values.append((1.0 / (1.0 + disagreement)).cpu().numpy())
                strong_values.append(strong_prediction.cpu().numpy() * std + mean)
        unlabeled_indices = np.asarray(manifest['unlabeled_indices'], dtype=np.int64)
        write_npz(
            output_dir / 'analysis_snapshot.npz',
            cohort_indices=unlabeled_indices,
            relative_paths=np.asarray([cohort['records'][idx]['path'] for idx in unlabeled_indices], dtype=str),
            true_ages=np.asarray([cohort['records'][idx]['age'] for idx in unlabeled_indices], dtype=np.float32),
            pseudo_label=np.concatenate(pseudo_values),
            frozen_probe_prediction=np.concatenate(probe_values),
            disagreement=np.abs(np.concatenate(probe_values) - np.concatenate(pseudo_values)),
            trust_weight=np.concatenate(trust_values),
            strong_view_prediction=np.concatenate(strong_values),
        )
        elapsed = time.time() - started
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
        config = vars(args).copy()
        metadata = {
            **args.benchmark_metadata,
            'method': 'rapl',
            'status': 'complete',
            'gpu_visible_index': 0 if device.type == 'cuda' else None,
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'wall_clock_seconds': elapsed,
            'peak_allocated_cuda_bytes': peak_allocated,
            'peak_reserved_cuda_bytes': peak_reserved,
            'checkpoint_size_bytes': checkpoint_size(args.save),
            'checkpoint_reloaded': True,
            'test_evaluations': 1,
            'test_used_for_selection': False,
            'best_epoch': best_epoch,
            'validation_mae': best_val,
            'model': (model.weight_identifier if args.backbone in ('dinov2','dinov3_convnext_tiny') else 'ImageNet-pretrained ResNet-50'),
            'target_backbone': args.backbone,
            'probe_backbone': (f'frozen {frozen.weight_identifier}' if _pb in ('dinov2','dinov3_convnext_tiny') else 'frozen ImageNet-pretrained ResNet-50'),
            'pretrained_weight_identifier': (model.weight_identifier if args.backbone in ('dinov2','dinov3_convnext_tiny') else 'torchvision.ResNet50_Weights.IMAGENET1K_V1'),
            'pretrained_weight_url': (model.weight_url if args.backbone == 'dinov2' else (model.model_identifier if args.backbone == 'dinov3_convnext_tiny' else 'https://download.pytorch.org/models/resnet50-0676ba61.pth')),
            'pretrained_weight_checksum_sha256': (model.weight_checksum_sha256 if args.backbone == 'dinov3_convnext_tiny' else None),
            'native_views': 'weak+strong',
        }
        write_json(output_dir / 'config.json', config)
        write_json(output_dir / 'metadata.json', metadata)
        write_json(output_dir / 'metrics.json', {
            'best_epoch': best_epoch,
            'validation_mae': best_val,
            'test_mae': mae,
            'test_r2': r2,
        })
        if sampler_stop is not None:
            sampler_stop.set()
            sampler_thread.join(timeout=5)

    if args.save and not is_image_benchmark:
        import os; os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'frozen_backbone': frozen.backbone.state_dict(),
            'probe': probe.state_dict(),
            'mean': mean, 'std': std,
        }, args.save)
        print(f'Saved to {args.save}')


def train_probe_text(args, loaders, scaler, device):
    lab, unlab, val, test = loaders
    mean, std = scaler
    feat_dim = next(iter(lab))[0].shape[1]

    feats, labels = [], []
    for x, y in lab:
        feats.append(x); labels.append(y)
    feats_all = torch.cat(feats)
    labels_all = torch.cat(labels)

    Z = torch.cat([feats_all, torch.ones(len(feats_all), 1)], dim=1)
    sol = torch.linalg.lstsq(Z, labels_all.unsqueeze(1)).solution
    probe = nn.Linear(feat_dim, 1)
    probe.weight.data.copy_(sol[:-1, 0].unsqueeze(0))
    probe.bias.data.copy_(sol[-1:, 0])
    probe.requires_grad_(False)
    probe.eval().to(device)

    pred = feats_all @ sol[:-1, 0] + sol[-1, 0]
    mae = (pred - labels_all).abs().mean()
    print(f'Probe trained: MAE={mae:.3f} (normalized)')

    model = TextRegressor(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    lit, uit = cycle(lab), cycle(unlab)
    best_val, best_state = float('inf'), None
    for ep in range(args.epochs):
        model.train()
        for _ in range(max(len(lab), len(unlab))):
            x_l, y_l = next(lit); x_u_w, x_u_s = next(uit)
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

            loss_sup = F.mse_loss(model(x_l), y_l)
            with torch.no_grad():
                pseudo = model(x_u_w).detach()
                probe_est = probe(x_u_w).squeeze(-1)
            disagreement = (probe_est - pseudo).abs()
            r = (1.0 / (1.0 + disagreement)).detach()
            pred_s = model(x_u_s)
            loss_u = (r * (pred_s - pseudo).pow(2)).mean()
            loss = loss_sup + args.lambda_u * loss_u
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in val:
                ps.append(model(x.to(device)).cpu() * std + mean)
                ys.append(y.cpu() * std + mean)
        p, y = torch.cat(ps), torch.cat(ys)
        val_mae = (p - y).abs().mean().item()
        r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
        print(f'ep {ep+1}: val_mae={val_mae:.3f}, r2={r2:.4f}')
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in test:
            ps.append(model(x.to(device)).cpu() * std + mean)
            ys.append(y.cpu() * std + mean)
    p, y = torch.cat(ps), torch.cat(ys)
    test_mae = (p - y).abs().mean().item()
    r2 = 1 - ((p-y)**2).sum().item() / (((y-y.mean())**2).sum().item() + 1e-12)
    print(f'TEST probe: mae={test_mae:.3f}, r2={r2:.4f} (best val epoch)')


# ── Main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', default='utkface', choices=['utkface', 'imdb_wiki', 'stsb', 'utkface_official'])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--method', default='probe', choices=['probe', 'supervised'])
    p.add_argument('--labeled_ratio', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--fc_lr', type=float, default=1e-3)
    p.add_argument('--lambda_u', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save', type=str, default=None, help='Path to save checkpoint')
    p.add_argument(
        '--benchmark_manifest', type=str, default=None,
        help=('UTKFace benchmark seed manifest. Defaults to the packaged ratio-0.05 '
              'manifest matching --seed.'),
    )
    p.add_argument('--benchmark_output_dir', type=str, default=None)
    p.add_argument(
        '--inspect_data', action='store_true',
        help='Load and inspect one batch from each data split, then exit without training.',
    )
    p.add_argument('--backbone', default='resnet50', choices=['resnet50', 'dinov2', 'dinov3_convnext_tiny'])
    p.add_argument('--probe_backbone', default=None, choices=['resnet50', 'dinov2', 'dinov3_convnext_tiny'])
    p.add_argument('--dino', default='s', choices=['s', 'b', 'l'])
    p.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    p.set_defaults(pretrained=True)

    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = make_data(args)
    scaler = loaders[-2:]
    loaders = loaders[:4]

    if args.inspect_data:
        names = ('labeled', 'unlabeled', 'validation', 'test')
        for name, loader in zip(names, loaders):
            batch = next(iter(loader))
            shapes = [tuple(value.shape) for value in batch]
            print(f'{name} batch shapes: {shapes}')
        if hasattr(args, 'benchmark_metadata'):
            print('Benchmark metadata: ' + json.dumps(args.benchmark_metadata, sort_keys=True))
        return

    if args.method == 'probe':
        train_probe_method(args, loaders, scaler, device)
    elif args.method == 'supervised':
        if args.dataset == 'stsb': train_supervised_text(args, loaders, scaler, device)
        else: train_supervised(args, loaders, scaler, device)


if __name__ == '__main__':
    main()
