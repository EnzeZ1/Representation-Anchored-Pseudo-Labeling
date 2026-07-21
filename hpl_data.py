"""
Load UTKFace matching official HPL processing exactly.
Reimplements their data loading to avoid import conflicts.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RandAugmentMC(object):
    """Simplified RandAugment matching official HPL (n=3, m=4)."""
    def __init__(self, n=3, m=4):
        self.n = n
        self.m = m
        self.augment = transforms.RandAugment(num_ops=n, magnitude=m)
    def __call__(self, img):
        return self.augment(img)


class TransformFixMatch(object):
    """Official HPL weak/strong augmentation pair."""
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=200, padding=8, padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=200, padding=8, padding_mode='reflect'),
            RandAugmentMC(n=3, m=4)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixMatch224(object):
    """ImageNet-compatible weak/strong augmentation pair (224×224)."""
    def __init__(self):
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 10),
            transforms.ToTensor(), norm])

    def __call__(self, x):
        return self.weak(x), self.strong(x)


class UTKFace(Dataset):
    def __init__(self, csv_path, img_dir, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        df = pd.read_csv(csv_path)
        df['SPLIT'] = df['SPLIT'].str.lower()
        df = df.query(f'SPLIT == "{split}"')
        self.data = df['FileName'].tolist()
        self.targets = df['age'].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        with Image.open(os.path.join(self.img_dir, "UTKFace", self.data[index])) as img:
            img = img.convert('RGB')
        label = np.asarray(self.targets[index]).astype('float32')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class UTKFace_SSL(UTKFace):
    def __init__(self, csv_path, img_dir, split='train', transform=None,
                 index_list=None, is_labeled=False, ssl_mult=1):
        super().__init__(csv_path, img_dir, split, transform)
        if index_list is not None:
            self.data = [self.data[i] for i in index_list]
            self.targets = [self.targets[i] for i in index_list]
        if is_labeled:
            self.data = self.data * ssl_mult
            self.targets = self.targets * ssl_mult


class UnlabeledDataset(Dataset):
    """Wraps UTKFace_SSL with TransformFixMatch to return (weak, strong) only."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        (weak, strong), _ = self.dataset[i]
        return weak, strong


def get_mean_and_std(dataset, batch_size=8, num_workers=4):
    """Compute pixel mean/std from dataset — matches official utils."""
    dataloader = DataLoader(dataset, batch_size=batch_size,
                           num_workers=num_workers, shuffle=True)
    n, s1, s2 = 0, 0., 0.
    for batch in tqdm.tqdm(dataloader):
        x = batch[0]
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(s2 / n - mean ** 2).astype(np.float32)
    return mean, std


def make_data_hpl_official(args):
    """Load UTKFace with official HPL splits and processing."""
    # Find the HPL data directory
    hpl_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'Heteroscedastic-Pseudo-Labels-main', 'utkface', 'data')

    csv_path = os.path.join(hpl_data_dir, 'utkface.csv')
    img_dir = hpl_data_dir

    # Compute image normalization from train set (matching official)
    print('Computing image mean/std from train set...')
    raw_train = UTKFace(csv_path, img_dir, split='train',
                        transform=transforms.ToTensor())
    img_mean, img_std = get_mean_and_std(raw_train)
    print(f'img mean std {img_mean} {img_std}')

    # Label normalization from FULL train set
    train_all = UTKFace(csv_path, img_dir, split='train', transform=None)
    print(f'total train data: {len(train_all)}')
    targets = np.array(train_all.targets, dtype=np.float32)
    y_mean = float(targets.mean())
    y_std = float(targets.std())
    print(f'Mean of targets: {y_mean:.2f}')
    print(f'Std of targets: {y_std:.2f}')

    # Split labeled/unlabeled (matching official: fixed seed, fixed sizes)
    indices = np.arange(len(train_all))
    np.random.seed(0)
    np.random.shuffle(indices)

    fixed_sizes = {0.05: 500, 0.10: 1000, 0.20: 2000}
    if args.labeled_ratio in fixed_sizes:
        num_labeled = fixed_sizes[args.labeled_ratio]
    else:
        num_labeled = max(1, int(len(train_all) * args.labeled_ratio))

    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    # Use ImageNet-compatible 224 preprocessing (matches frozen backbone's pretraining)
    img_size = 224
    norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])

    # Datasets
    lab_ds = UTKFace_SSL(csv_path, img_dir, split='train',
                         index_list=labeled_idx, transform=transform_labeled,
                         is_labeled=True, ssl_mult=2)
    print(f'labeled data after duplicates: {len(lab_ds)}')

    unlab_raw = UTKFace_SSL(csv_path, img_dir, split='train',
                           index_list=unlabeled_idx,
                           transform=TransformFixMatch224(),
                           is_labeled=False)
    unlab_ds = UnlabeledDataset(unlab_raw)
    print(f'Using SSL_SPLIT unlabeled {len(unlab_ds)}')

    val_ds = UTKFace(csv_path, img_dir, split='val', transform=transform_val)
    print(f'val data {len(val_ds)}')

    test_ds = UTKFace(csv_path, img_dir, split='test', transform=transform_val)
    print(f'test data {len(test_ds)}')

    # Normalize labels
    for ds in [lab_ds, val_ds, test_ds]:
        ds.targets = [(t - y_mean) / y_std for t in ds.targets]

    def loader(ds, shuffle=True, drop=True):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                         num_workers=args.workers, pin_memory=True, drop_last=drop)

    return (loader(lab_ds), loader(unlab_ds),
            loader(val_ds, False, False), loader(test_ds, False, False),
            y_mean, y_std)