import os
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader
from typing import cast
from datasets import load_dataset, load_from_disk, DatasetDict

_to_rgb = transforms.Lambda(lambda x: x.convert("RGB") if hasattr(x, "convert") else x)
_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Strong aug: fine-grained / large-class datasets
train_transform = transforms.Compose([
    _to_rgb,
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    _normalize,
])

# Mild aug: medical / structured datasets (PCAM, EMNIST, FashionMNIST...)
mild_train_transform = transforms.Compose([
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _normalize,
])

# No aug: text-rendered images (RenderedSST2)
minimal_train_transform = transforms.Compose([
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _normalize,
])

# Test transform (shared)
vit_transform = transforms.Compose([
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _normalize,
])

# Per-dataset train transform overrides
_DATASET_TRAIN_TRANSFORMS = {
    "RenderedSST2": minimal_train_transform,
    "PCAM":         mild_train_transform,
    "EMNIST":       mild_train_transform,
    "MNIST":        mild_train_transform,
    "FashionMNIST": mild_train_transform,
}

class HuggingFaceWrapper(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img_key = 'image' if 'image' in item else 'img'
        img = item[img_key]
        label = item['label']

        if self.transform:
            img = self.transform(img)

        return img, label

def _create_dataloader(dataset_name: str, is_train: bool, batch_size: int, num_workers: int) -> DataLoader:
    dataset_dir = './dataset/raw'
    os.makedirs(dataset_dir, exist_ok=True)

    t_flag = is_train
    s_flag = 'train' if is_train else 'test'
    pet_s_flag = 'trainval' if is_train else 'test'
    shuffle = is_train
    if is_train:
        tf = _DATASET_TRAIN_TRANSFORMS.get(dataset_name, train_transform)
    else:
        tf = vit_transform

    tv_configs = {
        "EuroSAT":      lambda: dset.EuroSAT(dataset_dir, transform=tf, download=True),
        "SVHN":         lambda: dset.SVHN(dataset_dir, split=s_flag, transform=tf, download=True),
        "GTSRB":        lambda: dset.GTSRB(dataset_dir, split=s_flag, transform=tf, download=True),
        "MNIST":        lambda: dset.MNIST(dataset_dir, train=t_flag, transform=tf, download=True),
        "DTD":          lambda: dset.DTD(dataset_dir, split=s_flag, transform=tf, download=True),
        "Flowers102":   lambda: dset.Flowers102(dataset_dir, split=s_flag, transform=tf, download=True),
        "PCAM":         lambda: dset.PCAM(dataset_dir, split=s_flag, transform=tf, download=True),
        "OxfordIIITPet":lambda: dset.OxfordIIITPet(dataset_dir, split=pet_s_flag, transform=tf, download=True),
        "STL10":        lambda: dset.STL10(dataset_dir, split=s_flag, transform=tf, download=True),
        "CIFAR100":     lambda: dset.CIFAR100(dataset_dir, train=t_flag, transform=tf, download=True),
        "CIFAR10":      lambda: dset.CIFAR10(dataset_dir, train=t_flag, transform=tf, download=True),
        "Food101":      lambda: dset.Food101(dataset_dir, split=s_flag, transform=tf, download=True),
        "FashionMNIST": lambda: dset.FashionMNIST(dataset_dir, train=t_flag, transform=tf, download=True),
        "EMNIST":       lambda: dset.EMNIST(dataset_dir, split='byclass', train=t_flag, transform=tf, download=True),
        "RenderedSST2": lambda: dset.RenderedSST2(dataset_dir, split=s_flag, transform=tf, download=True),
    }

    hf_online_configs = {
        "RESISC45": ("timm/resisc45", 'train' if is_train else 'validation'),
    }

    local_datasets = ['SUN397', 'Cars']

    def _make_loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    if dataset_name in tv_configs:
        try:
            return _make_loader(tv_configs[dataset_name]())
        except Exception as e:
            raise RuntimeError(f"{dataset_name} 로드 실패: {e}") from e

    if dataset_name in hf_online_configs:
        hf_path, split = hf_online_configs[dataset_name]
        hf_ds = load_dataset(hf_path, split=split, cache_dir=dataset_dir)
        return _make_loader(HuggingFaceWrapper(hf_ds, transform=tf))

    if dataset_name in local_datasets:
        path = os.path.join(dataset_dir, dataset_name)
        loaded_ds = cast(DatasetDict, load_from_disk(path))
        target_split = 'train' if is_train else 'test'
        if not is_train and target_split not in loaded_ds:
            target_split = 'validation' if 'validation' in loaded_ds else list(loaded_ds.keys())[-1]
        return _make_loader(HuggingFaceWrapper(loaded_ds[target_split], transform=tf))

    raise ValueError(f"알 수 없는 데이터셋 이름: '{dataset_name}'")

def get_train_dataloader(dataset_name: str, batch_size: int = 32, num_workers: int = 2) -> DataLoader:
    return _create_dataloader(dataset_name, is_train=True, batch_size=batch_size, num_workers=num_workers)

def get_test_dataloader(dataset_name: str, batch_size: int = 32, num_workers: int = 2) -> DataLoader:
    return _create_dataloader(dataset_name, is_train=False, batch_size=batch_size, num_workers=num_workers)
