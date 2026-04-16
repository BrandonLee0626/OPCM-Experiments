import os
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader
from typing import cast
from datasets import load_from_disk, DatasetDict

DATASET_DIR = '/data/image_classification'

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
_vit_normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
_clip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
_to_rgb  = transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x)
_BICUBIC = transforms.InterpolationMode.BICUBIC

# ---------------------------------------------------------------------------
# ViT transforms
# ---------------------------------------------------------------------------
_vit_train = transforms.Compose([              # strong aug: most datasets
    _to_rgb,
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    _vit_normalize,
])
_vit_mild_train = transforms.Compose([         # mild aug: medical / structured
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _vit_normalize,
])
_vit_minimal_train = transforms.Compose([      # no aug: text-rendered images
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _vit_normalize,
])
_vit_test = transforms.Compose([
    _to_rgb,
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _vit_normalize,
])

# ---------------------------------------------------------------------------
# CLIP transforms  (BICUBIC interpolation throughout)
# ---------------------------------------------------------------------------
_clip_train = transforms.Compose([             # strong aug: most datasets
    _to_rgb,
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=_BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _clip_normalize,
])
_clip_mild_train = transforms.Compose([        # mild aug: medical / structured
    _to_rgb,
    transforms.Resize(256, interpolation=_BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _clip_normalize,
])
_clip_minimal_train = transforms.Compose([     # no aug: text-rendered images
    _to_rgb,
    transforms.Resize(256, interpolation=_BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _clip_normalize,
])
_clip_test = transforms.Compose([
    _to_rgb,
    transforms.Resize(224, interpolation=_BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _clip_normalize,
])

# ---------------------------------------------------------------------------
# Per-dataset train-transform overrides (applied for both ViT and CLIP)
# ---------------------------------------------------------------------------
_VIT_TRAIN_OVERRIDES = {
    'RenderedSST2': _vit_minimal_train,
    'PCAM':         _vit_mild_train,
    'EMNIST':       _vit_mild_train,
    'MNIST':        _vit_mild_train,
    'FashionMNIST': _vit_mild_train,
}
_CLIP_TRAIN_OVERRIDES = {
    'RenderedSST2': _clip_minimal_train,
    'PCAM':         _clip_mild_train,
    'EMNIST':       _clip_mild_train,
    'MNIST':        _clip_mild_train,
    'FashionMNIST': _clip_mild_train,
}


# ---------------------------------------------------------------------------
# HuggingFace dataset wrapper
# ---------------------------------------------------------------------------
class HuggingFaceWrapper(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds    = hf_ds
        self.transform = transform
        # Detect image key once at init rather than on every __getitem__
        features = getattr(hf_ds, 'features', {})
        self._img_key = 'image' if 'image' in features else 'img'

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item  = self.hf_ds[idx]
        img   = item[self._img_key]
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Core dataloader factory
# ---------------------------------------------------------------------------
def _create_dataloader(dataset_name: str, is_train: bool, batch_size: int,
                       num_workers: int, model_type: str = 'vit') -> DataLoader:
    s_flag     = 'train' if is_train else 'test'
    pet_s_flag = 'trainval' if is_train else 'test'

    if model_type == 'clip':
        tf = _CLIP_TRAIN_OVERRIDES.get(dataset_name, _clip_train) if is_train else _clip_test
    elif is_train:
        tf = _VIT_TRAIN_OVERRIDES.get(dataset_name, _vit_train)
    else:
        tf = _vit_test

    def _make_loader(ds: Dataset) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0))

    def _load_imagefolder(folder: str, classes_file: str | None = None) -> dset.ImageFolder:
        """Load ImageFolder, remapping targets to original class order if classes_file exists."""
        ds = dset.ImageFolder(folder, transform=tf)
        if classes_file is None or not os.path.exists(classes_file):
            return ds
        with open(classes_file) as f:
            ordered_names = [line.rstrip('\n') for line in f]
        # alpha_idx (ImageFolder default) → original_idx (HF label order)
        remap = {ds.class_to_idx[n]: i for i, n in enumerate(ordered_names) if n in ds.class_to_idx}
        if all(remap.get(i, i) == i for i in range(len(ds.classes))):
            return ds  # ordering already matches — no remap needed
        ds.targets = [remap.get(t, t) for t in ds.targets]
        ds.samples = [(p, remap.get(lbl, lbl)) for p, lbl in ds.samples]
        ds.class_to_idx = {n: i for i, n in enumerate(ordered_names) if n in ds.class_to_idx}
        ds.classes = ordered_names
        return ds

    def _imagefolder_or(name: str, split: str, fallback) -> Dataset:
        """Return ImageFolder (with class-order correction) if converted, else call fallback()."""
        root   = os.path.join(DATASET_DIR, f'{name}_imagefolder')
        folder = os.path.join(root, split)
        if os.path.isdir(folder):
            return _load_imagefolder(folder, classes_file=os.path.join(root, 'classes.txt'))
        return fallback()

    tv_configs = {
        # --- already stored as image files: no conversion needed ---
        'EuroSAT':       lambda: dset.EuroSAT(DATASET_DIR, transform=tf, download=True),
        'GTSRB':         lambda: dset.GTSRB(DATASET_DIR, split=s_flag, transform=tf, download=True),
        'DTD':           lambda: dset.DTD(DATASET_DIR, split=s_flag, transform=tf, download=True),
        'Flowers102':    lambda: dset.Flowers102(DATASET_DIR, split=s_flag, transform=tf, download=True),
        'OxfordIIITPet': lambda: dset.OxfordIIITPet(DATASET_DIR, split=pet_s_flag, transform=tf, download=True),
        'Food101':       lambda: dset.Food101(DATASET_DIR, split=s_flag, transform=tf, download=True),
        'RenderedSST2':  lambda: dset.RenderedSST2(DATASET_DIR, split=s_flag, transform=tf, download=True),
        # --- stored as ImageFolder after one-time conversion ---
        # RESISC45 uses 'validation' for test (mirrors original HF split naming)
        'RESISC45':      lambda: _load_imagefolder(
            os.path.join(DATASET_DIR, 'RESISC45', 'train' if is_train else 'validation'),
            classes_file=os.path.join(DATASET_DIR, 'RESISC45', 'classes.txt')),
        # --- binary/HDF5 formats: ImageFolder if converted, else torchvision fallback ---
        # SVHN:  scipy .mat load + 32×32 heavy upscaling
        'SVHN':          lambda: _imagefolder_or('SVHN', s_flag,
            lambda: dset.SVHN(DATASET_DIR, split=s_flag, transform=tf, download=True)),
        # PCAM:  h5py can't be safely forked → DataLoader hangs with num_workers > 0
        'PCAM':          lambda: _imagefolder_or('PCAM', s_flag,
            lambda: dset.PCAM(DATASET_DIR, split=s_flag, transform=tf, download=True)),
        # STL10: full binary array loaded into RAM, 96×96 images
        'STL10':         lambda: _imagefolder_or('STL10', s_flag,
            lambda: dset.STL10(DATASET_DIR, split=s_flag, transform=tf, download=True)),
        # --- small in-memory datasets: fast enough without conversion ---
        'MNIST':         lambda: dset.MNIST(DATASET_DIR, train=is_train, transform=tf, download=True),
        'FashionMNIST':  lambda: dset.FashionMNIST(DATASET_DIR, train=is_train, transform=tf, download=True),
        'EMNIST':        lambda: dset.EMNIST(DATASET_DIR, split='byclass', train=is_train, transform=tf, download=True),
        'CIFAR10':       lambda: dset.CIFAR10(DATASET_DIR, train=is_train, transform=tf, download=True),
        'CIFAR100':      lambda: dset.CIFAR100(DATASET_DIR, train=is_train, transform=tf, download=True),
    }

    if dataset_name in tv_configs:
        try:
            return _make_loader(tv_configs[dataset_name]())
        except Exception as e:
            raise RuntimeError(f'Failed to load {dataset_name}: {e}') from e

    # SUN397 / Cars: use ImageFolder if converted, else fall back to HuggingFace Arrow.
    # classes.txt ensures label ordering matches the original dataset.
    if dataset_name in ('SUN397', 'Cars'):
        imagefolder_root  = os.path.join(DATASET_DIR, f'{dataset_name}_imagefolder')
        split             = 'train' if is_train else 'test'
        imagefolder_split = os.path.join(imagefolder_root, split)
        if os.path.isdir(imagefolder_root):
            if not is_train and not os.path.isdir(imagefolder_split):
                alt = os.path.join(imagefolder_root, 'validation')
                imagefolder_split = alt if os.path.isdir(alt) else imagefolder_split
            return _make_loader(_load_imagefolder(
                imagefolder_split,
                classes_file=os.path.join(imagefolder_root, 'classes.txt'),
            ))
        path      = os.path.join(DATASET_DIR, dataset_name)
        loaded_ds = cast(DatasetDict, load_from_disk(path))
        if not is_train and split not in loaded_ds:
            split = 'validation' if 'validation' in loaded_ds else list(loaded_ds.keys())[-1]
        return _make_loader(HuggingFaceWrapper(loaded_ds[split], transform=tf))

    raise ValueError(f"Unknown dataset name: '{dataset_name}'")


def get_train_dataloader(dataset_name: str, batch_size: int = 32,
                         num_workers: int = 2, model_type: str = 'vit') -> DataLoader:
    return _create_dataloader(dataset_name, is_train=True, batch_size=batch_size,
                              num_workers=num_workers, model_type=model_type)

def get_test_dataloader(dataset_name: str, batch_size: int = 32,
                        num_workers: int = 2, model_type: str = 'vit') -> DataLoader:
    return _create_dataloader(dataset_name, is_train=False, batch_size=batch_size,
                              num_workers=num_workers, model_type=model_type)
