"""Classification dataset and data loading utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from edge_model.classification.config import ClassificationConfig


class DocumentDataset(Dataset):
    """Dataset that loads document images from subdirectories, using directory name as label."""

    def __init__(self, root_dir: str | Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []

        # Discover class directories (sorted for deterministic ordering)
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]

        for label_idx, class_dir in enumerate(class_dirs):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label


class _SampleListDataset(Dataset):
    """Lightweight dataset wrapping a list of (path, label) with a transform."""

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_transforms(image_size: int, is_training: bool = True) -> transforms.Compose:
    """Return torchvision transforms for training or evaluation."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def create_dataloaders(
    root_dir: str | Path,
    config: ClassificationConfig,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with stratified split."""
    full_dataset = DocumentDataset(root_dir)

    labels = np.array([label for _, label in full_dataset.samples])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_indices, val_indices = next(splitter.split(np.zeros(len(labels)), labels))

    train_transform = get_transforms(config.image_size, is_training=True)
    val_transform = get_transforms(config.image_size, is_training=False)

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_dataset = _SampleListDataset(train_samples, train_transform)
    val_dataset = _SampleListDataset(val_samples, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
