"""Tests for classification dataset and data loading."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from edge_model.classification.config import ClassificationConfig
from edge_model.classification.dataset import (
    DocumentDataset,
    create_dataloaders,
    get_transforms,
)


@pytest.fixture()
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a mock dataset directory with small images in subdirectories."""
    class_names = ["arztbesuchsbestaetigung", "lieferschein", "reisekostenbeleg"]
    images_per_class = 6

    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(images_per_class):
            img = Image.new("RGB", (100, 140), color=(i * 40, 100, 200))
            img.save(class_dir / f"img_{i:03d}.png")

    return tmp_path


class TestDocumentDataset:
    """Tests for DocumentDataset class."""

    def test_loads_images_from_subdirectories(self, mock_dataset_dir: Path) -> None:
        dataset = DocumentDataset(mock_dataset_dir)
        assert len(dataset) == 18  # 3 classes * 6 images

    def test_class_names_sorted(self, mock_dataset_dir: Path) -> None:
        dataset = DocumentDataset(mock_dataset_dir)
        assert dataset.class_names == ["arztbesuchsbestaetigung", "lieferschein", "reisekostenbeleg"]

    def test_label_mapping_correct(self, mock_dataset_dir: Path) -> None:
        dataset = DocumentDataset(mock_dataset_dir)
        # First 6 samples should have label 0 (arztbesuchsbestaetigung)
        for i in range(6):
            _, label = dataset[i]
            assert label == 0
        # Next 6 should have label 1 (lieferschein)
        for i in range(6, 12):
            _, label = dataset[i]
            assert label == 1
        # Last 6 should have label 2 (reisekostenbeleg)
        for i in range(12, 18):
            _, label = dataset[i]
            assert label == 2

    def test_getitem_returns_tensor_and_int(self, mock_dataset_dir: Path) -> None:
        dataset = DocumentDataset(mock_dataset_dir)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_getitem_with_transform(self, mock_dataset_dir: Path) -> None:
        transform = get_transforms(224, is_training=False)
        dataset = DocumentDataset(mock_dataset_dir, transform=transform)
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        class_dir = tmp_path / "test_class"
        class_dir.mkdir()
        # Create a JSON file (should be ignored)
        (class_dir / "label.json").write_text("{}")
        # Create a PNG file (should be included)
        Image.new("RGB", (50, 50)).save(class_dir / "img.png")
        dataset = DocumentDataset(tmp_path)
        assert len(dataset) == 1

    def test_empty_directory(self, tmp_path: Path) -> None:
        dataset = DocumentDataset(tmp_path)
        assert len(dataset) == 0
        assert dataset.class_names == []


class TestGetTransforms:
    """Tests for get_transforms function."""

    def test_training_transforms_output_shape(self) -> None:
        transform = get_transforms(224, is_training=True)
        img = Image.new("RGB", (800, 1130))
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_eval_transforms_output_shape(self) -> None:
        transform = get_transforms(224, is_training=False)
        img = Image.new("RGB", (800, 1130))
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_training_transforms_normalized(self) -> None:
        transform = get_transforms(224, is_training=True)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = transform(img)
        # Normalized values should be roughly in [-2.5, 2.5] range
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_eval_transforms_deterministic(self) -> None:
        transform = get_transforms(224, is_training=False)
        img = Image.new("RGB", (800, 1130), color=(100, 150, 200))
        result1 = transform(img)
        result2 = transform(img)
        assert torch.allclose(result1, result2)

    def test_custom_image_size(self) -> None:
        transform = get_transforms(128, is_training=False)
        img = Image.new("RGB", (800, 1130))
        result = transform(img)
        assert result.shape == (3, 128, 128)


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_creates_train_and_val_loaders(self, mock_dataset_dir: Path) -> None:
        config = ClassificationConfig(batch_size=4)
        train_loader, val_loader = create_dataloaders(mock_dataset_dir, config, val_split=0.2)
        assert train_loader is not None
        assert val_loader is not None

    def test_split_sizes(self, mock_dataset_dir: Path) -> None:
        config = ClassificationConfig(batch_size=4)
        train_loader, val_loader = create_dataloaders(mock_dataset_dir, config, val_split=0.2)
        train_count = sum(len(batch[1]) for batch in train_loader)
        val_count = sum(len(batch[1]) for batch in val_loader)
        total = train_count + val_count
        assert total == 18
        # Approximately 80/20 split
        assert train_count > val_count

    def test_batch_shape(self, mock_dataset_dir: Path) -> None:
        config = ClassificationConfig(batch_size=4, image_size=224)
        train_loader, _ = create_dataloaders(mock_dataset_dir, config)
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == images.shape[0]

    def test_stratified_split_maintains_proportions(self, mock_dataset_dir: Path) -> None:
        config = ClassificationConfig(batch_size=4)
        train_loader, val_loader = create_dataloaders(mock_dataset_dir, config, val_split=0.2)

        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.tolist())

        val_labels = []
        for _, labels in val_loader:
            val_labels.extend(labels.tolist())

        # Each class should appear in both train and val
        train_classes = set(train_labels)
        val_classes = set(val_labels)
        assert len(train_classes) == 3
        assert len(val_classes) == 3

        # Train class distribution should be roughly equal
        train_counts = np.bincount(train_labels)
        assert train_counts.min() >= 3  # At least 3 per class in train (out of ~5 per class)
