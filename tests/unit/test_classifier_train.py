"""Tests for classification training script."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from edge_model.classification.config import ClassificationConfig
from edge_model.classification.train import (
    create_model,
    freeze_backbone,
    train_epoch,
    unfreeze_all,
    validate_epoch,
)


@pytest.fixture()
def tiny_config() -> ClassificationConfig:
    """Config for testing with minimal resources."""
    return ClassificationConfig(
        image_size=224,
        num_classes=3,
        batch_size=2,
        lr_frozen=1e-4,
        lr_unfrozen=1e-5,
        epochs_frozen=1,
        epochs_unfrozen=1,
    )


@pytest.fixture()
def tiny_dataset_dir(tmp_path: Path) -> Path:
    """Create a tiny dataset with 2 images per class for fast testing."""
    class_names = ["arztbesuchsbestaetigung", "lieferschein", "reisekostenbeleg"]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(i * 80, 100, 200))
            img.save(class_dir / f"img_{i:03d}.png")
    return tmp_path


class TestCreateModel:
    """Tests for create_model function."""

    def test_output_shape(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 3)

    def test_output_num_classes(self) -> None:
        config = ClassificationConfig(num_classes=5)
        model = create_model(config)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (1, 5)

    def test_model_is_nn_module(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)
        assert isinstance(model, torch.nn.Module)


class TestFreezeBackbone:
    """Tests for freeze/unfreeze functions."""

    def test_frozen_backbone_params(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)
        freeze_backbone(model)

        for name, param in model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad is True, f"{name} should be trainable"
            else:
                assert param.requires_grad is False, f"{name} should be frozen"

    def test_frozen_has_some_trainable(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)
        freeze_backbone(model)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0, "At least classifier params should be trainable"

    def test_unfreeze_all_params(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)
        freeze_backbone(model)
        unfreeze_all(model)

        for param in model.parameters():
            assert param.requires_grad is True

    def test_frozen_has_fewer_trainable_than_unfrozen(self, tiny_config: ClassificationConfig) -> None:
        model = create_model(tiny_config)

        total_params = sum(1 for p in model.parameters())

        freeze_backbone(model)
        frozen_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        unfreeze_all(model)
        unfrozen_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        assert frozen_trainable < unfrozen_trainable
        assert unfrozen_trainable == total_params


class TestTrainEpoch:
    """Tests for train_epoch and validate_epoch functions."""

    def test_train_epoch_runs(self, tiny_config: ClassificationConfig, tiny_dataset_dir: Path) -> None:
        from edge_model.classification.dataset import create_dataloaders

        model = create_model(tiny_config)
        device = torch.device("cpu")
        model = model.to(device)

        train_loader, _ = create_dataloaders(tiny_dataset_dir, tiny_config, val_split=0.5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0.0 <= acc <= 1.0

    def test_validate_epoch_runs(self, tiny_config: ClassificationConfig, tiny_dataset_dir: Path) -> None:
        from edge_model.classification.dataset import create_dataloaders

        model = create_model(tiny_config)
        device = torch.device("cpu")
        model = model.to(device)

        _, val_loader = create_dataloaders(tiny_dataset_dir, tiny_config, val_split=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        loss, acc, per_class = validate_epoch(model, val_loader, criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(per_class, dict)
        assert loss >= 0
        assert 0.0 <= acc <= 1.0

    def test_validate_per_class_metrics(self, tiny_config: ClassificationConfig, tiny_dataset_dir: Path) -> None:
        from edge_model.classification.dataset import create_dataloaders

        model = create_model(tiny_config)
        device = torch.device("cpu")
        model = model.to(device)

        _, val_loader = create_dataloaders(tiny_dataset_dir, tiny_config, val_split=0.5)
        criterion = torch.nn.CrossEntropyLoss()

        _, _, per_class = validate_epoch(model, val_loader, criterion, device)
        for class_id, metrics in per_class.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
