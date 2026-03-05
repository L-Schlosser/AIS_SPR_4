"""Classification model training with two-phase transfer learning."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from edge_model.classification.config import ClassificationConfig
from edge_model.classification.dataset import create_dataloaders


def create_model(config: ClassificationConfig) -> nn.Module:
    """Load pretrained EfficientNet-Lite0 and replace classifier head for 3 classes."""
    import timm

    model = timm.create_model(config.model_name, pretrained=True, num_classes=config.num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Validate for one epoch. Returns (avg_loss, accuracy, per_class_metrics)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision/recall
    per_class: dict[str, dict[str, float]] = {}
    num_classes = max(max(all_labels, default=0), max(all_preds, default=0)) + 1
    for c in range(num_classes):
        tp = sum(1 for p, lbl in zip(all_preds, all_labels) if p == c and lbl == c)
        fp = sum(1 for p, lbl in zip(all_preds, all_labels) if p == c and lbl != c)
        fn = sum(1 for p, lbl in zip(all_preds, all_labels) if p != c and lbl == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_class[str(c)] = {"precision": precision, "recall": recall}

    return avg_loss, accuracy, per_class


def train(config: ClassificationConfig, data_dir: str, output_dir: str) -> dict:
    """Full two-phase training: frozen backbone then full fine-tuning."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_dir, config)

    # Create model
    model = create_model(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    metrics_history: list[dict] = []

    # Phase 1: Train with frozen backbone
    print("\n--- Phase 1: Frozen backbone ---")
    freeze_backbone(model)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr_frozen)

    for epoch in range(config.epochs_frozen):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{config.epochs_frozen} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        metrics_history.append({
            "phase": "frozen", "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")

    # Phase 2: Unfreeze all and fine-tune
    print("\n--- Phase 2: Full fine-tuning ---")
    unfreeze_all(model)
    optimizer = Adam(model.parameters(), lr=config.lr_unfrozen)

    for epoch in range(config.epochs_unfrozen):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{config.epochs_unfrozen} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        metrics_history.append({
            "phase": "unfrozen", "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / "best_model.pt")

    # Save final metrics
    final_metrics = {
        "best_val_accuracy": best_val_acc,
        "per_class_metrics": per_class,
        "history": metrics_history,
        "config": {
            "model_name": config.model_name,
            "image_size": config.image_size,
            "num_classes": config.num_classes,
            "epochs_frozen": config.epochs_frozen,
            "epochs_unfrozen": config.epochs_unfrozen,
        },
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {output_path / 'best_model.pt'}")
    print(f"Metrics saved to {output_path / 'metrics.json'}")

    for class_id, m in per_class.items():
        print(f"  Class {class_id}: precision={m['precision']:.4f}, recall={m['recall']:.4f}")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()

    cfg = ClassificationConfig()
    train(cfg, args.data_dir, args.output_dir)
