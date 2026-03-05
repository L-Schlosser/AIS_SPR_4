"""Validation script for ONNX classification models."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from edge_model.classification.config import ClassificationConfig


def _preprocess_image(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    """Load and preprocess an image for ONNX inference."""
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    return img_array[np.newaxis, ...]


def compute_metrics(
    predictions: list[int],
    labels: list[int],
    class_names: list[str],
) -> dict:
    """Compute classification metrics from predictions and labels.

    Args:
        predictions: List of predicted class indices.
        labels: List of ground truth class indices.
        class_names: List of class name strings.

    Returns:
        Dict with overall accuracy, per-class precision/recall/F1, and confusion matrix.
    """
    num_classes = len(class_names)
    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)

    # Overall accuracy
    accuracy = float(np.mean(preds_arr == labels_arr)) if len(labels_arr) > 0 else 0.0

    # Confusion matrix (rows = true, cols = predicted)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels_arr, preds_arr):
        confusion_matrix[true_label, pred_label] += 1

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(confusion_matrix[i, i])
        fp = int(confusion_matrix[:, i].sum() - tp)
        fn = int(confusion_matrix[i, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(confusion_matrix[i, :].sum()),
        }

    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def validate_onnx_model(
    model_path: str,
    data_dir: str,
    config: ClassificationConfig,
) -> dict:
    """Validate an ONNX classification model on a dataset.

    Args:
        model_path: Path to the ONNX model file.
        data_dir: Path to directory with class subdirectories containing images.
        config: Classification configuration.

    Returns:
        Dict with all computed metrics.
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    data_path = Path(data_dir)
    predictions: list[int] = []
    labels: list[int] = []

    # Build class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}

    # Load and classify images
    for class_name in config.class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            input_data = _preprocess_image(img_path, config.image_size)
            output = session.run(None, {"image": input_data})[0]
            predicted_idx = int(np.argmax(output, axis=1)[0])
            predictions.append(predicted_idx)
            labels.append(class_to_idx[class_name])

    # Compute metrics
    metrics = compute_metrics(predictions, labels, config.class_names)

    # Print formatted table
    print(f"\nValidation Results ({len(labels)} images)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"\n{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*70}")
    for name, cls_metrics in metrics["per_class"].items():
        print(
            f"{name:<30} {cls_metrics['precision']:>10.4f} {cls_metrics['recall']:>10.4f} "
            f"{cls_metrics['f1']:>10.4f} {cls_metrics['support']:>10d}"
        )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    header = f"{'':>30}"
    for name in config.class_names:
        header += f" {name[:12]:>12}"
    print(header)
    for i, name in enumerate(config.class_names):
        row = f"{name:>30}"
        for j in range(len(config.class_names)):
            row += f" {metrics['confusion_matrix'][i][j]:>12d}"
        print(row)

    # Save results to JSON
    output_path = Path(model_path).parent / "validation_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ONNX classification model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to validation data directory")
    args = parser.parse_args()

    cfg = ClassificationConfig()
    validate_onnx_model(args.model_path, args.data_dir, cfg)
