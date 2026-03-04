"""NER token classification training using HuggingFace Trainer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from edge_model.extraction.config import ExtractionConfig
from edge_model.extraction.dataset import NERDataset, load_ner_data
from edge_model.extraction.labels import LABEL_SETS, get_id2label, get_label2id


def compute_metrics_fn(label_list: list[str]):
    """Return a compute_metrics function for the HuggingFace Trainer.

    Uses seqeval-style evaluation: ignores -100 and "O" tags when
    computing entity-level precision, recall, and F1.
    """

    def compute_metrics(eval_preds) -> dict[str, float]:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Collect per-entity-type TP/FP/FN counts
        entity_tp: dict[str, int] = {}
        entity_fp: dict[str, int] = {}
        entity_fn: dict[str, int] = {}

        for pred_seq, label_seq in zip(predictions, labels):
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue

                pred_tag = label_list[pred_id] if 0 <= pred_id < len(label_list) else "O"
                true_tag = label_list[label_id] if 0 <= label_id < len(label_list) else "O"

                # Skip O tags for entity-level metrics
                if true_tag == "O" and pred_tag == "O":
                    continue

                if true_tag != "O":
                    entity_tp.setdefault(true_tag, 0)
                    entity_fp.setdefault(true_tag, 0)
                    entity_fn.setdefault(true_tag, 0)
                    if pred_tag == true_tag:
                        entity_tp[true_tag] += 1
                    else:
                        entity_fn[true_tag] += 1

                if pred_tag != "O" and pred_tag != true_tag:
                    entity_fp.setdefault(pred_tag, 0)
                    entity_fp[pred_tag] += 1

        # Compute micro-averaged metrics
        total_tp = sum(entity_tp.values())
        total_fp = sum(entity_fp.values())
        total_fn = sum(entity_fn.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Also compute overall accuracy (including O tags)
        correct = 0
        total = 0
        for pred_seq, label_seq in zip(predictions, labels):
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                total += 1
                if pred_id == label_id:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    return compute_metrics


def compute_per_entity_f1(
    predictions: np.ndarray, labels: np.ndarray, label_list: list[str]
) -> dict[str, dict[str, float]]:
    """Compute per-entity-type precision, recall, and F1."""
    entity_tp: dict[str, int] = {}
    entity_fp: dict[str, int] = {}
    entity_fn: dict[str, int] = {}

    for pred_seq, label_seq in zip(predictions, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue

            pred_tag = label_list[pred_id] if 0 <= pred_id < len(label_list) else "O"
            true_tag = label_list[label_id] if 0 <= label_id < len(label_list) else "O"

            if true_tag == "O" and pred_tag == "O":
                continue

            if true_tag != "O":
                entity_tp.setdefault(true_tag, 0)
                entity_fp.setdefault(true_tag, 0)
                entity_fn.setdefault(true_tag, 0)
                if pred_tag == true_tag:
                    entity_tp[true_tag] += 1
                else:
                    entity_fn[true_tag] += 1

            if pred_tag != "O" and pred_tag != true_tag:
                entity_fp.setdefault(pred_tag, 0)
                entity_fp[pred_tag] += 1

    result: dict[str, dict[str, float]] = {}
    all_entities = set(entity_tp) | set(entity_fp) | set(entity_fn)
    for entity in sorted(all_entities):
        tp = entity_tp.get(entity, 0)
        fp = entity_fp.get(entity, 0)
        fn = entity_fn.get(entity, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[entity] = {"precision": prec, "recall": rec, "f1": f1_val}

    return result


def train_ner_model(
    config: ExtractionConfig,
    document_type: str,
    train_path: str,
    val_path: str,
    output_dir: str,
) -> dict:
    """Train a NER model for a specific document type.

    Args:
        config: Extraction configuration.
        document_type: One of the supported document types.
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        output_dir: Directory to save trained model and tokenizer.

    Returns:
        Dict with training metrics.
    """
    if document_type not in LABEL_SETS:
        msg = f"Unknown document type: {document_type}. Must be one of {list(LABEL_SETS.keys())}"
        raise ValueError(msg)

    labels = LABEL_SETS[document_type]
    label2id = get_label2id(labels)
    id2label = get_id2label(labels)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Training NER model for: {document_type}")
    print(f"Labels ({len(labels)}): {labels}")

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
    model = DistilBertForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    # Load datasets
    train_samples = load_ner_data(train_path)
    val_samples = load_ner_data(val_path)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_dataset = NERDataset(train_samples, tokenizer, label2id, config.max_length)
    val_dataset = NERDataset(val_samples, tokenizer, label2id, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn(labels),
    )

    # Train
    train_result = trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()

    # Save best model and tokenizer
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Compute per-entity metrics
    eval_preds = trainer.predict(val_dataset)
    predictions = np.argmax(eval_preds.predictions, axis=-1)
    per_entity = compute_per_entity_f1(predictions, eval_preds.label_ids, labels)

    # Print per-entity F1 scores
    print("\nPer-entity F1 scores:")
    for entity, metrics in per_entity.items():
        print(f"  {entity}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

    # Save metrics
    metrics = {
        "document_type": document_type,
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_result,
        "per_entity_f1": per_entity,
        "config": {
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "epochs": config.epochs,
            "num_labels": len(labels),
        },
    }
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {output_path}")
    print(f"Eval F1: {eval_result.get('eval_f1', 0):.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER extraction model")
    parser.add_argument("--document-type", type=str, required=True, help="Document type to train for")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save trained model")
    args = parser.parse_args()

    cfg = ExtractionConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ner_model(cfg, args.document_type, args.train_path, args.val_path, args.output_dir)
