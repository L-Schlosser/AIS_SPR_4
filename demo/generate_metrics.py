"""Generate evaluation metrics for the demo dashboard.

Runs classifier validation on sample images and NER evaluation on validation JSONL files.
Saves results to demo/metrics/ as JSON files.

Usage:
    uv run python demo/generate_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast

from edge_model.classification.config import ClassificationConfig
from edge_model.classification.validate import _preprocess_image
from edge_model.extraction.labels import LABEL_SETS, get_id2label

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_MODEL = PROJECT_ROOT / "edge_model/classification/models/classifier_int8.onnx"
DATA_DIR = PROJECT_ROOT / "data/samples"
METRICS_DIR = Path(__file__).resolve().parent / "metrics"

DOC_TYPES = ["arztbesuchsbestaetigung", "lieferschein", "reisekostenbeleg"]

NER_MODEL_PATHS = {
    "arztbesuchsbestaetigung": PROJECT_ROOT / "edge_model/extraction/models/arztbesuch/onnx/quantized/model.onnx",
    "reisekostenbeleg": PROJECT_ROOT / "edge_model/extraction/models/reisekosten/onnx/quantized/model.onnx",
    "lieferschein": PROJECT_ROOT / "edge_model/extraction/models/lieferschein/onnx/quantized/model.onnx",
}

NER_TOKENIZER_PATHS = {
    "arztbesuchsbestaetigung": PROJECT_ROOT / "edge_model/extraction/models/arztbesuch/onnx/quantized",
    "reisekostenbeleg": PROJECT_ROOT / "edge_model/extraction/models/reisekosten/onnx/quantized",
    "lieferschein": PROJECT_ROOT / "edge_model/extraction/models/lieferschein/onnx/quantized",
}


def generate_classifier_metrics() -> dict:
    """Run classifier on sample images and compute metrics."""
    if not CLASSIFIER_MODEL.exists():
        print(f"Classifier model not found: {CLASSIFIER_MODEL}")
        return {}

    config = ClassificationConfig()
    session = ort.InferenceSession(str(CLASSIFIER_MODEL), providers=["CPUExecutionProvider"])

    all_preds = []
    all_labels = []
    class_to_idx = {name: idx for idx, name in enumerate(config.class_names)}

    for class_name in config.class_names:
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            print(f"  Skipping {class_name}: directory not found")
            continue
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        print(f"  {class_name}: {len(images)} images")
        for img_path in images:
            input_data = _preprocess_image(img_path, config.image_size)
            output = session.run(None, {"image": input_data})[0]
            pred_idx = int(np.argmax(output, axis=1)[0])
            all_preds.append(pred_idx)
            all_labels.append(class_to_idx[class_name])

    if not all_labels:
        print("  No images found for classification evaluation")
        return {}

    # Use sklearn for metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=config.class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    per_class = {}
    for name in config.class_names:
        per_class[name] = {
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        }

    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "total_images": len(all_labels),
    }

    print(f"  Accuracy: {metrics['accuracy']:.4f} ({len(all_labels)} images)")
    return metrics


def _run_ner_inference(
    session: ort.InferenceSession,
    tokenizer: DistilBertTokenizerFast,
    tokens: list[str],
    ner_tags: list[str],
    label_list: list[str],
    max_length: int = 256,
) -> tuple[list[str], list[str]]:
    """Run NER inference on a single sample and return (predicted_tags, true_tags)."""
    # Use is_split_into_words=True to match training tokenization
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoding["input_ids"].astype(np.int64)
    attention_mask = encoding["attention_mask"].astype(np.int64)
    logits = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    tag_ids = np.argmax(logits, axis=-1)[0]
    id2label = get_id2label(label_list)

    word_ids = encoding.word_ids(batch_index=0)

    # Build per-word predictions (first subword token wins)
    word_preds: dict[int, str] = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_preds:
            word_preds[word_id] = id2label.get(int(tag_ids[idx]), "O")

    # Collect aligned predictions vs ground truth
    pred_tags = []
    true_tags = []
    for i, tag in enumerate(ner_tags):
        true_tags.append(tag)
        pred_tags.append(word_preds.get(i, "O"))

    return pred_tags, true_tags


def generate_ner_metrics() -> dict:
    """Run NER evaluation on validation JSONL files for each document type."""
    all_ner_metrics = {}

    for doc_type in DOC_TYPES:
        model_path = NER_MODEL_PATHS[doc_type]
        tokenizer_path = NER_TOKENIZER_PATHS[doc_type]
        val_path = DATA_DIR / f"{doc_type}_ner_val.jsonl"

        if not model_path.exists():
            print(f"  Skipping {doc_type}: model not found at {model_path}")
            continue
        if not val_path.exists():
            print(f"  Skipping {doc_type}: validation data not found at {val_path}")
            continue

        label_list = LABEL_SETS[doc_type]
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(tokenizer_path))

        # Load validation samples
        samples = []
        with open(val_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        print(f"  {doc_type}: {len(samples)} val samples")

        # Collect all predictions and labels
        all_pred_tags: list[str] = []
        all_true_tags: list[str] = []

        for sample in samples:
            pred_tags, true_tags = _run_ner_inference(
                session, tokenizer, sample["tokens"], sample["ner_tags"], label_list
            )
            all_pred_tags.extend(pred_tags)
            all_true_tags.extend(true_tags)

        # Compute per-entity metrics (exclude O tags)
        entity_tp: dict[str, int] = {}
        entity_fp: dict[str, int] = {}
        entity_fn: dict[str, int] = {}

        for pred_tag, true_tag in zip(all_pred_tags, all_true_tags):
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

        per_entity = {}
        all_entities = set(entity_tp) | set(entity_fp) | set(entity_fn)
        for entity in sorted(all_entities):
            tp = entity_tp.get(entity, 0)
            fp = entity_fp.get(entity, 0)
            fn = entity_fn.get(entity, 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_entity[entity] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1_val, 4),
            }

        # Overall micro-averaged F1
        total_tp = sum(entity_tp.values())
        total_fp = sum(entity_fp.values())
        total_fn = sum(entity_fn.values())
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        # Overall token accuracy (including O)
        correct = sum(1 for p, t in zip(all_pred_tags, all_true_tags) if p == t)
        total = len(all_pred_tags)
        accuracy = correct / total if total > 0 else 0.0

        all_ner_metrics[doc_type] = {
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "accuracy": round(accuracy, 4),
            "per_entity": per_entity,
            "num_samples": len(samples),
        }

        print(f"    F1: {micro_f1:.4f}, Accuracy: {accuracy:.4f}")

    return all_ner_metrics


def main() -> None:
    """Generate all metrics and save to demo/metrics/."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating classifier metrics...")
    classifier_metrics = generate_classifier_metrics()
    if classifier_metrics:
        out_path = METRICS_DIR / "classifier_metrics.json"
        with open(out_path, "w") as f:
            json.dump(classifier_metrics, f, indent=2)
        print(f"  Saved to {out_path}")
    else:
        print("  Skipped — no classifier model or data available")

    print("\nGenerating NER metrics...")
    ner_metrics = generate_ner_metrics()
    if ner_metrics:
        out_path = METRICS_DIR / "ner_metrics.json"
        with open(out_path, "w") as f:
            json.dump(ner_metrics, f, indent=2)
        print(f"  Saved to {out_path}")
    else:
        print("  Skipped — no NER models or data available")

    # Summary
    if classifier_metrics or ner_metrics:
        print("\nDone! Metrics saved to demo/metrics/")
    else:
        print("\nNo metrics generated. Ensure models are trained and data exists.")
        sys.exit(1)


if __name__ == "__main__":
    main()
