"""Download base DistilBERT model and optionally fine-tune on synthetic NER data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from synthetic_data import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class WordLevelNERDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        label2id: dict[str, int],
        max_length: int,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        words: list[str] = sample["words"]
        tags: list[str] = sample["tags"]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                tag = tags[word_id] if word_id < len(tags) else "O"
                labels.append(self.label2id.get(tag, self.label2id["O"]))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def align_predictions_to_words(
    predictions: list[int],
    word_ids: list[int | None],
    id2label: dict[int, str],
) -> list[str]:
    word_tags: list[str] = []
    seen: set[int] = set()
    for pred, word_id in zip(predictions, word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        word_tags.append(id2label.get(pred, "O"))
    return word_tags


def build_compute_metrics(id2label: dict[int, str]):
    """Entity-token level precision/recall/F1 (ignores 'O' and padding)."""
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = fp = fn = correct = total = 0
        for p_row, l_row in zip(preds, labels):
            for p, l in zip(p_row, l_row):
                if l == -100:
                    continue
                total += 1
                if p == l:
                    correct += 1
                l_ent = id2label[int(l)] != "O"
                p_ent = id2label[int(p)] != "O"
                if l_ent and p == l:
                    tp += 1
                elif p_ent and p != l:
                    fp += 1
                if l_ent and p != l:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "accuracy": correct / total if total else 0.0,
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
        }

    return compute_metrics


def fine_tune(
    model,
    tokenizer,
    config: dict,
    output_dir: Path,
) -> None:
    mcfg = config["model"]
    labels: list[str] = config["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    samples = generate_synthetic_dataset(
        samples_per_type=mcfg["synthetic_samples_per_type"],
        document_types=config["document_types"],
    )

    eval_ratio = mcfg.get("eval_ratio", 0.08)
    split = int(len(samples) * (1.0 - eval_ratio))
    train_samples = samples[:split]
    eval_samples = samples[split:]

    train_ds = WordLevelNERDataset(train_samples, tokenizer, label2id, mcfg["max_length"])
    eval_ds = WordLevelNERDataset(eval_samples, tokenizer, label2id, mcfg["max_length"])

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=mcfg["synthetic_epochs"],
        per_device_train_batch_size=mcfg["batch_size"],
        per_device_eval_batch_size=mcfg["batch_size"],
        learning_rate=float(mcfg["learning_rate"]),
        warmup_ratio=mcfg.get("warmup_ratio", 0.06),
        weight_decay=mcfg.get("weight_decay", 0.01),
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(id2label),
    )

    print(f"Fine-tuning on {len(train_samples)} synthetic samples "
          f"(eval on {len(eval_samples)}) …")
    trainer.train()
    print("Final eval:", trainer.evaluate())
    print("Fine-tuning complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare NER model")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels: list[str] = config["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    base_model = config["model"]["base_model"]
    output_dir = ROOT / config["paths"]["pytorch_model_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer and model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if config["model"].get("fine_tune_synthetic", True) and not args.skip_finetune:
        fine_tune(model, tokenizer, config, output_dir)

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": base_model,
        "num_labels": len(labels),
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "max_length": config["model"]["max_length"],
        "tokenizer_cased": config["model"].get("tokenizer_cased", False),
        "fields_by_type": config["fields_by_type"],
        "field_display_de": config.get("field_display_de", {}),
    }
    with open(output_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
