"""NER dataset and dataloader creation for token classification."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast

from edge_model.extraction.config import ExtractionConfig


def load_ner_data(jsonl_path: str | Path) -> list[dict]:
    """Load and validate NER samples from a JSONL file.

    Each line must contain: {"tokens": [...], "ner_tags": [...], "document_type": "..."}.
    """
    samples = []
    jsonl_path = Path(jsonl_path)
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "tokens" not in sample or "ner_tags" not in sample:
                msg = f"Line {line_num}: missing 'tokens' or 'ner_tags'"
                raise ValueError(msg)
            if len(sample["tokens"]) != len(sample["ner_tags"]):
                n_tok, n_tag = len(sample["tokens"]), len(sample["ner_tags"])
                msg = f"Line {line_num}: tokens ({n_tok}) and ner_tags ({n_tag}) length mismatch"
                raise ValueError(msg)
            samples.append(sample)
    return samples


class NERDataset(Dataset):
    """Token classification dataset with subword alignment.

    When a word is split into subwords by the tokenizer, only the first
    subword receives the original BIO tag. Subsequent subwords and special
    tokens ([CLS], [SEP], [PAD]) receive label -100 (ignored in loss).
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: DistilBertTokenizerFast,
        label2id: dict[str, int],
        max_length: int = 256,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Tokenize with word-level tracking
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word: use original tag
                if word_id < len(ner_tags):
                    tag = ner_tags[word_id]
                    aligned_labels.append(self.label2id.get(tag, self.label2id.get("O", 0)))
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent subword of same word: ignore in loss
                aligned_labels.append(-100)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def create_ner_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    config: ExtractionConfig,
    labels: list[str],
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for NER.

    Args:
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        config: Extraction configuration.
        labels: List of BIO label strings.

    Returns:
        Tuple of (train_dataloader, val_dataloader).
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)
    label2id = {label: idx for idx, label in enumerate(labels)}

    train_samples = load_ner_data(train_path)
    val_samples = load_ner_data(val_path)

    train_dataset = NERDataset(train_samples, tokenizer, label2id, config.max_length)
    val_dataset = NERDataset(val_samples, tokenizer, label2id, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader
