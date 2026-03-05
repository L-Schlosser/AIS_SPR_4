"""Tests for NER dataset loading, tokenization, and dataloader creation."""

import json

import pytest
import torch
from transformers import DistilBertTokenizerFast

from edge_model.extraction.config import ExtractionConfig
from edge_model.extraction.dataset import NERDataset, create_ner_dataloaders, load_ner_data
from edge_model.extraction.labels import ARZTBESUCH_LABELS, get_label2id

# --- Fixtures ---


@pytest.fixture
def sample_ner_samples():
    """Minimal NER samples for testing."""
    return [
        {
            "tokens": ["Patient", ":", "Max", "Mustermann", "Datum", ":", "01.01.2024"],
            "ner_tags": ["O", "O", "B-PATIENT", "I-PATIENT", "O", "O", "B-DATE"],
            "document_type": "arztbesuchsbestaetigung",
        },
        {
            "tokens": ["Dr.", "Schmidt", "Praxis", "Muster"],
            "ner_tags": ["B-DOCTOR", "I-DOCTOR", "B-FACILITY", "I-FACILITY"],
            "document_type": "arztbesuchsbestaetigung",
        },
    ]


@pytest.fixture
def jsonl_file(sample_ner_samples, tmp_path):
    """Write sample NER data to a temporary JSONL file."""
    path = tmp_path / "test_ner.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for sample in sample_ner_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def tokenizer():
    """Load DistilBert tokenizer."""
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")


@pytest.fixture
def label2id():
    """Label to ID mapping for arztbesuch labels."""
    return get_label2id(ARZTBESUCH_LABELS)


# --- Test load_ner_data ---


class TestLoadNERData:
    """Tests for JSONL loading and validation."""

    def test_load_valid_jsonl(self, jsonl_file):
        samples = load_ner_data(jsonl_file)
        assert len(samples) == 2
        assert "tokens" in samples[0]
        assert "ner_tags" in samples[0]

    def test_load_preserves_content(self, jsonl_file):
        samples = load_ner_data(jsonl_file)
        assert samples[0]["tokens"][0] == "Patient"
        assert samples[0]["ner_tags"][2] == "B-PATIENT"

    def test_load_missing_tokens_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"ner_tags": ["O"]}) + "\n")
        with pytest.raises(ValueError, match="missing 'tokens' or 'ner_tags'"):
            load_ner_data(path)

    def test_load_missing_ner_tags_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"tokens": ["hello"]}) + "\n")
        with pytest.raises(ValueError, match="missing 'tokens' or 'ner_tags'"):
            load_ner_data(path)

    def test_load_length_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"tokens": ["a", "b"], "ner_tags": ["O"]}) + "\n")
        with pytest.raises(ValueError, match="length mismatch"):
            load_ner_data(path)

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        samples = load_ner_data(path)
        assert samples == []

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"tokens": ["hi"], "ner_tags": ["O"]}) + "\n")
            f.write("\n")
            f.write(json.dumps({"tokens": ["bye"], "ner_tags": ["O"]}) + "\n")
        samples = load_ner_data(path)
        assert len(samples) == 2


# --- Test NERDataset ---


class TestNERDataset:
    """Tests for NER dataset tokenization and subword alignment."""

    def test_dataset_length(self, sample_ner_samples, tokenizer, label2id):
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        assert len(ds) == 2

    def test_getitem_returns_expected_keys(self, sample_ner_samples, tokenizer, label2id):
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_shapes(self, sample_ner_samples, tokenizer, label2id):
        max_len = 64
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=max_len)
        item = ds[0]
        assert item["input_ids"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        assert item["labels"].shape == (max_len,)

    def test_special_tokens_get_minus_100(self, sample_ner_samples, tokenizer, label2id):
        """[CLS] and [SEP] should have label -100."""
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        item = ds[0]
        # First token is [CLS]
        assert item["labels"][0].item() == -100
        # Find [SEP] — last non-pad token
        attention = item["attention_mask"]
        seq_len = attention.sum().item()
        assert item["labels"][seq_len - 1].item() == -100

    def test_padding_tokens_get_minus_100(self, sample_ner_samples, tokenizer, label2id):
        """Padding positions should have label -100."""
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        item = ds[0]
        attention = item["attention_mask"]
        # All positions where attention_mask == 0 should have labels == -100
        pad_positions = (attention == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            for pos in pad_positions:
                assert item["labels"][pos].item() == -100

    def test_subword_alignment(self, tokenizer, label2id):
        """When a word is split into subwords, only first gets the label."""
        # Use a word likely to be split into subwords
        samples = [
            {
                "tokens": ["Arztbesuchsbestätigung"],
                "ner_tags": ["B-FACILITY"],
                "document_type": "arztbesuchsbestaetigung",
            }
        ]
        ds = NERDataset(samples, tokenizer, label2id, max_length=32)
        item = ds[0]
        labels = item["labels"]

        # Count non -100 labels (excluding special tokens and subwords)
        real_labels = [lbl.item() for lbl in labels if lbl.item() != -100]
        # Only the first subword should have a real label
        assert len(real_labels) == 1
        assert real_labels[0] == label2id["B-FACILITY"]

    def test_known_label_mapping(self, sample_ner_samples, tokenizer, label2id):
        """Verify specific tokens get expected label IDs."""
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        item = ds[0]
        labels = item["labels"]

        # Collect non-special labels
        real_labels = [lbl.item() for lbl in labels if lbl.item() != -100]
        # The tags are: O, O, B-PATIENT, I-PATIENT, O, O, B-DATE
        expected_ids = [
            label2id["O"],
            label2id["O"],
            label2id["B-PATIENT"],
            label2id["I-PATIENT"],
            label2id["O"],
            label2id["O"],
            label2id["B-DATE"],
        ]
        # real_labels may have more entries if some words are not split,
        # but first 7 should match (assuming no subword splitting for these simple tokens)
        assert real_labels[:7] == expected_ids

    def test_dtypes(self, sample_ner_samples, tokenizer, label2id):
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long


# --- Test batch collation ---


class TestBatchCollation:
    """Tests for DataLoader batching."""

    def test_batch_collation(self, sample_ner_samples, tokenizer, label2id):
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 64)
        assert batch["attention_mask"].shape == (2, 64)
        assert batch["labels"].shape == (2, 64)

    def test_single_item_batch(self, sample_ner_samples, tokenizer, label2id):
        ds = NERDataset(sample_ner_samples, tokenizer, label2id, max_length=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (1, 64)


# --- Test create_ner_dataloaders ---


class TestCreateNERDataloaders:
    """Tests for the dataloader factory function."""

    def test_create_dataloaders(self, tmp_path):
        """Test that create_ner_dataloaders returns train and val loaders."""
        # Create small train/val JSONL files
        samples = [
            {"tokens": ["Hallo", "Welt"], "ner_tags": ["O", "O"], "document_type": "arztbesuchsbestaetigung"}
            for _ in range(8)
        ]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        for path, data in [(train_path, samples), (val_path, samples[:2])]:
            with open(path, "w", encoding="utf-8") as f:
                for s in data:
                    f.write(json.dumps(s) + "\n")

        config = ExtractionConfig(batch_size=4, max_length=32)
        train_loader, val_loader = create_ner_dataloaders(
            train_path, val_path, config, ARZTBESUCH_LABELS
        )
        assert len(train_loader) == 2  # 8 samples / batch_size 4
        assert len(val_loader) == 1  # 2 samples / batch_size 4 (1 partial batch)

        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[0] == 4
        assert batch["input_ids"].shape[1] == 32
