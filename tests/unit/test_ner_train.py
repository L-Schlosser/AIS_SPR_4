"""Tests for NER training script."""

import json
from pathlib import Path

import numpy as np
import pytest

from edge_model.extraction.config import ExtractionConfig
from edge_model.extraction.labels import ARZTBESUCH_LABELS, LABEL_SETS
from edge_model.extraction.train import compute_metrics_fn, compute_per_entity_f1


@pytest.fixture()
def tiny_config() -> ExtractionConfig:
    """Minimal config for fast testing."""
    return ExtractionConfig(
        model_name="distilbert-base-german-cased",
        max_length=64,
        batch_size=2,
        lr=5e-5,
        epochs=1,
        weight_decay=0.01,
    )


@pytest.fixture()
def tiny_ner_data(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal NER JSONL files for training and validation."""
    train_samples = [
        {"tokens": ["Patient", ":", "Max", "Mustermann", "Datum", ":", "01.01.2024"],
         "ner_tags": ["O", "O", "B-PATIENT", "I-PATIENT", "O", "O", "B-DATE"],
         "document_type": "arztbesuchsbestaetigung"},
        {"tokens": ["Dr.", "Schmidt", "Praxis", "Am", "Markt"],
         "ner_tags": ["B-DOCTOR", "I-DOCTOR", "B-FACILITY", "I-FACILITY", "I-FACILITY"],
         "document_type": "arztbesuchsbestaetigung"},
        {"tokens": ["Bestätigung", "Arztbesuch", "Patient", "Anna", "Berger"],
         "ner_tags": ["O", "O", "O", "B-PATIENT", "I-PATIENT"],
         "document_type": "arztbesuchsbestaetigung"},
        {"tokens": ["Dauer", ":", "30", "Minuten", "Zeit", ":", "14:30"],
         "ner_tags": ["O", "O", "B-DURATION", "O", "O", "O", "B-TIME"],
         "document_type": "arztbesuchsbestaetigung"},
    ]

    val_samples = [
        {"tokens": ["Patient", ":", "Lisa", "Müller", "Datum", ":", "15.03.2024"],
         "ner_tags": ["O", "O", "B-PATIENT", "I-PATIENT", "O", "O", "B-DATE"],
         "document_type": "arztbesuchsbestaetigung"},
        {"tokens": ["Dr.", "Weber", "Klinik", "Süd"],
         "ner_tags": ["B-DOCTOR", "I-DOCTOR", "B-FACILITY", "I-FACILITY"],
         "document_type": "arztbesuchsbestaetigung"},
    ]

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return train_path, val_path


class TestModelCreation:
    """Tests for NER model creation with correct num_labels."""

    def test_model_has_correct_num_labels(self) -> None:
        from transformers import DistilBertForTokenClassification

        labels = ARZTBESUCH_LABELS
        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-german-cased",
            num_labels=len(labels),
        )
        assert model.config.num_labels == len(labels)

    def test_model_output_shape(self) -> None:
        import torch
        from transformers import DistilBertForTokenClassification

        labels = ARZTBESUCH_LABELS
        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-german-cased",
            num_labels=len(labels),
        )
        model.eval()

        dummy_input = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)

        with torch.no_grad():
            output = model(input_ids=dummy_input, attention_mask=attention_mask)

        assert output.logits.shape == (1, 32, len(labels))

    def test_all_document_types_have_labels(self) -> None:
        for doc_type in ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]:
            assert doc_type in LABEL_SETS
            assert len(LABEL_SETS[doc_type]) > 1


class TestComputeMetrics:
    """Tests for compute_metrics function with known inputs."""

    def test_perfect_predictions(self) -> None:
        labels = ["O", "B-PATIENT", "I-PATIENT", "B-DATE"]
        compute_metrics = compute_metrics_fn(labels)

        # Perfect predictions: all non-O tags match
        logits = np.zeros((2, 5, 4))
        label_ids = np.array([
            [-100, 1, 2, 3, -100],
            [-100, 1, 0, 0, -100],
        ])

        # Set logits so argmax gives correct predictions
        for i in range(2):
            for j in range(5):
                if label_ids[i, j] != -100:
                    logits[i, j, label_ids[i, j]] = 10.0

        result = compute_metrics((logits, label_ids))
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["accuracy"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        labels = ["O", "B-PATIENT", "I-PATIENT", "B-DATE"]
        compute_metrics = compute_metrics_fn(labels)

        # All wrong: predict O for everything
        logits = np.zeros((1, 4, 4))
        logits[:, :, 0] = 10.0  # Always predict O
        label_ids = np.array([[1, 2, 3, 1]])  # All entity tags

        result = compute_metrics((logits, label_ids))
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_ignores_minus_100_labels(self) -> None:
        labels = ["O", "B-PATIENT"]
        compute_metrics = compute_metrics_fn(labels)

        logits = np.zeros((1, 5, 2))
        logits[0, 2, 1] = 10.0  # Predict B-PATIENT at position 2
        label_ids = np.array([[-100, -100, 1, -100, -100]])

        result = compute_metrics((logits, label_ids))
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_returns_expected_keys(self) -> None:
        labels = ["O", "B-PATIENT"]
        compute_metrics = compute_metrics_fn(labels)

        logits = np.zeros((1, 3, 2))
        label_ids = np.array([[0, 0, 0]])

        result = compute_metrics((logits, label_ids))
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "accuracy" in result

    def test_mixed_predictions(self) -> None:
        labels = ["O", "B-PATIENT", "I-PATIENT", "B-DATE"]
        compute_metrics = compute_metrics_fn(labels)

        # 2 correct entity predictions out of 3
        logits = np.zeros((1, 5, 4))
        label_ids = np.array([[-100, 1, 2, 3, -100]])

        # Correctly predict B-PATIENT and I-PATIENT, miss B-DATE
        logits[0, 1, 1] = 10.0  # Correct: B-PATIENT
        logits[0, 2, 2] = 10.0  # Correct: I-PATIENT
        logits[0, 3, 0] = 10.0  # Wrong: predict O instead of B-DATE

        result = compute_metrics((logits, label_ids))
        assert 0.0 < result["f1"] < 1.0
        assert result["recall"] < 1.0


class TestPerEntityF1:
    """Tests for per-entity-type F1 computation."""

    def test_per_entity_metrics(self) -> None:
        labels = ["O", "B-PATIENT", "I-PATIENT", "B-DATE"]
        predictions = np.array([[1, 2, 3]])  # B-PATIENT, I-PATIENT, B-DATE
        label_ids = np.array([[1, 2, 0]])  # B-PATIENT correct, I-PATIENT correct, B-DATE wrong (true=O)

        result = compute_per_entity_f1(predictions, label_ids, labels)
        assert "B-PATIENT" in result
        assert "I-PATIENT" in result
        assert result["B-PATIENT"]["f1"] == 1.0
        assert result["I-PATIENT"]["f1"] == 1.0

    def test_empty_predictions(self) -> None:
        labels = ["O", "B-PATIENT"]
        predictions = np.array([[0, 0, 0]])  # All O
        label_ids = np.array([[0, 0, 0]])  # All O

        result = compute_per_entity_f1(predictions, label_ids, labels)
        assert len(result) == 0  # No entity predictions


class TestTrainingRun:
    """Tests for training running on tiny data without errors."""

    def test_training_runs_1_epoch(self, tiny_config: ExtractionConfig, tiny_ner_data: tuple[Path, Path]) -> None:
        from edge_model.extraction.train import train_ner_model

        train_path, val_path = tiny_ner_data
        output_dir = train_path.parent / "output"

        metrics = train_ner_model(
            config=tiny_config,
            document_type="arztbesuchsbestaetigung",
            train_path=str(train_path),
            val_path=str(val_path),
            output_dir=str(output_dir),
        )

        assert "train_loss" in metrics
        assert "eval_metrics" in metrics
        assert "per_entity_f1" in metrics
        assert metrics["document_type"] == "arztbesuchsbestaetigung"

        # Check model files were saved
        assert (output_dir / "config.json").exists()
        assert (output_dir / "training_metrics.json").exists()

    def test_invalid_document_type_raises(self, tiny_config: ExtractionConfig) -> None:
        with pytest.raises(ValueError, match="Unknown document type"):
            from edge_model.extraction.train import train_ner_model

            train_ner_model(tiny_config, "invalid_type", "train.jsonl", "val.jsonl", "output")
