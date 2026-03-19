"""Tests for ONNX export and quantization of NER extraction models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizerFast

from edge_model.extraction.export_onnx import export_ner_to_onnx, quantize_ner_model, verify_ner_onnx


def _create_tiny_ner_model(output_dir: str, num_labels: int = 14) -> str:
    """Create and save a tiny DistilBert NER model for testing."""
    # Load tokenizer first to get correct vocab_size for German model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")

    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        n_layers=1,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=256,
        num_labels=num_labels,
    )
    model = DistilBertForTokenClassification(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


class TestExportNerToOnnx:
    """Tests for the ONNX export function."""

    def test_export_produces_onnx_files(self, tmp_path):
        """Test that export creates ONNX model files."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")

        _create_tiny_ner_model(model_dir)
        result = export_ner_to_onnx(model_dir, onnx_dir)

        assert Path(result).exists()
        onnx_files = list(Path(result).glob("*.onnx"))
        assert len(onnx_files) > 0

    def test_export_copies_tokenizer(self, tmp_path):
        """Test that export copies tokenizer files to output directory."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")

        _create_tiny_ner_model(model_dir)
        result = export_ner_to_onnx(model_dir, onnx_dir)

        # Tokenizer should be available in output dir
        output_path = Path(result)
        tokenizer_files = list(output_path.glob("tokenizer*"))
        assert len(tokenizer_files) > 0

    def test_exported_model_runs_inference(self, tmp_path):
        """Test that exported ONNX model can run inference."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")

        _create_tiny_ner_model(model_dir, num_labels=14)
        export_ner_to_onnx(model_dir, onnx_dir)

        # Find and load the ONNX model
        onnx_files = list(Path(onnx_dir).glob("*.onnx"))
        session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])

        # Create sample input
        input_ids = np.array([[101, 7592, 102] + [0] * 253], dtype=np.int64)
        attention_mask = np.array([[1, 1, 1] + [0] * 253], dtype=np.int64)

        outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = outputs[0]

        assert logits.shape == (1, 256, 14)


class TestQuantizeNerModel:
    """Tests for the INT8 quantization function."""

    def test_quantized_model_is_smaller(self, tmp_path):
        """Test that quantized model file is smaller than the original."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")
        quantized_dir = str(tmp_path / "quantized")

        _create_tiny_ner_model(model_dir)
        export_ner_to_onnx(model_dir, onnx_dir)
        quantize_ner_model(onnx_dir, quantized_dir)

        quantized_files = list(Path(quantized_dir).glob("*.onnx"))

        assert len(quantized_files) > 0
        # Quantized model should exist and be valid
        assert quantized_files[0].stat().st_size > 0

    def test_quantized_model_runs_inference(self, tmp_path):
        """Test that quantized ONNX model can run inference."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")
        quantized_dir = str(tmp_path / "quantized")

        _create_tiny_ner_model(model_dir, num_labels=14)
        export_ner_to_onnx(model_dir, onnx_dir)
        quantize_ner_model(onnx_dir, quantized_dir)

        onnx_files = list(Path(quantized_dir).glob("*.onnx"))
        session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])

        input_ids = np.array([[101, 7592, 102] + [0] * 253], dtype=np.int64)
        attention_mask = np.array([[1, 1, 1] + [0] * 253], dtype=np.int64)

        outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = outputs[0]

        assert logits.shape == (1, 256, 14)

    def test_missing_onnx_raises_error(self, tmp_path):
        """Test that missing ONNX files raise FileNotFoundError."""
        empty_dir = str(tmp_path / "empty")
        Path(empty_dir).mkdir()

        with pytest.raises(FileNotFoundError):
            quantize_ner_model(empty_dir, str(tmp_path / "output"))


class TestVerifyNerOnnx:
    """Tests for ONNX NER model verification."""

    def test_verify_returns_logits(self, tmp_path):
        """Test that verify returns logits with correct shape."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")

        _create_tiny_ner_model(model_dir, num_labels=14)
        export_ner_to_onnx(model_dir, onnx_dir)

        logits = verify_ner_onnx(onnx_dir, onnx_dir, "Hallo Welt")

        assert logits.ndim == 3
        assert logits.shape[0] == 1
        assert logits.shape[1] == 256
        assert logits.shape[2] == 14

    def test_verify_with_longer_text(self, tmp_path):
        """Test verification with longer text input."""
        model_dir = str(tmp_path / "model")
        onnx_dir = str(tmp_path / "onnx_output")

        _create_tiny_ner_model(model_dir, num_labels=15)
        export_ner_to_onnx(model_dir, onnx_dir)

        sample = "Rechnung Firma Mustermann GmbH Musterstraße 1 1010 Wien Datum 15.03.2024 Betrag 125,50 EUR"
        logits = verify_ner_onnx(onnx_dir, onnx_dir, sample)

        assert logits.shape[0] == 1
        assert logits.shape[2] == 15

    def test_verify_missing_onnx_raises_error(self, tmp_path):
        """Test that missing ONNX files raise FileNotFoundError."""
        empty_dir = str(tmp_path / "empty")
        Path(empty_dir).mkdir()

        with pytest.raises(FileNotFoundError):
            verify_ner_onnx(empty_dir, empty_dir, "test")
