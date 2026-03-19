"""Tests for extractor inference wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from edge_model.extraction.labels import ARZTBESUCH_LABELS, LIEFERSCHEIN_LABELS, REISEKOSTEN_LABELS


class TestExtractorInferenceTokenization:
    """Tests for tokenization and input shape preparation."""

    LABELS = ARZTBESUCH_LABELS
    MAX_LENGTH = 256

    def test_tokenizer_output_shapes(self):
        """Tokenizer produces input_ids and attention_mask of correct shape."""
        from transformers import DistilBertTokenizerFast

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")
        encoding = tokenizer(
            "Bestätigung Arztbesuch Patient Max Mustermann",
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LENGTH,
        )
        assert encoding["input_ids"].shape == (1, self.MAX_LENGTH)
        assert encoding["attention_mask"].shape == (1, self.MAX_LENGTH)

    def test_tokenizer_dtype(self):
        """Tokenizer output can be cast to int64 for ONNX."""
        from transformers import DistilBertTokenizerFast

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")
        encoding = tokenizer(
            "Test text",
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].astype(np.int64)
        assert input_ids.dtype == np.int64

    def test_offset_mapping_available(self):
        """Tokenizer returns offset_mapping when requested."""
        from transformers import DistilBertTokenizerFast

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")
        encoding = tokenizer(
            "Hello world",
            return_tensors="np",
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LENGTH,
        )
        assert "offset_mapping" in encoding
        assert encoding["offset_mapping"].shape[0] == 1
        assert encoding["offset_mapping"].shape[1] == self.MAX_LENGTH


class TestExtractorInference:
    """Tests for the ExtractorInference class using mocked ONNX session."""

    LABELS = ARZTBESUCH_LABELS

    def _make_mock_extractor(self, logits: np.ndarray | None = None):
        """Create an ExtractorInference with mocked ONNX session.

        Args:
            logits: Optional logits array (1, seq_len, num_labels).
                    If None, returns all-zeros logits (all O tags).
        """
        from edge_model.inference.extractor_inference import ExtractorInference

        if logits is None:
            logits = np.zeros((1, 256, len(self.LABELS)), dtype=np.float32)

        mock_session = MagicMock()
        mock_session.run.return_value = [logits]

        with patch("edge_model.inference.extractor_inference.ort.InferenceSession", return_value=mock_session):
            extractor = ExtractorInference("dummy.onnx", "distilbert-base-german-cased", self.LABELS)

        # Replace session with our mock (in case it was overwritten)
        extractor._session = mock_session
        return extractor

    def test_extract_returns_dict(self):
        """extract returns a dict."""
        extractor = self._make_mock_extractor()
        result = extractor.extract("Bestätigung Arztbesuch Patient Max Mustermann")
        assert isinstance(result, dict)

    def test_extract_all_o_tags_returns_empty(self):
        """When all predictions are O, extract returns empty dict."""
        logits = np.zeros((1, 256, len(self.LABELS)), dtype=np.float32)
        # Make O tag (index 0) have highest logit
        logits[:, :, 0] = 10.0
        extractor = self._make_mock_extractor(logits)

        result = extractor.extract("Bestätigung Arztbesuch Patient Max Mustermann")
        assert result == {}

    def test_extract_calls_session_with_correct_keys(self):
        """Session.run is called with input_ids and attention_mask."""
        extractor = self._make_mock_extractor()
        extractor.extract("Test text")

        call_args = extractor._session.run.call_args
        assert call_args[0][0] is None
        input_dict = call_args[0][1]
        assert "input_ids" in input_dict
        assert "attention_mask" in input_dict

    def test_extract_input_shapes(self):
        """Session receives inputs with shape (1, max_length)."""
        extractor = self._make_mock_extractor()
        extractor.extract("Some test text here")

        call_args = extractor._session.run.call_args
        input_dict = call_args[0][1]
        assert input_dict["input_ids"].shape == (1, 256)
        assert input_dict["attention_mask"].shape == (1, 256)

    def test_extract_input_dtypes(self):
        """Session receives int64 inputs."""
        extractor = self._make_mock_extractor()
        extractor.extract("Test")

        call_args = extractor._session.run.call_args
        input_dict = call_args[0][1]
        assert input_dict["input_ids"].dtype == np.int64
        assert input_dict["attention_mask"].dtype == np.int64

    def test_extract_with_known_tags(self):
        """When model predicts specific B- tags, extract returns those fields."""
        from transformers import DistilBertTokenizerFast

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-german-cased")
        text = "Max Mustermann"
        encoding = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )

        # Find which token positions correspond to real tokens (non-special, non-padding)
        offset_mapping = encoding["offset_mapping"][0]
        real_token_indices = []
        for idx, offset in enumerate(offset_mapping):
            if offset[0] != 0 or offset[1] != 0:
                real_token_indices.append(idx)

        # Create logits that predict B-PATIENT for first real token, I-PATIENT for rest
        logits = np.zeros((1, 256, len(self.LABELS)), dtype=np.float32)
        logits[:, :, 0] = 5.0  # Default to O
        b_patient_idx = self.LABELS.index("B-PATIENT")
        i_patient_idx = self.LABELS.index("I-PATIENT")

        if len(real_token_indices) >= 1:
            logits[0, real_token_indices[0], 0] = 0.0
            logits[0, real_token_indices[0], b_patient_idx] = 10.0
        for idx in real_token_indices[1:]:
            logits[0, idx, 0] = 0.0
            logits[0, idx, i_patient_idx] = 10.0

        extractor = self._make_mock_extractor(logits)
        result = extractor.extract(text)
        assert "patient" in result
        assert "Max" in result["patient"] or "Mustermann" in result["patient"]


class TestExtractAndPostprocess:
    """Tests for extract_and_postprocess method."""

    LABELS = ARZTBESUCH_LABELS

    def _make_mock_extractor(self):
        """Create an ExtractorInference with all-O predictions."""
        from edge_model.inference.extractor_inference import ExtractorInference

        logits = np.zeros((1, 256, len(self.LABELS)), dtype=np.float32)
        logits[:, :, 0] = 10.0  # All O tags

        mock_session = MagicMock()
        mock_session.run.return_value = [logits]

        with patch("edge_model.inference.extractor_inference.ort.InferenceSession", return_value=mock_session):
            extractor = ExtractorInference("dummy.onnx", "distilbert-base-german-cased", self.LABELS)
        extractor._session = mock_session
        return extractor

    def test_postprocess_returns_dict(self):
        """extract_and_postprocess returns a dict."""
        extractor = self._make_mock_extractor()
        result = extractor.extract_and_postprocess("Test text", "arztbesuchsbestaetigung")
        assert isinstance(result, dict)

    def test_postprocess_adds_document_type(self):
        """Postprocessed output includes document_type field."""
        extractor = self._make_mock_extractor()
        result = extractor.extract_and_postprocess("Test text", "arztbesuchsbestaetigung")
        assert result["document_type"] == "arztbesuchsbestaetigung"

    def test_postprocess_unknown_type_raises(self):
        """Unknown document_type raises ValueError."""
        extractor = self._make_mock_extractor()
        with pytest.raises(ValueError, match="Unknown document type"):
            extractor.extract_and_postprocess("Test", "unknown_type")

    def test_postprocess_all_document_types(self):
        """extract_and_postprocess works for all 3 document types."""
        for doc_type, labels in [
            ("arztbesuchsbestaetigung", ARZTBESUCH_LABELS),
            ("reisekostenbeleg", REISEKOSTEN_LABELS),
            ("lieferschein", LIEFERSCHEIN_LABELS),
        ]:
            from edge_model.inference.extractor_inference import ExtractorInference

            logits = np.zeros((1, 256, len(labels)), dtype=np.float32)
            logits[:, :, 0] = 10.0

            mock_session = MagicMock()
            mock_session.run.return_value = [logits]

            with patch("edge_model.inference.extractor_inference.ort.InferenceSession", return_value=mock_session):
                extractor = ExtractorInference("dummy.onnx", "distilbert-base-german-cased", labels)
            extractor._session = mock_session

            result = extractor.extract_and_postprocess("Test text", doc_type)
            assert result["document_type"] == doc_type

    def test_postprocess_reisekosten_default_currency(self):
        """Reisekosten postprocessor adds default EUR currency."""
        from edge_model.inference.extractor_inference import ExtractorInference

        logits = np.zeros((1, 256, len(REISEKOSTEN_LABELS)), dtype=np.float32)
        logits[:, :, 0] = 10.0

        mock_session = MagicMock()
        mock_session.run.return_value = [logits]

        with patch("edge_model.inference.extractor_inference.ort.InferenceSession", return_value=mock_session):
            extractor = ExtractorInference("dummy.onnx", "distilbert-base-german-cased", REISEKOSTEN_LABELS)
        extractor._session = mock_session

        result = extractor.extract_and_postprocess("Test text", "reisekostenbeleg")
        assert result["currency"] == "EUR"
