"""Tests for the document processing pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from api.models import DocumentType, ProcessingResult
from edge_model.inference.config import PipelineConfig, load_config

# ---------------------------------------------------------------------------
# PipelineConfig and load_config tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.classifier_model_path == ""
        assert cfg.extractor_model_paths == {}
        assert cfg.extractor_tokenizer_paths == {}
        assert cfg.schemas_dir == "data/schemas"
        assert cfg.confidence_threshold == 0.7
        assert cfg.use_ocr is True

    def test_custom_values(self):
        cfg = PipelineConfig(
            classifier_model_path="cls.onnx",
            extractor_model_paths={"a": "a.onnx"},
            schemas_dir="schemas",
            confidence_threshold=0.5,
            use_ocr=False,
        )
        assert cfg.classifier_model_path == "cls.onnx"
        assert cfg.extractor_model_paths == {"a": "a.onnx"}
        assert cfg.confidence_threshold == 0.5
        assert cfg.use_ocr is False


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_valid_yaml(self, tmp_path):
        cfg_data = {
            "classifier_model_path": "models/cls.onnx",
            "extractor_model_paths": {"arztbesuchsbestaetigung": "models/arzt.onnx"},
            "extractor_tokenizer_paths": {"arztbesuchsbestaetigung": "models/arzt/"},
            "schemas_dir": "schemas",
            "confidence_threshold": 0.8,
            "use_ocr": False,
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

        cfg = load_config(str(cfg_file))
        assert cfg.classifier_model_path == "models/cls.onnx"
        assert cfg.extractor_model_paths == {"arztbesuchsbestaetigung": "models/arzt.onnx"}
        assert cfg.confidence_threshold == 0.8
        assert cfg.use_ocr is False

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_non_mapping_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- just a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_config(str(cfg_file))

    def test_load_partial_yaml_uses_defaults(self, tmp_path):
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("classifier_model_path: only_this.onnx\n", encoding="utf-8")
        cfg = load_config(str(cfg_file))
        assert cfg.classifier_model_path == "only_this.onnx"
        assert cfg.schemas_dir == "data/schemas"
        assert cfg.confidence_threshold == 0.7
        assert cfg.use_ocr is True


# ---------------------------------------------------------------------------
# DocumentPipeline tests (all components mocked)
# ---------------------------------------------------------------------------

class TestPipelineInitialization:
    """Tests for pipeline component initialization."""

    @patch("ocr.engine.OCREngine")
    @patch("edge_model.inference.pipeline.ExtractorInference")
    @patch("edge_model.inference.pipeline.ClassifierInference")
    @patch("edge_model.inference.pipeline.SchemaValidator")
    def test_init_creates_all_components(self, mock_validator, mock_classifier, mock_extractor, mock_ocr):
        cfg = PipelineConfig(
            classifier_model_path="cls.onnx",
            extractor_model_paths={
                "arztbesuchsbestaetigung": "arzt.onnx",
                "reisekostenbeleg": "reise.onnx",
            },
            extractor_tokenizer_paths={
                "arztbesuchsbestaetigung": "arzt/",
                "reisekostenbeleg": "reise/",
            },
            use_ocr=True,
        )

        from edge_model.inference.pipeline import DocumentPipeline

        DocumentPipeline(cfg)

        mock_classifier.assert_called_once()
        assert mock_extractor.call_count == 2
        mock_validator.assert_called_once_with("data/schemas")
        mock_ocr.assert_called_once()

    @patch("edge_model.inference.pipeline.ExtractorInference")
    @patch("edge_model.inference.pipeline.ClassifierInference")
    @patch("edge_model.inference.pipeline.SchemaValidator")
    def test_init_no_ocr_when_disabled(self, mock_validator, mock_classifier, mock_extractor):
        cfg = PipelineConfig(
            classifier_model_path="cls.onnx",
            use_ocr=False,
        )

        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline(cfg)
        assert pipeline._ocr is None


class TestPipelineProcess:
    """Tests for the process() flow."""

    def _make_pipeline(self, doc_type="arztbesuchsbestaetigung", confidence=0.95, ocr_text="some text",
                       extracted_fields=None, valid=True):
        """Helper to create a pipeline with mocked internals."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = object.__new__(DocumentPipeline)
        pipeline._config = PipelineConfig(confidence_threshold=0.7, use_ocr=True)

        # Mock classifier
        pipeline._classifier = MagicMock()
        pipeline._classifier.predict.return_value = (doc_type, confidence)

        # Mock OCR
        pipeline._ocr = MagicMock()
        ocr_result = MagicMock()
        ocr_result.text = ocr_text
        pipeline._ocr.extract_text.return_value = ocr_result

        # Mock extractor
        if extracted_fields is None:
            extracted_fields = {"document_type": doc_type, "patient_name": "Max Mustermann"}
        mock_extractor = MagicMock()
        mock_extractor.extract_and_postprocess.return_value = extracted_fields
        pipeline._extractors = {doc_type: mock_extractor}

        # Mock validator
        pipeline._validator = MagicMock()
        if valid:
            pipeline._validator.validate.return_value = (True, [])
        else:
            pipeline._validator.validate.return_value = (False, ["some error"])

        return pipeline

    def test_full_flow_returns_processing_result(self):
        pipeline = self._make_pipeline()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert isinstance(result, ProcessingResult)
        assert result.document_type == DocumentType.arztbesuchsbestaetigung
        assert result.confidence == 0.95
        assert result.raw_text == "some text"
        assert result.fields["patient_name"] == "Max Mustermann"

    def test_low_confidence_returns_empty_fields(self):
        pipeline = self._make_pipeline(confidence=0.3)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert result.confidence == 0.3
        assert result.fields == {}
        assert result.raw_text is None
        # OCR should not have been called
        pipeline._ocr.extract_text.assert_not_called()

    def test_correct_extractor_chosen_for_doc_type(self):
        pipeline = self._make_pipeline(doc_type="reisekostenbeleg",
                                       extracted_fields={"document_type": "reisekostenbeleg", "vendor_name": "Hotel"})
        # Add a second extractor to verify selection
        other_extractor = MagicMock()
        pipeline._extractors["arztbesuchsbestaetigung"] = other_extractor

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert result.document_type == DocumentType.reisekostenbeleg
        assert result.fields["vendor_name"] == "Hotel"
        other_extractor.extract_and_postprocess.assert_not_called()

    def test_validation_errors_attached_to_fields(self):
        pipeline = self._make_pipeline(valid=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert "_validation_errors" in result.fields
        assert result.fields["_validation_errors"] == ["some error"]

    def test_no_ocr_skips_extraction(self):
        pipeline = self._make_pipeline()
        pipeline._ocr = None
        pipeline._config = PipelineConfig(confidence_threshold=0.7, use_ocr=False)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert result.fields == {}
        assert result.raw_text is None

    def test_empty_ocr_text_skips_extraction(self):
        pipeline = self._make_pipeline(ocr_text="")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.process(image)

        assert result.fields == {}

    def test_classifier_called_with_image(self):
        pipeline = self._make_pipeline()
        image = np.ones((50, 80, 3), dtype=np.uint8) * 128
        pipeline.process(image)

        pipeline._classifier.predict.assert_called_once()
        call_arg = pipeline._classifier.predict.call_args[0][0]
        assert call_arg.shape == (50, 80, 3)


class TestPipelineProcessFile:
    """Tests for process_file()."""

    def test_process_file_loads_and_processes(self, tmp_path):
        from PIL import Image

        from edge_model.inference.pipeline import DocumentPipeline

        # Create a small test image file
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        # Create pipeline with mocked internals
        pipeline = object.__new__(DocumentPipeline)
        pipeline._config = PipelineConfig(confidence_threshold=0.7, use_ocr=False)
        pipeline._classifier = MagicMock()
        pipeline._classifier.predict.return_value = ("lieferschein", 0.9)
        pipeline._ocr = None
        pipeline._extractors = {}
        pipeline._validator = MagicMock()

        result = pipeline.process_file(str(img_path))
        assert isinstance(result, ProcessingResult)
        assert result.document_type == DocumentType.lieferschein


class TestPipelineFromConfig:
    """Tests for the from_config class method."""

    @patch("ocr.engine.OCREngine")
    @patch("edge_model.inference.pipeline.ExtractorInference")
    @patch("edge_model.inference.pipeline.ClassifierInference")
    @patch("edge_model.inference.pipeline.SchemaValidator")
    def test_from_config_loads_yaml(self, mock_validator, mock_classifier, mock_extractor, mock_ocr, tmp_path):
        cfg_data = {
            "classifier_model_path": "cls.onnx",
            "extractor_model_paths": {"arztbesuchsbestaetigung": "arzt.onnx"},
            "extractor_tokenizer_paths": {"arztbesuchsbestaetigung": "arzt/"},
            "confidence_threshold": 0.6,
            "use_ocr": True,
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config(str(cfg_file))
        assert pipeline._config.confidence_threshold == 0.6
        mock_classifier.assert_called_once()
