"""Main document processing pipeline orchestrator."""

from __future__ import annotations

import numpy as np
from PIL import Image

from api.models import DocumentType, ProcessingResult
from edge_model.extraction.labels import LABEL_SETS
from edge_model.inference.classifier_inference import ClassifierInference
from edge_model.inference.config import PipelineConfig, load_config
from edge_model.inference.extractor_inference import ExtractorInference
from edge_model.inference.validator import SchemaValidator


class DocumentPipeline:
    """Orchestrates classification, OCR, extraction, and validation."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize all pipeline components from config.

        Args:
            config: Pipeline configuration with model paths and settings.
        """
        self._config = config

        # Classifier
        class_names = [dt.value for dt in DocumentType]
        self._classifier = ClassifierInference(config.classifier_model_path, class_names)

        # Per-document-type extractors
        self._extractors: dict[str, ExtractorInference] = {}
        for doc_type in config.extractor_model_paths:
            labels = LABEL_SETS.get(doc_type, [])
            self._extractors[doc_type] = ExtractorInference(
                model_path=config.extractor_model_paths[doc_type],
                tokenizer_path=config.extractor_tokenizer_paths[doc_type],
                labels=labels,
            )

        # Validator
        self._validator = SchemaValidator(config.schemas_dir)

        # OCR (lazy import to avoid hard dependency when use_ocr=False)
        self._ocr = None
        if config.use_ocr:
            from ocr.engine import OCREngine

            self._ocr = OCREngine()

    def process(self, image: np.ndarray) -> ProcessingResult:
        """Run the full pipeline on an image.

        Steps:
            1. Classify document type
            2. Check confidence threshold
            3. Run OCR to extract text
            4. Run type-specific NER extractor
            5. Validate against JSON schema

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            ProcessingResult with classification, extracted fields, and metadata.
        """
        # Step 1: Classify
        doc_type, confidence = self._classifier.predict(image)

        # Step 2: Low confidence guard
        if confidence < self._config.confidence_threshold:
            return ProcessingResult(
                document_type=DocumentType(doc_type),
                fields={},
                confidence=confidence,
                raw_text=None,
            )

        # Step 3: OCR
        raw_text = None
        if self._ocr is not None:
            ocr_result = self._ocr.extract_text(image)
            raw_text = ocr_result.text

        # Step 4: Extract fields
        fields: dict = {}
        if raw_text and doc_type in self._extractors:
            fields = self._extractors[doc_type].extract_and_postprocess(raw_text, doc_type)

        # Step 5: Validate
        if fields:
            is_valid, errors = self._validator.validate(fields, doc_type)
            if not is_valid:
                fields["_validation_errors"] = errors

        return ProcessingResult(
            document_type=DocumentType(doc_type),
            fields=fields,
            confidence=confidence,
            raw_text=raw_text,
        )

    def process_file(self, file_path: str) -> ProcessingResult:
        """Load an image file and run the pipeline.

        Args:
            file_path: Path to an image file (JPEG/PNG).

        Returns:
            ProcessingResult with all pipeline outputs.
        """
        img = Image.open(file_path).convert("RGB")
        image = np.array(img)
        return self.process(image)

    @classmethod
    def from_config(cls, config_path: str) -> DocumentPipeline:
        """Create a pipeline from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Initialized DocumentPipeline.
        """
        config = load_config(config_path)
        return cls(config)
