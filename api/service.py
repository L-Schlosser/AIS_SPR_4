"""Document processing service layer."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from api.models import DocumentType, ProcessingResult
from edge_model.inference.config import load_config
from edge_model.inference.pipeline import DocumentPipeline
from edge_model.inference.validator import SchemaValidator


class DocumentService:
    """High-level service for processing document images.

    Wraps the DocumentPipeline with convenience methods for
    byte-based input, schema access, and supported type queries.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the service by loading config and creating the pipeline.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self._config = load_config(config_path)
        self._pipeline = DocumentPipeline(self._config)
        self._validator = SchemaValidator(self._config.schemas_dir)

    def process_image(self, image_bytes: bytes) -> ProcessingResult:
        """Process a document image from raw bytes.

        Args:
            image_bytes: Raw JPEG or PNG image bytes.

        Returns:
            ProcessingResult with classification, extracted fields, and metadata.

        Raises:
            ValueError: If the image bytes cannot be decoded.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Failed to decode image: {exc}") from exc
        image = np.array(img)
        return self._pipeline.process(image)

    def process_image_file(self, file_path: str) -> ProcessingResult:
        """Process a document image from a file path.

        Args:
            file_path: Path to a JPEG or PNG image file.

        Returns:
            ProcessingResult with all pipeline outputs.
        """
        return self._pipeline.process_file(file_path)

    def get_supported_types(self) -> list[str]:
        """Return the list of supported document types.

        Returns:
            List of document type string values.
        """
        return [dt.value for dt in DocumentType]

    def get_schema(self, document_type: str) -> dict:
        """Return the JSON schema for a given document type.

        Args:
            document_type: One of the supported document type strings.

        Returns:
            The raw JSON schema as a dict.

        Raises:
            ValueError: If the document type is not recognized.
        """
        return self._validator.get_schema(document_type)
