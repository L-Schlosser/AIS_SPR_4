"""Tests for the document service layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.models import DocumentType, ProcessingResult

# ---------------------------------------------------------------------------
# DocumentService initialization
# ---------------------------------------------------------------------------


class TestServiceInitialization:
    """Tests for DocumentService __init__."""

    @patch("api.service.SchemaValidator")
    @patch("api.service.DocumentPipeline")
    @patch("api.service.load_config")
    def test_init_loads_config_and_creates_pipeline(self, mock_load, mock_pipeline_cls, mock_validator):
        from api.service import DocumentService

        mock_cfg = MagicMock()
        mock_cfg.schemas_dir = "data/schemas"
        mock_load.return_value = mock_cfg

        service = DocumentService("my_config.yaml")

        mock_load.assert_called_once_with("my_config.yaml")
        mock_pipeline_cls.assert_called_once_with(mock_cfg)
        mock_validator.assert_called_once_with("data/schemas")
        assert service._pipeline is mock_pipeline_cls.return_value

    @patch("api.service.SchemaValidator")
    @patch("api.service.DocumentPipeline")
    @patch("api.service.load_config")
    def test_init_default_config_path(self, mock_load, mock_pipeline_cls, mock_validator):
        from api.service import DocumentService

        mock_cfg = MagicMock()
        mock_cfg.schemas_dir = "data/schemas"
        mock_load.return_value = mock_cfg

        DocumentService()

        mock_load.assert_called_once_with("config.yaml")


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------


class TestProcessImage:
    """Tests for process_image with byte decoding."""

    def _make_service(self):
        """Create a DocumentService with mocked internals."""
        from api.service import DocumentService

        service = object.__new__(DocumentService)
        service._config = MagicMock()
        service._pipeline = MagicMock()
        service._validator = MagicMock()
        return service

    def test_decodes_png_bytes_and_runs_pipeline(self):
        import io

        from PIL import Image

        service = self._make_service()
        expected = ProcessingResult(
            document_type=DocumentType.arztbesuchsbestaetigung,
            fields={"patient_name": "Max"},
            confidence=0.9,
        )
        service._pipeline.process.return_value = expected

        # Create valid PNG bytes
        img = Image.new("RGB", (50, 50), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = service.process_image(png_bytes)

        assert result is expected
        service._pipeline.process.assert_called_once()
        # Verify the image was decoded correctly
        call_arg = service._pipeline.process.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)
        assert call_arg.shape == (50, 50, 3)

    def test_decodes_jpeg_bytes(self):
        import io

        from PIL import Image

        service = self._make_service()
        service._pipeline.process.return_value = ProcessingResult(
            document_type=DocumentType.reisekostenbeleg,
            fields={},
            confidence=0.5,
        )

        img = Image.new("RGB", (30, 40), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        result = service.process_image(jpeg_bytes)

        assert result.document_type == DocumentType.reisekostenbeleg

    def test_invalid_bytes_raises_value_error(self):
        service = self._make_service()

        with pytest.raises(ValueError, match="Failed to decode image"):
            service.process_image(b"not an image")


# ---------------------------------------------------------------------------
# process_image_file
# ---------------------------------------------------------------------------


class TestProcessImageFile:
    """Tests for process_image_file."""

    def test_delegates_to_pipeline_process_file(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)
        service._pipeline = MagicMock()
        expected = ProcessingResult(
            document_type=DocumentType.lieferschein,
            fields={"delivery_note_number": "LS-001"},
            confidence=0.88,
        )
        service._pipeline.process_file.return_value = expected

        result = service.process_image_file("/path/to/image.png")

        assert result is expected
        service._pipeline.process_file.assert_called_once_with("/path/to/image.png")


# ---------------------------------------------------------------------------
# get_supported_types
# ---------------------------------------------------------------------------


class TestGetSupportedTypes:
    """Tests for get_supported_types."""

    def test_returns_all_three_types(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)

        result = service.get_supported_types()

        assert len(result) == 3
        assert "arztbesuchsbestaetigung" in result
        assert "reisekostenbeleg" in result
        assert "lieferschein" in result

    def test_returns_list_of_strings(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)

        result = service.get_supported_types()

        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)


# ---------------------------------------------------------------------------
# get_schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Tests for get_schema."""

    def test_returns_valid_schema_dict(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)
        service._validator = MagicMock()
        service._validator.get_schema.return_value = {
            "type": "object",
            "required": ["document_type"],
        }

        result = service.get_schema("arztbesuchsbestaetigung")

        assert isinstance(result, dict)
        assert "type" in result
        service._validator.get_schema.assert_called_once_with("arztbesuchsbestaetigung")

    def test_unknown_type_raises_value_error(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)
        service._validator = MagicMock()
        service._validator.get_schema.side_effect = ValueError("No schema found")

        with pytest.raises(ValueError, match="No schema found"):
            service.get_schema("unknown_type")

    def test_all_supported_types_delegate_correctly(self):
        from api.service import DocumentService

        service = object.__new__(DocumentService)
        service._validator = MagicMock()
        service._validator.get_schema.return_value = {"type": "object"}

        for doc_type in ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]:
            service.get_schema(doc_type)

        assert service._validator.get_schema.call_count == 3
