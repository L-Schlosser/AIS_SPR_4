"""Integration test for OCR engine with real RapidOCR inference."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr.engine import OCREngine


@pytest.mark.integration
class TestOCREngineIntegration:
    """Integration tests that run real OCR on generated images."""

    @staticmethod
    def _create_text_image(text: str, width: int = 800, height: int = 200, font_size: int = 48) -> np.ndarray:
        """Create a white image with black text drawn on it."""
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
        draw.text((50, 60), text, fill="black", font=font)
        return np.array(img)

    def test_extract_known_text(self) -> None:
        """OCR should extract 'Hello' and '123' from an image containing 'Hello World 123'."""
        image = self._create_text_image("Hello World 123")
        engine = OCREngine()
        result = engine.extract_text(image)

        extracted = result.text.lower()
        assert "hello" in extracted, f"Expected 'hello' in extracted text, got: {result.text}"
        assert "123" in result.text, f"Expected '123' in extracted text, got: {result.text}"

    def test_processing_time_is_measured(self) -> None:
        """Processing time should be a positive number."""
        image = self._create_text_image("Test timing 456")
        engine = OCREngine()
        result = engine.extract_text(image)

        assert result.processing_time_ms > 0, "Processing time should be positive"
        print(f"OCR processing time: {result.processing_time_ms:.1f} ms")

    def test_regions_have_bounding_boxes(self) -> None:
        """Detected regions should have valid bounding boxes."""
        image = self._create_text_image("Bounding box test")
        engine = OCREngine()
        result = engine.extract_text(image)

        assert len(result.regions) > 0, "Expected at least one detected region"
        for region in result.regions:
            x_min, y_min, x_max, y_max = region.bbox
            assert x_min < x_max, f"Invalid bbox x: {region.bbox}"
            assert y_min < y_max, f"Invalid bbox y: {region.bbox}"
            assert region.confidence > 0, f"Confidence should be positive: {region.confidence}"

    def test_extract_text_from_file(self, tmp_path) -> None:
        """OCR should work on a file path as well."""
        image = self._create_text_image("File test 789")
        img = Image.fromarray(image)
        file_path = tmp_path / "test_image.png"
        img.save(file_path)

        engine = OCREngine()
        result = engine.extract_text_from_file(str(file_path))

        assert "789" in result.text, f"Expected '789' in extracted text, got: {result.text}"

    def test_empty_image_returns_empty_text(self) -> None:
        """A blank white image should return empty or near-empty text."""
        img = Image.new("RGB", (400, 200), color="white")
        image = np.array(img)

        engine = OCREngine()
        result = engine.extract_text(image)

        # A blank image should have no or very few detected regions
        assert len(result.regions) <= 1, f"Expected at most 1 region on blank image, got {len(result.regions)}"
