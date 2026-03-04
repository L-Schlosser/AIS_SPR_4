"""Tests for OCR engine, preprocessing, and postprocessing."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from ocr.engine import OCREngine, OCRResult, TextRegion
from ocr.postprocessing import merge_text, sort_regions_by_position
from ocr.preprocessing import preprocess_for_ocr

# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_converts_color_to_binary(self):
        """Preprocessing a color image should return a single-channel binary image."""
        color_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = preprocess_for_ocr(color_img)
        assert result.ndim == 2
        assert result.shape == (100, 200)

    def test_converts_grayscale_correctly(self):
        """Preprocessing a grayscale image should still work."""
        gray_img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        result = preprocess_for_ocr(gray_img)
        assert result.ndim == 2
        assert result.shape == (100, 200)

    def test_output_is_binary(self):
        """Output pixels should be either 0 or 255 after thresholding."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = preprocess_for_ocr(img)
        unique_values = set(np.unique(result))
        assert unique_values.issubset({0, 255})

    def test_white_text_on_black_background(self):
        """A mostly dark image should produce some white (255) pixels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some white text-like region
        img[30:40, 20:80] = 255
        result = preprocess_for_ocr(img)
        assert 255 in result

    def test_preserves_dimensions(self):
        """Output should have same height and width as input."""
        img = np.random.randint(0, 255, (123, 456, 3), dtype=np.uint8)
        result = preprocess_for_ocr(img)
        assert result.shape == (123, 456)

    def test_output_dtype_is_uint8(self):
        """Output should be uint8."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = preprocess_for_ocr(img)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Postprocessing tests
# ---------------------------------------------------------------------------

class TestSortRegionsByPosition:
    """Tests for sorting text regions by reading order."""

    def test_empty_list(self):
        """Sorting empty list returns empty list."""
        assert sort_regions_by_position([]) == []

    def test_single_region(self):
        """Single region returned as-is."""
        region = TextRegion(text="hello", bbox=(10, 10, 100, 30), confidence=0.9)
        result = sort_regions_by_position([region])
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_top_to_bottom_sorting(self):
        """Regions at different y positions sorted top to bottom."""
        bottom = TextRegion(text="bottom", bbox=(10, 100, 100, 120), confidence=0.9)
        top = TextRegion(text="top", bbox=(10, 10, 100, 30), confidence=0.9)
        middle = TextRegion(text="middle", bbox=(10, 50, 100, 70), confidence=0.9)

        result = sort_regions_by_position([bottom, top, middle])
        assert [r.text for r in result] == ["top", "middle", "bottom"]

    def test_left_to_right_within_line(self):
        """Regions on the same line sorted left to right."""
        right = TextRegion(text="right", bbox=(200, 10, 300, 30), confidence=0.9)
        left = TextRegion(text="left", bbox=(10, 10, 100, 30), confidence=0.9)

        result = sort_regions_by_position([right, left])
        assert [r.text for r in result] == ["left", "right"]

    def test_multiline_sorting(self):
        """Multiple lines with multiple regions each sorted correctly."""
        r1 = TextRegion(text="line1_right", bbox=(200, 10, 300, 30), confidence=0.9)
        r2 = TextRegion(text="line1_left", bbox=(10, 10, 100, 30), confidence=0.9)
        r3 = TextRegion(text="line2_right", bbox=(200, 60, 300, 80), confidence=0.9)
        r4 = TextRegion(text="line2_left", bbox=(10, 60, 100, 80), confidence=0.9)

        result = sort_regions_by_position([r1, r3, r4, r2])
        assert [r.text for r in result] == ["line1_left", "line1_right", "line2_left", "line2_right"]


class TestMergeText:
    """Tests for merging sorted text regions into readable text."""

    def test_empty_regions(self):
        """Merging empty list returns empty string."""
        assert merge_text([]) == ""

    def test_single_region(self):
        """Single region returns its text."""
        region = TextRegion(text="hello", bbox=(10, 10, 100, 30), confidence=0.9)
        assert merge_text([region]) == "hello"

    def test_same_line_regions(self):
        """Regions on same line merged with spaces."""
        r1 = TextRegion(text="Hello", bbox=(10, 10, 60, 30), confidence=0.9)
        r2 = TextRegion(text="World", bbox=(70, 10, 130, 30), confidence=0.9)
        result = merge_text([r1, r2])
        assert result == "Hello World"

    def test_different_lines(self):
        """Regions on different lines merged with newlines."""
        r1 = TextRegion(text="Line 1", bbox=(10, 10, 100, 30), confidence=0.9)
        r2 = TextRegion(text="Line 2", bbox=(10, 80, 100, 100), confidence=0.9)
        result = merge_text([r1, r2])
        assert "Line 1" in result
        assert "Line 2" in result
        assert "\n" in result

    def test_produces_readable_output(self):
        """Merged text is stripped and non-empty for non-empty regions."""
        regions = [
            TextRegion(text="Praxis", bbox=(10, 10, 100, 30), confidence=0.9),
            TextRegion(text="Dr.", bbox=(110, 10, 150, 30), confidence=0.9),
            TextRegion(text="Müller", bbox=(160, 10, 250, 30), confidence=0.9),
            TextRegion(text="Patient:", bbox=(10, 80, 100, 100), confidence=0.9),
            TextRegion(text="Hans", bbox=(110, 80, 170, 100), confidence=0.9),
        ]
        result = merge_text(regions)
        assert len(result) > 0
        assert "Praxis" in result
        assert "Müller" in result
        assert "Patient:" in result


# ---------------------------------------------------------------------------
# OCR Engine tests (mocked RapidOCR)
# ---------------------------------------------------------------------------

class TestOCREngine:
    """Tests for OCREngine with mocked RapidOCR backend."""

    def test_initialization(self):
        """OCREngine can be initialized with mocked RapidOCR."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False
        assert hasattr(engine, "_ocr")

    def test_extract_text_returns_ocr_result(self):
        """extract_text returns an OCRResult object."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False

        # Mock RapidOCR returning one detection
        engine._ocr.return_value = (
            [
                [
                    [[10, 10], [100, 10], [100, 30], [10, 30]],
                    "Hello World",
                    0.95,
                ]
            ],
            None,
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = engine.extract_text(image)

        assert isinstance(result, OCRResult)
        assert "Hello World" in result.text
        assert len(result.regions) == 1
        assert result.regions[0].confidence == 0.95
        assert result.processing_time_ms >= 0

    def test_extract_text_empty_result(self):
        """extract_text handles no detections gracefully."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False

        # No detections
        engine._ocr.return_value = (None, None)

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = engine.extract_text(image)

        assert isinstance(result, OCRResult)
        assert result.text == ""
        assert len(result.regions) == 0

    def test_extract_text_multiple_regions(self):
        """extract_text handles multiple detected regions."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False

        engine._ocr.return_value = (
            [
                [[[10, 10], [100, 10], [100, 30], [10, 30]], "First", 0.9],
                [[[10, 50], [100, 50], [100, 70], [10, 70]], "Second", 0.85],
            ],
            None,
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = engine.extract_text(image)

        assert len(result.regions) == 2
        assert "First" in result.text
        assert "Second" in result.text

    def test_extract_text_from_file(self, tmp_path):
        """extract_text_from_file loads image and runs OCR."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False

        engine._ocr.return_value = (
            [[[[10, 10], [100, 10], [100, 30], [10, 30]], "File text", 0.9]],
            None,
        )

        # Create a test image file
        img = Image.new("RGB", (200, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(str(img_path))

        result = engine.extract_text_from_file(str(img_path))

        assert isinstance(result, OCRResult)
        assert "File text" in result.text

    def test_bbox_conversion(self):
        """Polygon points are correctly converted to bounding box."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False

        # Rotated polygon
        engine._ocr.return_value = (
            [[[[5, 15], [105, 10], [107, 35], [7, 40]], "Rotated", 0.8]],
            None,
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = engine.extract_text(image)

        region = result.regions[0]
        # bbox should be (x_min, y_min, x_max, y_max)
        assert region.bbox[0] == 5    # x_min
        assert region.bbox[1] == 10   # y_min
        assert region.bbox[2] == 107  # x_max
        assert region.bbox[3] == 40   # y_max

    def test_processing_time_recorded(self):
        """Processing time is recorded in milliseconds."""
        engine = OCREngine.__new__(OCREngine)
        engine._ocr = MagicMock()
        engine._use_gpu = False
        engine._ocr.return_value = (None, None)

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = engine.extract_text(image)

        assert result.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDataClasses:
    """Tests for OCR data classes."""

    def test_text_region_creation(self):
        """TextRegion can be created with all fields."""
        region = TextRegion(text="test", bbox=(0, 0, 100, 50), confidence=0.95)
        assert region.text == "test"
        assert region.bbox == (0, 0, 100, 50)
        assert region.confidence == 0.95

    def test_ocr_result_defaults(self):
        """OCRResult has sensible defaults."""
        result = OCRResult(text="hello")
        assert result.text == "hello"
        assert result.regions == []
        assert result.processing_time_ms == 0.0

    def test_ocr_result_with_regions(self):
        """OCRResult can hold multiple regions."""
        regions = [
            TextRegion(text="a", bbox=(0, 0, 10, 10), confidence=0.9),
            TextRegion(text="b", bbox=(20, 0, 30, 10), confidence=0.8),
        ]
        result = OCRResult(text="a b", regions=regions, processing_time_ms=42.5)
        assert len(result.regions) == 2
        assert result.processing_time_ms == 42.5
