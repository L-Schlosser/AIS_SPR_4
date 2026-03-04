"""OCR engine using RapidOCR for text extraction from document images."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class TextRegion:
    """A detected text region with bounding box and confidence."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class OCRResult:
    """Result of OCR processing on an image."""

    text: str
    regions: list[TextRegion] = field(default_factory=list)
    processing_time_ms: float = 0.0


class OCREngine:
    """OCR engine wrapping RapidOCR for text extraction."""

    def __init__(self, use_gpu: bool = False) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()
        self._use_gpu = use_gpu

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a numpy image array and return structured results."""
        start = time.perf_counter()

        result = self._ocr(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        regions: list[TextRegion] = []
        if result and result[0]:
            for detection in result[0]:
                box_points = detection[0]
                text = detection[1]
                confidence = float(detection[2])

                # Convert polygon points to bounding box (x_min, y_min, x_max, y_max)
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                bbox = (min(xs), min(ys), max(xs), max(ys))

                regions.append(TextRegion(text=text, bbox=bbox, confidence=confidence))

        from ocr.postprocessing import merge_text, sort_regions_by_position

        sorted_regions = sort_regions_by_position(regions)
        full_text = merge_text(sorted_regions)

        return OCRResult(text=full_text, regions=sorted_regions, processing_time_ms=elapsed_ms)

    def extract_text_from_file(self, file_path: str) -> OCRResult:
        """Load an image from file and run OCR on it."""
        img = Image.open(file_path).convert("RGB")
        image_array = np.array(img)
        return self.extract_text(image_array)
