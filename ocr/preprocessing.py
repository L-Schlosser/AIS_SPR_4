"""Image preprocessing utilities for improving OCR accuracy."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess an image for better OCR results.

    Steps:
    1. Convert to grayscale if colored
    2. Apply adaptive thresholding for better text contrast
    3. Return preprocessed image
    """
    pil_image = Image.fromarray(image)

    # Convert to grayscale if colored
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")

    # Apply slight sharpening to improve text edges
    pil_image = pil_image.filter(ImageFilter.SHARPEN)

    # Apply adaptive thresholding via point operation
    # Use a simple threshold based on mean pixel value
    arr = np.array(pil_image, dtype=np.float32)
    threshold = _adaptive_threshold(arr, block_size=31)
    binary = ((arr > threshold) * 255).astype(np.uint8)

    return binary


def _adaptive_threshold(image: np.ndarray, block_size: int = 31) -> np.ndarray:
    """Compute adaptive threshold using local mean filtering.

    For each pixel, the threshold is the mean of its local neighborhood
    minus a constant offset, providing good binarization for documents
    with uneven lighting.
    """
    # Use PIL box blur to compute local mean
    pil_img = Image.fromarray(image.astype(np.uint8))
    radius = block_size // 2
    blurred = pil_img.filter(ImageFilter.BoxBlur(radius))
    local_mean = np.array(blurred, dtype=np.float32)

    # Threshold is local mean minus a small constant
    offset = 10.0
    return local_mean - offset
