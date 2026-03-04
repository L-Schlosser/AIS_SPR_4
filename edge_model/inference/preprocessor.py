"""Image preprocessing utilities for ONNX inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePreprocessor:
    """Preprocesses images for classification model inference."""

    @staticmethod
    def prepare_for_classification(image: np.ndarray, size: int = 224) -> np.ndarray:
        """Resize, normalize, and format an image for ONNX classification inference.

        Args:
            image: RGB image as numpy array (H, W, 3) with uint8 or float values.
            size: Target size for both width and height.

        Returns:
            Numpy array of shape (1, 3, size, size) ready for ONNX input.
        """
        pil_image = Image.fromarray(image).convert("RGB").resize((size, size))
        img_array = np.array(pil_image, dtype=np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_array = img_array.transpose(2, 0, 1)
        return img_array[np.newaxis, ...]

    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """Load an image file and return as RGB numpy array.

        Args:
            file_path: Path to the image file.

        Returns:
            RGB image as numpy array (H, W, 3) with uint8 values.
        """
        return np.array(Image.open(file_path).convert("RGB"))
