"""Classifier inference wrapper for ONNX models."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from edge_model.inference.preprocessor import ImagePreprocessor


class ClassifierInference:
    """Runs document classification inference using an ONNX model."""

    def __init__(self, model_path: str, class_names: list[str]) -> None:
        """Initialize the classifier with an ONNX model.

        Args:
            model_path: Path to the ONNX model file.
            class_names: List of class name strings, indexed by model output.
        """
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._class_names = class_names

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Classify a single image.

        Args:
            image: RGB image as numpy array (H, W, 3).

        Returns:
            Tuple of (class_name, confidence).
        """
        input_data = ImagePreprocessor.prepare_for_classification(image)
        logits = self._session.run(None, {"image": input_data})[0]
        return self._logits_to_prediction(logits[0])

    def predict_batch(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Classify a batch of images.

        Args:
            images: List of RGB images as numpy arrays (H, W, 3).

        Returns:
            List of (class_name, confidence) tuples.
        """
        if not images:
            return []

        batch = np.concatenate(
            [ImagePreprocessor.prepare_for_classification(img) for img in images],
            axis=0,
        )
        logits = self._session.run(None, {"image": batch})[0]
        return [self._logits_to_prediction(logits[i]) for i in range(logits.shape[0])]

    def _logits_to_prediction(self, logits: np.ndarray) -> tuple[str, float]:
        """Convert raw logits to (class_name, confidence)."""
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return self._class_names[idx], float(probs[idx])


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax over a 1-D array."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
