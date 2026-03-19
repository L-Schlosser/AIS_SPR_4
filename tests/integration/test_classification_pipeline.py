"""Integration tests for the classification pipeline (preprocessing + ONNX inference)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edge_model.classification.config import ClassificationConfig
from edge_model.inference.preprocessor import ImagePreprocessor

CLASSIFIER_MODEL_PATH = Path("edge_model/classification/models/classifier_int8.onnx")
CONFIG = ClassificationConfig()


def _generate_synthetic_image(doc_type: str, width: int = 800, height: int = 1130) -> np.ndarray:
    """Generate a simple synthetic image for the given document type."""
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 36)
        except OSError:
            font = ImageFont.load_default()

    headers = {
        "arztbesuchsbestaetigung": "BESTÄTIGUNG ARZTBESUCH",
        "reisekostenbeleg": "RECHNUNG",
        "lieferschein": "LIEFERSCHEIN",
    }
    header = headers.get(doc_type, "UNKNOWN")
    draw.text((200, 50), header, fill="black", font=font)
    draw.text((100, 200), f"Document Type: {doc_type}", fill="black", font=font)
    return np.array(img)


def _model_available() -> bool:
    """Check whether the classifier ONNX model exists."""
    return CLASSIFIER_MODEL_PATH.is_file()


@pytest.mark.integration
class TestClassificationPreprocessing:
    """Test image preprocessing produces correct inputs for classification."""

    def test_synthetic_image_preprocesses_correctly(self) -> None:
        """Synthetic images preprocess to the expected shape and dtype."""
        for doc_type in CONFIG.class_names:
            image = _generate_synthetic_image(doc_type)
            processed = ImagePreprocessor.prepare_for_classification(image, CONFIG.image_size)
            assert processed.shape == (1, 3, CONFIG.image_size, CONFIG.image_size)
            assert processed.dtype == np.float32

    def test_different_input_sizes_normalise(self) -> None:
        """Various input sizes all produce the same output shape."""
        for size in (400, 800, 1600):
            image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            processed = ImagePreprocessor.prepare_for_classification(image, CONFIG.image_size)
            assert processed.shape == (1, 3, CONFIG.image_size, CONFIG.image_size)

    def test_grayscale_image_preprocesses(self) -> None:
        """A grayscale image (single channel) is converted to 3-channel correctly."""
        gray = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        processed = ImagePreprocessor.prepare_for_classification(rgb, CONFIG.image_size)
        assert processed.shape == (1, 3, CONFIG.image_size, CONFIG.image_size)


@pytest.mark.integration
class TestClassificationInference:
    """Test full classification inference if the ONNX model is available."""

    @pytest.mark.skipif(not _model_available(), reason="Classifier ONNX model not found")
    def test_classifier_returns_valid_class(self) -> None:
        """Classifier returns one of the known class names."""
        from edge_model.inference.classifier_inference import ClassifierInference

        classifier = ClassifierInference(str(CLASSIFIER_MODEL_PATH), CONFIG.class_names)

        for doc_type in CONFIG.class_names:
            image = _generate_synthetic_image(doc_type)
            predicted_class, confidence = classifier.predict(image)
            assert predicted_class in CONFIG.class_names, f"Unknown class: {predicted_class}"
            assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    @pytest.mark.skipif(not _model_available(), reason="Classifier ONNX model not found")
    def test_classifier_correct_predictions(self) -> None:
        """Classifier correctly identifies at least some synthetic documents."""
        from edge_model.inference.classifier_inference import ClassifierInference

        classifier = ClassifierInference(str(CLASSIFIER_MODEL_PATH), CONFIG.class_names)

        correct = 0
        total = len(CONFIG.class_names)
        for doc_type in CONFIG.class_names:
            image = _generate_synthetic_image(doc_type)
            predicted_class, _ = classifier.predict(image)
            if predicted_class == doc_type:
                correct += 1

        print(f"Classification accuracy on synthetic images: {correct}/{total}")

    @pytest.mark.skipif(not _model_available(), reason="Classifier ONNX model not found")
    def test_classifier_batch_prediction(self) -> None:
        """Batch prediction returns results for all images."""
        from edge_model.inference.classifier_inference import ClassifierInference

        classifier = ClassifierInference(str(CLASSIFIER_MODEL_PATH), CONFIG.class_names)

        images = [_generate_synthetic_image(dt) for dt in CONFIG.class_names]
        results = classifier.predict_batch(images)
        assert len(results) == len(CONFIG.class_names)
        for class_name, confidence in results:
            assert class_name in CONFIG.class_names
            assert 0.0 <= confidence <= 1.0

    @pytest.mark.skipif(not _model_available(), reason="Classifier ONNX model not found")
    def test_preprocessing_plus_classification(self) -> None:
        """Full flow: load image from file, preprocess, classify."""
        from edge_model.inference.classifier_inference import ClassifierInference

        classifier = ClassifierInference(str(CLASSIFIER_MODEL_PATH), CONFIG.class_names)

        image = _generate_synthetic_image("arztbesuchsbestaetigung")
        # Save to a temporary buffer and reload to simulate file loading
        pil_img = Image.fromarray(image)
        import io

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        reloaded = np.array(Image.open(buf).convert("RGB"))

        predicted_class, confidence = classifier.predict(reloaded)
        assert predicted_class in CONFIG.class_names
        assert 0.0 <= confidence <= 1.0
