"""Tests for classifier inference wrapper and image preprocessor."""

from unittest.mock import MagicMock, patch

import numpy as np

from edge_model.inference.preprocessor import IMAGENET_MEAN, IMAGENET_STD, ImagePreprocessor


class TestImagePreprocessor:
    """Tests for the ImagePreprocessor class."""

    def test_output_shape(self):
        """prepare_for_classification returns (1, 3, 224, 224)."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image)
        assert result.shape == (1, 3, 224, 224)

    def test_custom_size(self):
        """Custom size parameter changes output dimensions."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image, size=128)
        assert result.shape == (1, 3, 128, 128)

    def test_output_dtype(self):
        """Output is float32."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image)
        assert result.dtype == np.float32

    def test_normalization_range(self):
        """Normalized values fall roughly in [-2.5, 2.5] range (ImageNet stats)."""
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image)
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_zero_image_normalization(self):
        """Black image (all zeros) normalizes to -mean/std per channel."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image)
        expected_per_channel = -IMAGENET_MEAN / IMAGENET_STD
        for c in range(3):
            np.testing.assert_allclose(result[0, c, 0, 0], expected_per_channel[c], atol=1e-5)

    def test_white_image_normalization(self):
        """White image (all 255) normalizes to (1-mean)/std per channel."""
        image = np.full((224, 224, 3), 255, dtype=np.uint8)
        result = ImagePreprocessor.prepare_for_classification(image)
        expected_per_channel = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
        for c in range(3):
            np.testing.assert_allclose(result[0, c, 0, 0], expected_per_channel[c], atol=1e-5)

    def test_load_image(self, tmp_path):
        """load_image returns a uint8 RGB numpy array."""
        from PIL import Image

        img = Image.new("RGB", (100, 80), color=(128, 64, 32))
        path = str(tmp_path / "test.png")
        img.save(path)

        result = ImagePreprocessor.load_image(path)
        assert result.shape == (80, 100, 3)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 128  # R
        assert result[0, 0, 1] == 64   # G
        assert result[0, 0, 2] == 32   # B

    def test_load_image_converts_to_rgb(self, tmp_path):
        """load_image converts grayscale to RGB."""
        from PIL import Image

        img = Image.new("L", (50, 50), color=100)
        path = str(tmp_path / "gray.png")
        img.save(path)

        result = ImagePreprocessor.load_image(path)
        assert result.shape == (50, 50, 3)
        assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]  # R == G == B


class TestClassifierInference:
    """Tests for the ClassifierInference class using mocked ONNX session."""

    CLASS_NAMES = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]

    def _make_mock_inference(self, logits: np.ndarray):
        """Create a ClassifierInference with a mocked ONNX session."""
        from edge_model.inference.classifier_inference import ClassifierInference

        mock_session = MagicMock()
        mock_session.run.return_value = [logits]

        with patch("edge_model.inference.classifier_inference.ort.InferenceSession", return_value=mock_session):
            ci = ClassifierInference("dummy.onnx", self.CLASS_NAMES)
        return ci

    def test_predict_returns_tuple(self):
        """predict returns (class_name, confidence) tuple."""
        logits = np.array([[10.0, 1.0, 1.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = ci.predict(image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_class_name(self):
        """predict returns correct class name for highest logit."""
        logits = np.array([[1.0, 10.0, 1.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        name, _ = ci.predict(image)
        assert name == "reisekostenbeleg"

    def test_predict_confidence_range(self):
        """Confidence is between 0 and 1."""
        logits = np.array([[1.0, 5.0, 1.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        _, confidence = ci.predict(image)
        assert 0.0 <= confidence <= 1.0

    def test_predict_high_confidence_for_dominant_logit(self):
        """Large logit difference produces high confidence."""
        logits = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        _, confidence = ci.predict(image)
        assert confidence > 0.99

    def test_predict_third_class(self):
        """predict correctly returns the third class."""
        logits = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        name, _ = ci.predict(image)
        assert name == "lieferschein"

    def test_predict_batch_empty(self):
        """predict_batch with empty list returns empty list."""
        logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        ci = self._make_mock_inference(logits)

        result = ci.predict_batch([])
        assert result == []

    def test_predict_batch_multiple(self):
        """predict_batch returns correct count of results."""
        batch_logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=np.float32,
        )
        ci = self._make_mock_inference(batch_logits)
        images = [np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8) for _ in range(3)]

        results = ci.predict_batch(images)
        assert len(results) == 3
        assert results[0][0] == "arztbesuchsbestaetigung"
        assert results[1][0] == "reisekostenbeleg"
        assert results[2][0] == "lieferschein"

    def test_predict_calls_session_with_correct_input_name(self):
        """Session.run is called with input name 'image'."""
        logits = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        mock_session = MagicMock()
        mock_session.run.return_value = [logits]

        with patch("edge_model.inference.classifier_inference.ort.InferenceSession", return_value=mock_session):
            from edge_model.inference.classifier_inference import ClassifierInference

            ci = ClassifierInference("dummy.onnx", self.CLASS_NAMES)

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ci.predict(image)

        call_args = mock_session.run.call_args
        assert call_args[0][0] is None
        input_dict = call_args[0][1]
        assert "image" in input_dict
        assert input_dict["image"].shape == (1, 3, 224, 224)


class TestSoftmax:
    """Tests for the softmax utility."""

    def test_softmax_sums_to_one(self):
        from edge_model.inference.classifier_inference import _softmax

        result = _softmax(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_softmax_uniform(self):
        from edge_model.inference.classifier_inference import _softmax

        result = _softmax(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_softmax_large_values(self):
        """Softmax handles large values without overflow."""
        from edge_model.inference.classifier_inference import _softmax

        result = _softmax(np.array([1000.0, 0.0, 0.0]))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)
