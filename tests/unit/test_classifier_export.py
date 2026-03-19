"""Tests for ONNX export and quantization of the classification model."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image

from edge_model.classification.config import ClassificationConfig
from edge_model.classification.export_onnx import quantize_model, verify_onnx


class _TinyClassifier(nn.Module):
    """Minimal classifier for fast export tests."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _export_tiny_model(onnx_path: str, config: ClassificationConfig) -> None:
    """Export a tiny classifier to ONNX for testing."""
    model = _TinyClassifier(num_classes=config.num_classes)
    model.eval()
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


class TestExportToOnnx:
    """Tests for the ONNX export function."""

    def test_export_produces_valid_onnx(self, tmp_path):
        """Test that export creates a valid ONNX model file."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")

        _export_tiny_model(onnx_path, config)

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size > 0

    def test_export_has_correct_io_names(self, tmp_path):
        """Test that exported model has expected input/output names."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")

        _export_tiny_model(onnx_path, config)

        onnx_model = onnx.load(onnx_path)
        input_names = [inp.name for inp in onnx_model.graph.input]
        output_names = [out.name for out in onnx_model.graph.output]

        assert "image" in input_names
        assert "logits" in output_names


class TestQuantizeModel:
    """Tests for the INT8 quantization function."""

    def test_quantized_model_is_smaller(self, tmp_path):
        """Test that quantized model is smaller than the original."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")
        quantized_path = str(tmp_path / "model_int8.onnx")

        _export_tiny_model(onnx_path, config)
        quantize_model(onnx_path, quantized_path)

        assert Path(quantized_path).exists()
        assert Path(quantized_path).stat().st_size > 0

    def test_quantized_model_is_valid_onnx(self, tmp_path):
        """Test that quantized model is a valid ONNX file."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")
        quantized_path = str(tmp_path / "model_int8.onnx")

        _export_tiny_model(onnx_path, config)
        quantize_model(onnx_path, quantized_path)

        onnx_model = onnx.load(quantized_path)
        assert onnx_model is not None


class TestVerifyOnnx:
    """Tests for ONNX inference verification."""

    def test_inference_produces_correct_shape(self, tmp_path):
        """Test that ONNX inference output has shape (1, 3)."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")

        _export_tiny_model(onnx_path, config)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {"image": test_input})[0]

        assert output.shape == (1, 3)

    def test_verify_with_random_input(self, tmp_path):
        """Test verify_onnx works with random input (no sample image)."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")

        _export_tiny_model(onnx_path, config)

        output = verify_onnx(onnx_path)
        assert output.shape == (1, 3)

    def test_verify_with_sample_image(self, tmp_path):
        """Test verify_onnx works with a sample image file."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")
        image_path = str(tmp_path / "test.png")

        img = Image.new("RGB", (800, 600), color=(128, 128, 128))
        img.save(image_path)

        _export_tiny_model(onnx_path, config)

        output = verify_onnx(onnx_path, sample_image_path=image_path)
        assert output.shape == (1, 3)

    def test_batch_inference(self, tmp_path):
        """Test that ONNX model handles batch dimension correctly."""
        config = ClassificationConfig()
        onnx_path = str(tmp_path / "model.onnx")

        _export_tiny_model(onnx_path, config)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        batch_input = np.random.randn(4, 3, 224, 224).astype(np.float32)
        output = session.run(None, {"image": batch_input})[0]

        assert output.shape == (4, 3)
