"""ONNX export and quantization for the classification model."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from edge_model.classification.config import ClassificationConfig


def export_to_onnx(model_path: str, output_path: str, config: ClassificationConfig) -> str:
    """Export a PyTorch classifier model to ONNX format.

    Args:
        model_path: Path to saved PyTorch model (.pt state_dict).
        output_path: Path for the output ONNX file.
        config: Classification configuration.

    Returns:
        Path to the exported ONNX file.
    """
    from edge_model.classification.train import create_model

    # Load model
    model = create_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    # Export
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_file),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Validate exported model
    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model: {output_file} ({size_mb:.2f} MB)")

    return str(output_file)


def quantize_model(onnx_path: str, output_path: str) -> str:
    """Apply float16 quantization to an ONNX model.

    Uses float16 conversion which preserves accuracy much better than INT8 dynamic
    quantization for convolution-heavy architectures like EfficientNet.

    Args:
        onnx_path: Path to the input ONNX model.
        output_path: Path for the quantized output model.

    Returns:
        Path to the quantized ONNX file.
    """
    from onnxruntime.transformers import float16

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_file))

    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quantized_size = output_file.stat().st_size / (1024 * 1024)
    print(f"Quantized model: {output_file} ({quantized_size:.2f} MB)")
    reduction_pct = (1 - quantized_size / original_size) * 100
    print(f"Size reduction: {original_size:.2f} MB -> {quantized_size:.2f} MB ({reduction_pct:.1f}%)")

    return str(output_file)


def verify_onnx(onnx_path: str, sample_image_path: str | None = None) -> np.ndarray:
    """Verify an ONNX model by running inference.

    Args:
        onnx_path: Path to the ONNX model.
        sample_image_path: Optional path to a sample image. If None, uses random input.

    Returns:
        Model output array.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if sample_image_path is not None:
        image = Image.open(sample_image_path).convert("RGB").resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        img_array = img_array.transpose(2, 0, 1)
        input_data = img_array[np.newaxis, ...]
    else:
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    output = session.run(None, {"image": input_data})[0]

    # Apply softmax for confidence
    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
    probs = exp_output / exp_output.sum(axis=1, keepdims=True)

    class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
    predicted_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0, predicted_idx])

    print(f"Predicted class: {class_names[predicted_idx]} (confidence: {confidence:.4f})")
    assert output.shape == (1, 3), f"Expected output shape (1, 3), got {output.shape}"

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export classifier to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to PyTorch model (.pt)")
    parser.add_argument("--output-path", type=str, required=True, help="Path for ONNX output")
    args = parser.parse_args()

    cfg = ClassificationConfig()

    # Export
    onnx_path = export_to_onnx(args.model_path, args.output_path, cfg)

    # Quantize
    quantized_path = str(Path(args.output_path).with_stem(Path(args.output_path).stem + "_int8"))
    quantize_model(onnx_path, quantized_path)

    # Verify both
    print("\n--- Verifying original model ---")
    verify_onnx(onnx_path)

    print("\n--- Verifying quantized model ---")
    verify_onnx(quantized_path)
