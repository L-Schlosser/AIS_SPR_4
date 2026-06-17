"""Export fine-tuned PyTorch model to ONNX with dynamic axes and INT8 quantization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _OnnxWrapper(torch.nn.Module):
    """Wrap HF model so ONNX export receives positional tensor inputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_fp32_onnx(model, output_path: Path, opset: int = 14) -> None:
    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    input_ids = torch.ones((1, 128), dtype=torch.long)
    attention_mask = torch.ones((1, 128), dtype=torch.long)

    # dynamo=False uses the legacy ONNX exporter (more reliable on Windows)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"FP32 ONNX saved: {output_path}")


def export_with_optimum(pytorch_dir: Path, output_path: Path) -> None:
    """Fallback export via Hugging Face Optimum."""
    from optimum.onnxruntime import ORTModelForTokenClassification

    ort_model = ORTModelForTokenClassification.from_pretrained(
        str(pytorch_dir), export=True
    )
    ort_model.save_pretrained(output_path.parent)
    exported = output_path.parent / "model.onnx"
    if exported.exists() and exported != output_path:
        shutil.move(str(exported), str(output_path))
    print(f"Optimum ONNX saved: {output_path}")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 ONNX saved: {int8_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--use-optimum", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pytorch_dir = ROOT / config["paths"]["pytorch_model_dir"]
    onnx_dir = ROOT / config["paths"]["onnx_model_dir"]
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if not (pytorch_dir / "config.json").exists():
        raise FileNotFoundError(
            f"PyTorch model not found in {pytorch_dir}. Run download_model.py first."
        )

    final_path = onnx_dir / config["paths"]["onnx_model_file"]
    fp32_path = onnx_dir / "model_fp32.onnx"
    opset = config["onnx"].get("opset_version", 14)

    if args.use_optimum:
        export_with_optimum(pytorch_dir, final_path if args.no_quantize else fp32_path)
    else:
        print(f"Loading model from {pytorch_dir}")
        model = AutoModelForTokenClassification.from_pretrained(pytorch_dir)
        export_fp32_onnx(model, fp32_path, opset=opset)

    tokenizer = AutoTokenizer.from_pretrained(pytorch_dir)

    if config["onnx"].get("quantize_int8", True) and not args.no_quantize:
        if not fp32_path.exists():
            fp32_path = onnx_dir / "model.onnx"
        quantize_int8(fp32_path, final_path)
        if fp32_path.exists() and fp32_path != final_path:
            fp32_path.unlink()
    elif fp32_path.exists() and final_path != fp32_path:
        shutil.move(str(fp32_path), str(final_path))

    tokenizer_dir = onnx_dir / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    meta_src = pytorch_dir / "model_meta.json"
    meta_dst = onnx_dir / "model_meta.json"
    if meta_src.exists():
        shutil.copy(meta_src, meta_dst)
    else:
        labels = config["labels"]
        meta = {
            "labels": labels,
            "id2label": {str(i): lb for i, lb in enumerate(labels)},
            "max_length": config["model"]["max_length"],
            "fields_by_type": config["fields_by_type"],
            "field_display_de": config.get("field_display_de", {}),
            "tokenizer_cased": config["model"].get("tokenizer_cased", True),
        }
        with open(meta_dst, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"Final model: {final_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
