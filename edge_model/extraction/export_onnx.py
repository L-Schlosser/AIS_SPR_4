"""ONNX export and quantization for NER extraction models."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

from edge_model.extraction.labels import LABEL_SETS, get_id2label


def export_ner_to_onnx(model_dir: str, output_path: str) -> str:
    """Export a HuggingFace NER model to ONNX format using optimum.

    Args:
        model_dir: Path to saved HuggingFace model directory (contains config.json, model.safetensors).
        output_path: Path for the output ONNX directory.

    Returns:
        Path to the exported ONNX directory.
    """
    from optimum.onnxruntime import ORTModelForTokenClassification

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to ONNX via optimum
    model = ORTModelForTokenClassification.from_pretrained(model_dir, export=True)
    model.save_pretrained(str(output_dir))

    # Also copy tokenizer files if present in source dir
    tokenizer_path = Path(model_dir)
    if (tokenizer_path / "tokenizer.json").exists() or (tokenizer_path / "tokenizer_config.json").exists():
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(tokenizer_path))
        tokenizer.save_pretrained(str(output_dir))

    # Print model size
    onnx_files = list(output_dir.glob("*.onnx"))
    for onnx_file in onnx_files:
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"Exported ONNX model: {onnx_file} ({size_mb:.2f} MB)")

    return str(output_dir)


def quantize_ner_model(onnx_dir: str, output_dir: str) -> str:
    """Apply INT8 dynamic quantization to an ONNX NER model.

    Args:
        onnx_dir: Path to the ONNX model directory.
        output_dir: Path for the quantized output directory.

    Returns:
        Path to the quantized model directory.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find the ONNX model file
    onnx_files = list(onnx_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in {onnx_dir}")

    model_file = onnx_files[0]
    quantized_file = out_path / model_file.name

    quantize_dynamic(
        str(model_file),
        str(quantized_file),
        weight_type=QuantType.QInt8,
    )

    original_size = model_file.stat().st_size / (1024 * 1024)
    quantized_size = quantized_file.stat().st_size / (1024 * 1024)
    print(f"Quantized model: {quantized_file} ({quantized_size:.2f} MB)")
    reduction_pct = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    print(f"Size reduction: {original_size:.2f} MB -> {quantized_size:.2f} MB ({reduction_pct:.1f}%)")

    # Copy non-ONNX files (config, tokenizer) to output dir
    for f in onnx_path.iterdir():
        if f.suffix != ".onnx" and f.is_file():
            dest = out_path / f.name
            if not dest.exists():
                dest.write_bytes(f.read_bytes())

    return str(out_path)


def verify_ner_onnx(model_dir: str, tokenizer_dir: str, sample_text: str) -> np.ndarray:
    """Verify an ONNX NER model by running inference on sample text.

    Args:
        model_dir: Path to ONNX model directory.
        tokenizer_dir: Path to tokenizer directory.
        sample_text: Sample text to run inference on.

    Returns:
        Model output logits array.
    """
    # Find ONNX file
    onnx_files = list(Path(model_dir).glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in {model_dir}")

    session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)

    # Tokenize
    encoding = tokenizer(sample_text, return_tensors="np", padding="max_length", truncation=True, max_length=256)

    # Run inference
    input_feed = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    outputs = session.run(None, input_feed)
    logits = outputs[0]

    # Get predictions
    predictions = np.argmax(logits, axis=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    print(f"Input text: {sample_text}")
    print("Predicted tags:")
    for token, pred_id in zip(tokens, predictions):
        if token in ("[PAD]", "[CLS]", "[SEP]"):
            continue
        print(f"  {token}: tag_id={pred_id}")

    seq_len = encoding["input_ids"].shape[1]
    num_labels = logits.shape[-1]
    expected_shape = (1, seq_len, num_labels)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NER models to ONNX")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path for ONNX output directory")
    parser.add_argument("--document-type", type=str, default=None, help="Document type for verification labels")
    args = parser.parse_args()

    # Export
    onnx_dir = export_ner_to_onnx(args.model_dir, args.output_dir)

    # Quantize
    quantized_dir = str(Path(args.output_dir) / "quantized")
    quantize_ner_model(onnx_dir, quantized_dir)

    # Verify
    sample = "Bestätigung Arztbesuch Patient Max Mustermann Datum 15.03.2024"
    print("\n--- Verifying exported model ---")
    verify_ner_onnx(onnx_dir, args.model_dir, sample)

    print("\n--- Verifying quantized model ---")
    verify_ner_onnx(quantized_dir, args.model_dir, sample)

    # Print label info if document type specified
    if args.document_type and args.document_type in LABEL_SETS:
        labels = LABEL_SETS[args.document_type]
        id2label = get_id2label(labels)
        print(f"\nLabel mapping for {args.document_type}:")
        for idx, label in id2label.items():
            print(f"  {idx}: {label}")
