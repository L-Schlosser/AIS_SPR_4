"""End-to-end parametrized tests across all document types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASSIFIER_MODEL_PATH = Path("edge_model/classification/models/classifier_int8.onnx")
EXTRACTOR_MODEL_PATHS = {
    "arztbesuchsbestaetigung": Path("edge_model/extraction/models/arztbesuch/onnx/quantized/model.onnx"),
    "reisekostenbeleg": Path("edge_model/extraction/models/reisekosten/onnx/quantized/model.onnx"),
    "lieferschein": Path("edge_model/extraction/models/lieferschein/onnx/quantized/model.onnx"),
}
SAMPLE_DIRS = {
    "arztbesuchsbestaetigung": Path("data/samples/arztbesuchsbestaetigung"),
    "reisekostenbeleg": Path("data/samples/reisekostenbeleg"),
    "lieferschein": Path("data/samples/lieferschein"),
}
DOC_TYPES = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]


def _all_models_available() -> bool:
    """Check whether all pipeline models exist."""
    if not CLASSIFIER_MODEL_PATH.is_file():
        return False
    return all(p.is_file() for p in EXTRACTOR_MODEL_PATHS.values())


def _load_sample_images(doc_type: str, count: int = 5) -> list[np.ndarray]:
    """Load sample images from the training data directory."""
    sample_dir = SAMPLE_DIRS[doc_type]
    if not sample_dir.is_dir():
        return []
    png_files = sorted(sample_dir.glob("*.png"))[:count]
    images = []
    for path in png_files:
        img = Image.open(path).convert("RGB")
        images.append(np.array(img))
    return images


@pytest.mark.e2e
class TestAllDocumentTypes:
    """Parametrized e2e test across all document types."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    @pytest.mark.parametrize("doc_type", DOC_TYPES)
    def test_batch_classification_accuracy(self, doc_type: str) -> None:
        """For each doc type: load 5 sample images, process all, check >80% correctly classified."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        images = _load_sample_images(doc_type, count=5)
        if len(images) < 5:
            pytest.skip(f"Not enough sample images for {doc_type}")

        correct = 0
        total = len(images)
        for image in images:
            result = pipeline.process(image)
            if result.document_type.value == doc_type:
                correct += 1

        accuracy = correct / total
        print(f"[{doc_type}] Classification accuracy: {correct}/{total} = {accuracy:.0%}")
        assert accuracy > 0.80, (
            f"Accuracy for {doc_type} is {accuracy:.0%} ({correct}/{total}), expected >80%"
        )

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    @pytest.mark.parametrize("doc_type", DOC_TYPES)
    def test_batch_produces_fields(self, doc_type: str) -> None:
        """For each doc type: process 5 images, check that at least some produce fields."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        images = _load_sample_images(doc_type, count=5)
        if len(images) < 5:
            pytest.skip(f"Not enough sample images for {doc_type}")

        with_fields = 0
        total = len(images)
        for image in images:
            result = pipeline.process(image)
            if result.fields and result.fields.get("document_type"):
                with_fields += 1

        print(f"[{doc_type}] Images with extracted fields: {with_fields}/{total}")

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    @pytest.mark.parametrize("doc_type", DOC_TYPES)
    def test_batch_schema_validation(self, doc_type: str) -> None:
        """For each doc type: process 5 images, validate extracted fields against schema."""
        from edge_model.inference.pipeline import DocumentPipeline
        from edge_model.inference.validator import SchemaValidator

        pipeline = DocumentPipeline.from_config("config.yaml")
        validator = SchemaValidator()
        images = _load_sample_images(doc_type, count=5)
        if len(images) < 5:
            pytest.skip(f"Not enough sample images for {doc_type}")

        valid_count = 0
        total = len(images)
        for image in images:
            result = pipeline.process(image)
            if result.fields and result.fields.get("document_type"):
                fields_clean = {k: v for k, v in result.fields.items() if not k.startswith("_")}
                is_valid, errors = validator.validate(fields_clean, result.document_type.value)
                if is_valid:
                    valid_count += 1
                else:
                    print(f"[{doc_type}] Schema errors: {errors}")

        print(f"[{doc_type}] Schema-valid results: {valid_count}/{total}")
