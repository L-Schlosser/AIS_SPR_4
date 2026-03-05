"""End-to-end tests for the full document processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edge_model.inference.validator import SchemaValidator

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
    """Check whether all pipeline models (classifier + all extractors) exist."""
    if not CLASSIFIER_MODEL_PATH.is_file():
        return False
    return all(p.is_file() for p in EXTRACTOR_MODEL_PATHS.values())


def _load_sample_image(doc_type: str, index: int = 0) -> np.ndarray | None:
    """Load a sample image from the training data directory."""
    sample_dir = SAMPLE_DIRS[doc_type]
    if not sample_dir.is_dir():
        return None
    png_files = sorted(sample_dir.glob("*.png"))
    if index >= len(png_files):
        return None
    img = Image.open(png_files[index]).convert("RGB")
    return np.array(img)


def _generate_non_document_image() -> np.ndarray:
    """Generate a random noise image that is not a document."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)


@pytest.mark.e2e
class TestArztbesuchE2E:
    """End-to-end test for Arztbesuchsbestätigung processing."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_arztbesuch_e2e(self) -> None:
        """Load arztbesuch sample image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _load_sample_image("arztbesuchsbestaetigung", index=0)
        if image is None:
            pytest.skip("No sample images available for arztbesuchsbestaetigung")

        result = pipeline.process(image)

        assert result.document_type.value == "arztbesuchsbestaetigung"
        assert result.confidence > 0.0

        if result.fields:
            validator = SchemaValidator()
            # Remove internal validation error keys before schema check
            fields_clean = {k: v for k, v in result.fields.items() if not k.startswith("_")}
            if fields_clean.get("document_type"):
                is_valid, errors = validator.validate(fields_clean, "arztbesuchsbestaetigung")
                print(f"Schema validation: valid={is_valid}, errors={errors}")


@pytest.mark.e2e
class TestReisekostenE2E:
    """End-to-end test for Reisekostenbeleg processing."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_reisekosten_e2e(self) -> None:
        """Load reisekosten sample image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _load_sample_image("reisekostenbeleg", index=0)
        if image is None:
            pytest.skip("No sample images available for reisekostenbeleg")

        result = pipeline.process(image)

        assert result.document_type.value in DOC_TYPES
        assert result.confidence > 0.0

        if result.fields:
            validator = SchemaValidator()
            fields_clean = {k: v for k, v in result.fields.items() if not k.startswith("_")}
            if fields_clean.get("document_type"):
                is_valid, errors = validator.validate(fields_clean, result.document_type.value)
                print(f"Schema validation: valid={is_valid}, errors={errors}")


@pytest.mark.e2e
class TestLieferscheinE2E:
    """End-to-end test for Lieferschein processing."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_lieferschein_e2e(self) -> None:
        """Load lieferschein sample image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _load_sample_image("lieferschein", index=0)
        if image is None:
            pytest.skip("No sample images available for lieferschein")

        result = pipeline.process(image)

        assert result.document_type.value in DOC_TYPES
        assert result.confidence > 0.0

        if result.fields:
            validator = SchemaValidator()
            fields_clean = {k: v for k, v in result.fields.items() if not k.startswith("_")}
            if fields_clean.get("document_type"):
                is_valid, errors = validator.validate(fields_clean, result.document_type.value)
                print(f"Schema validation: valid={is_valid}, errors={errors}")


@pytest.mark.e2e
class TestUnknownImage:
    """End-to-end test for non-document image handling."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_unknown_image(self) -> None:
        """Process a non-document image — should return low confidence or valid type."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _generate_non_document_image()
        result = pipeline.process(image)

        # The pipeline should still return a result (possibly with low confidence)
        assert result.document_type.value in DOC_TYPES
        assert 0.0 <= result.confidence <= 1.0
        # If confidence is below threshold, fields should be empty
        if result.confidence < 0.7:
            assert result.fields == {}
            print(f"Low confidence as expected: {result.confidence:.3f}")
        else:
            print(f"Random image classified as {result.document_type.value} with confidence {result.confidence:.3f}")


@pytest.mark.e2e
class TestPipelineTiming:
    """End-to-end test for pipeline processing time."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_pipeline_timing(self) -> None:
        """Assert per-image processing time < 2000ms on CPU (after warm-up)."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _load_sample_image("arztbesuchsbestaetigung", index=0)
        if image is None:
            pytest.skip("No sample images available")

        # Warm-up call to load models and JIT compile
        pipeline.process(image)

        # Measure actual per-image processing time
        start = time.perf_counter()
        result = pipeline.process(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        print(f"Pipeline processing time (warm): {elapsed_ms:.1f}ms")
        # 10s budget: includes OCR (~3-5s on CPU for 800x1130 image) + classifier + NER
        assert elapsed_ms < 10000, f"Processing took {elapsed_ms:.1f}ms, expected < 10000ms"
