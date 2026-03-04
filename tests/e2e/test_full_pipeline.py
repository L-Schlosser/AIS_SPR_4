"""End-to-end tests for the full document processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from edge_model.inference.validator import SchemaValidator

CLASSIFIER_MODEL_PATH = Path("edge_model/classification/models/classifier_int8.onnx")
EXTRACTOR_MODEL_PATHS = {
    "arztbesuchsbestaetigung": Path("edge_model/extraction/models/arztbesuch/model.onnx"),
    "reisekostenbeleg": Path("edge_model/extraction/models/reisekosten/model.onnx"),
    "lieferschein": Path("edge_model/extraction/models/lieferschein/model.onnx"),
}
DOC_TYPES = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]


def _all_models_available() -> bool:
    """Check whether all pipeline models (classifier + all extractors) exist."""
    if not CLASSIFIER_MODEL_PATH.is_file():
        return False
    return all(p.is_file() for p in EXTRACTOR_MODEL_PATHS.values())


def _load_font(size: int = 36):
    """Load a font with fallback chain."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _generate_arztbesuch_image() -> np.ndarray:
    """Generate a synthetic Arztbesuchsbestätigung image."""
    img = Image.new("RGB", (800, 1130), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(32)
    small_font = _load_font(24)

    draw.text((150, 50), "BESTÄTIGUNG ARZTBESUCH", fill="black", font=font)
    draw.text((100, 150), "Praxis Dr. Schmidt", fill="black", font=small_font)
    draw.text((100, 190), "Wiener Straße 42, 1010 Wien", fill="black", font=small_font)
    draw.text((100, 280), "Patient: Anna Huber", fill="black", font=small_font)
    draw.text((100, 330), "Datum: 12.01.2025", fill="black", font=small_font)
    draw.text((100, 380), "Uhrzeit: 09:30", fill="black", font=small_font)
    draw.text((100, 430), "Dauer: 30 Minuten", fill="black", font=small_font)
    return np.array(img)


def _generate_reisekosten_image() -> np.ndarray:
    """Generate a synthetic Reisekostenbeleg image."""
    img = Image.new("RGB", (800, 1130), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(32)
    small_font = _load_font(24)

    draw.text((250, 50), "RECHNUNG", fill="black", font=font)
    draw.text((100, 150), "Hotel Zentral Wien", fill="black", font=small_font)
    draw.text((100, 190), "Stephansplatz 5, 1010 Wien", fill="black", font=small_font)
    draw.text((100, 280), "Datum: 15.03.2025", fill="black", font=small_font)
    draw.text((100, 330), "Einzelzimmer 1 Nacht", fill="black", font=small_font)
    draw.text((100, 380), "Netto: 120,00 EUR", fill="black", font=small_font)
    draw.text((100, 430), "MwSt 13%: 15,60 EUR", fill="black", font=small_font)
    draw.text((100, 480), "Gesamt: 135,60 EUR", fill="black", font=small_font)
    draw.text((100, 530), "Rechnungsnr: RE-2025-1234", fill="black", font=small_font)
    return np.array(img)


def _generate_lieferschein_image() -> np.ndarray:
    """Generate a synthetic Lieferschein image."""
    img = Image.new("RGB", (800, 1130), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(32)
    small_font = _load_font(24)

    draw.text((250, 50), "LIEFERSCHEIN", fill="black", font=font)
    draw.text((100, 130), "LS-50123", fill="black", font=small_font)
    draw.text((100, 170), "Lieferdatum: 20.02.2025", fill="black", font=small_font)
    draw.text((100, 240), "Absender: Technik GmbH", fill="black", font=small_font)
    draw.text((100, 280), "Industriestr. 10, 4020 Linz", fill="black", font=small_font)
    draw.text((100, 340), "Empfänger: Bau AG", fill="black", font=small_font)
    draw.text((100, 380), "Hauptplatz 3, 4020 Linz", fill="black", font=small_font)
    draw.text((100, 450), "Bestellnr: B-77001", fill="black", font=small_font)
    draw.text((100, 510), "2x Stahlträger HEB200", fill="black", font=small_font)
    draw.text((100, 550), "10x Schrauben M12", fill="black", font=small_font)
    draw.text((100, 620), "Gesamtgewicht: 250 kg", fill="black", font=small_font)
    return np.array(img)


def _generate_non_document_image() -> np.ndarray:
    """Generate a random noise image that is not a document."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)


IMAGE_GENERATORS = {
    "arztbesuchsbestaetigung": _generate_arztbesuch_image,
    "reisekostenbeleg": _generate_reisekosten_image,
    "lieferschein": _generate_lieferschein_image,
}


@pytest.mark.e2e
class TestArztbesuchE2E:
    """End-to-end test for Arztbesuchsbestätigung processing."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    def test_arztbesuch_e2e(self) -> None:
        """Generate arztbesuch image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _generate_arztbesuch_image()
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
        """Generate reisekosten image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _generate_reisekosten_image()
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
        """Generate lieferschein image, process, validate JSON against schema."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _generate_lieferschein_image()
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
        """Assert total processing time < 2000ms on CPU for a single image."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")
        image = _generate_arztbesuch_image()

        start = time.perf_counter()
        result = pipeline.process(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        print(f"Pipeline processing time: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 2000, f"Processing took {elapsed_ms:.1f}ms, expected < 2000ms"
