"""End-to-end parametrized tests across all document types."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

CLASSIFIER_MODEL_PATH = Path("edge_model/classification/models/classifier_int8.onnx")
EXTRACTOR_MODEL_PATHS = {
    "arztbesuchsbestaetigung": Path("edge_model/extraction/models/arztbesuch/model.onnx"),
    "reisekostenbeleg": Path("edge_model/extraction/models/reisekosten/model.onnx"),
    "lieferschein": Path("edge_model/extraction/models/lieferschein/model.onnx"),
}
DOC_TYPES = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]


def _all_models_available() -> bool:
    """Check whether all pipeline models exist."""
    if not CLASSIFIER_MODEL_PATH.is_file():
        return False
    return all(p.is_file() for p in EXTRACTOR_MODEL_PATHS.values())


def _load_font(size: int = 24):
    """Load a font with fallback chain."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _generate_variant_image(doc_type: str, variant: int) -> np.ndarray:
    """Generate a variant synthetic image for a given document type.

    Each variant has slightly different content/positioning to simulate
    realistic variation in real documents.
    """
    rng = np.random.default_rng(variant)
    width, height = 800, 1130
    # Slight background variation
    bg_val = int(rng.integers(240, 256))
    img = Image.new("RGB", (width, height), color=(bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)
    font = _load_font(32)
    small_font = _load_font(22)

    y_offset = int(rng.integers(30, 80))

    if doc_type == "arztbesuchsbestaetigung":
        draw.text((150, y_offset), "BESTÄTIGUNG ARZTBESUCH", fill="black", font=font)
        names = ["Dr. Müller", "Dr. Schmidt", "Dr. Weber", "Dr. Bauer", "Dr. Wagner"]
        patients = ["Max Huber", "Anna Klein", "Peter Maier", "Lisa Braun", "Georg Moser"]
        draw.text((100, y_offset + 100), f"Praxis {names[variant % len(names)]}", fill="black", font=small_font)
        draw.text((100, y_offset + 150), f"Patient: {patients[variant % len(patients)]}", fill="black", font=small_font)
        draw.text((100, y_offset + 200), f"Datum: {10 + variant}.01.2025", fill="black", font=small_font)
        draw.text((100, y_offset + 250), f"Uhrzeit: {8 + variant}:30", fill="black", font=small_font)
        draw.text((100, y_offset + 300), f"Dauer: {15 + variant * 10} Minuten", fill="black", font=small_font)

    elif doc_type == "reisekostenbeleg":
        headers = ["RECHNUNG", "QUITTUNG", "BELEG", "RECHNUNG", "QUITTUNG"]
        draw.text((250, y_offset), headers[variant % len(headers)], fill="black", font=font)
        vendors = ["Hotel Sacher", "Restaurant Figl", "Taxi Wien", "Hotel Europa", "Gasthof Stern"]
        draw.text((100, y_offset + 100), vendors[variant % len(vendors)], fill="black", font=small_font)
        draw.text((100, y_offset + 150), f"Datum: {10 + variant}.02.2025", fill="black", font=small_font)
        amount = 50.0 + variant * 30.5
        draw.text((100, y_offset + 200), f"Gesamt: {amount:.2f} EUR", fill="black", font=small_font)
        draw.text((100, y_offset + 250), "MwSt 13%", fill="black", font=small_font)

    elif doc_type == "lieferschein":
        draw.text((250, y_offset), "LIEFERSCHEIN", fill="black", font=font)
        draw.text((100, y_offset + 80), f"LS-{50000 + variant}", fill="black", font=small_font)
        draw.text((100, y_offset + 130), f"Lieferdatum: {10 + variant}.03.2025", fill="black", font=small_font)
        senders = ["Logistik GmbH", "Transport AG", "Spedition Huber", "Fracht GmbH", "Lieferservice Wien"]
        recipients = ["Bau AG", "Technik GmbH", "Industrie KG", "Handel OG", "Produktion GmbH"]
        draw.text((100, y_offset + 190), f"Absender: {senders[variant % len(senders)]}", fill="black", font=small_font)
        draw.text(
            (100, y_offset + 240), f"Empfänger: {recipients[variant % len(recipients)]}", fill="black", font=small_font
        )
        draw.text((100, y_offset + 300), f"{variant + 1}x Artikel-{1000 + variant}", fill="black", font=small_font)

    # Add slight noise to a fraction of pixels for variation
    arr = np.array(img)
    noise = rng.integers(-5, 6, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return arr


@pytest.mark.e2e
class TestAllDocumentTypes:
    """Parametrized e2e test across all document types."""

    @pytest.mark.skipif(not _all_models_available(), reason="Pipeline models not available")
    @pytest.mark.parametrize("doc_type", DOC_TYPES)
    def test_batch_classification_accuracy(self, doc_type: str) -> None:
        """For each doc type: generate 5 images, process all, check >80% correctly classified."""
        from edge_model.inference.pipeline import DocumentPipeline

        pipeline = DocumentPipeline.from_config("config.yaml")

        correct = 0
        total = 5
        for variant in range(total):
            image = _generate_variant_image(doc_type, variant)
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

        with_fields = 0
        total = 5
        for variant in range(total):
            image = _generate_variant_image(doc_type, variant)
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

        valid_count = 0
        total = 5
        for variant in range(total):
            image = _generate_variant_image(doc_type, variant)
            result = pipeline.process(image)
            if result.fields and result.fields.get("document_type"):
                fields_clean = {k: v for k, v in result.fields.items() if not k.startswith("_")}
                is_valid, errors = validator.validate(fields_clean, result.document_type.value)
                if is_valid:
                    valid_count += 1
                else:
                    print(f"[{doc_type}][variant {variant}] Schema errors: {errors}")

        print(f"[{doc_type}] Schema-valid results: {valid_count}/{total}")
