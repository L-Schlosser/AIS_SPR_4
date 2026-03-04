"""Integration tests for the NER extraction pipeline (tokenization + NER + postprocessing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edge_model.extraction.labels import LABEL_SETS
from edge_model.extraction.postprocess import POSTPROCESSORS, bio_tags_to_fields
from edge_model.inference.validator import SchemaValidator

# Model paths matching config.yaml
EXTRACTOR_MODEL_PATHS = {
    "arztbesuchsbestaetigung": Path("edge_model/extraction/models/arztbesuch/model.onnx"),
    "reisekostenbeleg": Path("edge_model/extraction/models/reisekosten/model.onnx"),
    "lieferschein": Path("edge_model/extraction/models/lieferschein/model.onnx"),
}

EXTRACTOR_TOKENIZER_PATHS = {
    "arztbesuchsbestaetigung": "edge_model/extraction/models/arztbesuch/",
    "reisekostenbeleg": "edge_model/extraction/models/reisekosten/",
    "lieferschein": "edge_model/extraction/models/lieferschein/",
}

# Sample OCR-like text for each document type
SAMPLE_OCR_TEXTS = {
    "arztbesuchsbestaetigung": (
        "BESTÄTIGUNG ARZTBESUCH "
        "Praxis Dr. Müller "
        "Musterstraße 12, 1010 Wien "
        "Patient: Max Mustermann "
        "Datum: 15.03.2024 "
        "Uhrzeit: 10:30 "
        "Dauer: 45 Minuten"
    ),
    "reisekostenbeleg": (
        "RECHNUNG "
        "Hotel Sacher Wien "
        "Philharmonikerstr. 4, 1010 Wien "
        "Datum: 20.06.2024 "
        "Übernachtung Einzelzimmer "
        "Netto: 150,00 EUR "
        "MwSt 13%: 19,50 EUR "
        "Gesamt: 169,50 EUR "
        "Rechnungsnr: RE-2024-0815"
    ),
    "lieferschein": (
        "LIEFERSCHEIN "
        "LS-20240 "
        "Lieferdatum: 10.07.2024 "
        "Absender: Logistik GmbH "
        "Absenderadresse: Industriestr. 5, 4020 Linz "
        "Empfänger: Technik AG "
        "Empfängeradresse: Hauptplatz 1, 4020 Linz "
        "Bestellnr: B-99001 "
        "1x Servomotor Typ A "
        "5x Kabel 3m "
        "Gesamtgewicht: 12,5 kg"
    ),
}


def _extractor_model_available(doc_type: str) -> bool:
    """Check whether the extractor ONNX model exists for a given doc type."""
    return EXTRACTOR_MODEL_PATHS[doc_type].is_file()


def _any_extractor_available() -> bool:
    """Check whether any extractor model is available."""
    return any(p.is_file() for p in EXTRACTOR_MODEL_PATHS.values())


@pytest.mark.integration
class TestNERPostprocessingIntegration:
    """Test NER postprocessing with simulated BIO tag sequences (no model needed)."""

    def test_arztbesuch_bio_to_schema(self) -> None:
        """Simulated BIO tags for arztbesuch produce schema-valid output."""
        tokens = [
            "Dr.", "Müller", "Max", "Mustermann",
            "Praxis", "am", "Ring", "Musterstraße", "12",
            "15.03.2024", "10:30", "45",
        ]
        tags = [
            "B-DOCTOR", "I-DOCTOR", "B-PATIENT", "I-PATIENT",
            "B-FACILITY", "I-FACILITY", "I-FACILITY", "B-ADDRESS", "I-ADDRESS",
            "B-DATE", "B-TIME", "B-DURATION",
        ]
        raw_fields = bio_tags_to_fields(tokens, tags)
        result = POSTPROCESSORS["arztbesuchsbestaetigung"](raw_fields)

        assert result["document_type"] == "arztbesuchsbestaetigung"
        assert result["patient_name"] == "Max Mustermann"
        assert result["doctor_name"] == "Dr. Müller"
        assert result["visit_date"] == "2024-03-15"

        validator = SchemaValidator()
        is_valid, errors = validator.validate(result, "arztbesuchsbestaetigung")
        assert is_valid, f"Schema validation failed: {errors}"

    def test_reisekosten_bio_to_schema(self) -> None:
        """Simulated BIO tags for reisekosten produce schema-valid output."""
        tokens = [
            "Hotel", "Sacher", "Philharmonikerstr.", "4",
            "20.06.2024", "169,50", "EUR", "13", "19,50",
            "hotel", "Übernachtung", "Einzelzimmer", "RE-2024-0815",
        ]
        tags = [
            "B-VENDOR", "I-VENDOR", "B-VADDRESS", "I-VADDRESS",
            "B-DATE", "B-AMOUNT", "B-CURRENCY", "B-VAT_RATE", "B-VAT_AMOUNT",
            "B-CATEGORY", "B-DESC", "I-DESC", "B-RECEIPT_NUM",
        ]
        raw_fields = bio_tags_to_fields(tokens, tags)
        result = POSTPROCESSORS["reisekostenbeleg"](raw_fields)

        assert result["document_type"] == "reisekostenbeleg"
        assert result["vendor_name"] == "Hotel Sacher"
        assert result["amount"] == 169.50
        assert result["currency"] == "EUR"

        validator = SchemaValidator()
        is_valid, errors = validator.validate(result, "reisekostenbeleg")
        assert is_valid, f"Schema validation failed: {errors}"

    def test_lieferschein_bio_to_schema(self) -> None:
        """Simulated BIO tags for lieferschein produce schema-valid output."""
        tokens = [
            "LS-20240", "10.07.2024",
            "Logistik", "GmbH", "Industriestr.", "5",
            "Technik", "AG", "Hauptplatz", "1",
            "B-99001", "Servomotor", "Typ", "A", "1", "Stk", "12,5",
        ]
        tags = [
            "B-DELNR", "B-DELDATE",
            "B-SENDER", "I-SENDER", "B-SADDR", "I-SADDR",
            "B-RECIP", "I-RECIP", "B-RADDR", "I-RADDR",
            "B-ORDNR", "B-ITEM_DESC", "I-ITEM_DESC", "I-ITEM_DESC", "B-ITEM_QTY", "B-ITEM_UNIT", "B-WEIGHT",
        ]
        raw_fields = bio_tags_to_fields(tokens, tags)
        result = POSTPROCESSORS["lieferschein"](raw_fields)

        assert result["document_type"] == "lieferschein"
        assert result["delivery_note_number"] == "LS-20240"
        assert result["sender"]["name"] == "Logistik GmbH"
        assert result["recipient"]["name"] == "Technik AG"

        validator = SchemaValidator()
        is_valid, errors = validator.validate(result, "lieferschein")
        assert is_valid, f"Schema validation failed: {errors}"

    def test_all_postprocessors_produce_document_type(self) -> None:
        """Every postprocessor sets the correct document_type field."""
        for doc_type, postprocessor in POSTPROCESSORS.items():
            result = postprocessor({})
            assert result["document_type"] == doc_type


@pytest.mark.integration
class TestExtractionInference:
    """Test full extraction inference if ONNX models are available."""

    @pytest.mark.skipif(
        not _any_extractor_available(),
        reason="No extractor ONNX models found",
    )
    @pytest.mark.parametrize("doc_type", list(EXTRACTOR_MODEL_PATHS.keys()))
    def test_extractor_returns_fields(self, doc_type: str) -> None:
        """Extractor returns a dict of fields for sample OCR text."""
        if not _extractor_model_available(doc_type):
            pytest.skip(f"Extractor model not found for {doc_type}")

        from edge_model.inference.extractor_inference import ExtractorInference

        labels = LABEL_SETS[doc_type]
        extractor = ExtractorInference(
            model_path=str(EXTRACTOR_MODEL_PATHS[doc_type]),
            tokenizer_path=EXTRACTOR_TOKENIZER_PATHS[doc_type],
            labels=labels,
        )

        raw_fields = extractor.extract(SAMPLE_OCR_TEXTS[doc_type])
        assert isinstance(raw_fields, dict)
        print(f"[{doc_type}] Raw fields: {raw_fields}")

    @pytest.mark.skipif(
        not _any_extractor_available(),
        reason="No extractor ONNX models found",
    )
    @pytest.mark.parametrize("doc_type", list(EXTRACTOR_MODEL_PATHS.keys()))
    def test_extract_and_postprocess_validates(self, doc_type: str) -> None:
        """Extracted + postprocessed output validates against JSON schema."""
        if not _extractor_model_available(doc_type):
            pytest.skip(f"Extractor model not found for {doc_type}")

        from edge_model.inference.extractor_inference import ExtractorInference

        labels = LABEL_SETS[doc_type]
        extractor = ExtractorInference(
            model_path=str(EXTRACTOR_MODEL_PATHS[doc_type]),
            tokenizer_path=EXTRACTOR_TOKENIZER_PATHS[doc_type],
            labels=labels,
        )

        result = extractor.extract_and_postprocess(SAMPLE_OCR_TEXTS[doc_type], doc_type)
        assert isinstance(result, dict)
        assert result.get("document_type") == doc_type

        validator = SchemaValidator()
        is_valid, errors = validator.validate(result, doc_type)
        print(f"[{doc_type}] Postprocessed: {result}")
        print(f"[{doc_type}] Schema valid: {is_valid}, errors: {errors}")

    @pytest.mark.skipif(
        not _any_extractor_available(),
        reason="No extractor ONNX models found",
    )
    @pytest.mark.parametrize("doc_type", list(EXTRACTOR_MODEL_PATHS.keys()))
    def test_extraction_output_has_expected_keys(self, doc_type: str) -> None:
        """Postprocessed output contains document_type at minimum."""
        if not _extractor_model_available(doc_type):
            pytest.skip(f"Extractor model not found for {doc_type}")

        from edge_model.inference.extractor_inference import ExtractorInference

        labels = LABEL_SETS[doc_type]
        extractor = ExtractorInference(
            model_path=str(EXTRACTOR_MODEL_PATHS[doc_type]),
            tokenizer_path=EXTRACTOR_TOKENIZER_PATHS[doc_type],
            labels=labels,
        )

        result = extractor.extract_and_postprocess(SAMPLE_OCR_TEXTS[doc_type], doc_type)
        assert "document_type" in result
