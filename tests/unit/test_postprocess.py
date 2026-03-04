"""Tests for NER postprocessing: BIO tag merging and document-specific postprocessors."""

import json
from pathlib import Path

import jsonschema
import pytest

from edge_model.extraction.postprocess import (
    POSTPROCESSORS,
    bio_tags_to_fields,
    postprocess_arztbesuch,
    postprocess_lieferschein,
    postprocess_reisekosten,
)

# ---------------------------------------------------------------------------
# bio_tags_to_fields
# ---------------------------------------------------------------------------


class TestBioTagsToFields:
    """Test BIO tag merging into field dicts."""

    def test_simple_b_tag(self):
        tokens = ["Max"]
        tags = ["B-PATIENT"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Max"}

    def test_b_and_i_tags(self):
        tokens = ["Max", "Mustermann"]
        tags = ["B-PATIENT", "I-PATIENT"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Max Mustermann"}

    def test_multiple_fields(self):
        tokens = ["Max", "Mustermann", "Dr.", "Schmidt"]
        tags = ["B-PATIENT", "I-PATIENT", "B-DOCTOR", "I-DOCTOR"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Max Mustermann", "doctor": "Dr. Schmidt"}

    def test_o_tags_ignored(self):
        tokens = ["Patient", ":", "Max", "Mustermann"]
        tags = ["O", "O", "B-PATIENT", "I-PATIENT"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Max Mustermann"}

    def test_subword_cleaning(self):
        tokens = ["Muster", "##mann"]
        tags = ["B-PATIENT", "I-PATIENT"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Muster mann"}

    def test_empty_input(self):
        result = bio_tags_to_fields([], [])
        assert result == {}

    def test_all_o_tags(self):
        result = bio_tags_to_fields(["Hello", "World"], ["O", "O"])
        assert result == {}

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            bio_tags_to_fields(["a", "b"], ["O"])

    def test_duplicate_b_tag_keeps_first(self):
        tokens = ["2024-01-01", "2024-02-02"]
        tags = ["B-DATE", "B-DATE"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"date": "2024-01-01"}

    def test_i_tag_without_b_ignored(self):
        tokens = ["Max", "Mustermann"]
        tags = ["O", "I-PATIENT"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {}

    def test_mixed_fields_with_o(self):
        tokens = ["Patient", ":", "Max", "Datum", ":", "01.01.2024"]
        tags = ["O", "O", "B-PATIENT", "O", "O", "B-DATE"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"patient": "Max", "date": "01.01.2024"}

    def test_multi_token_subword(self):
        tokens = ["Arzt", "##praxis", "##zentrum"]
        tags = ["B-FACILITY", "I-FACILITY", "I-FACILITY"]
        result = bio_tags_to_fields(tokens, tags)
        assert result == {"facility": "Arzt praxis zentrum"}


# ---------------------------------------------------------------------------
# postprocess_arztbesuch
# ---------------------------------------------------------------------------


class TestPostprocessArztbesuch:
    """Test Arztbesuchsbestätigung postprocessor."""

    def test_full_fields(self):
        raw = {
            "patient": "Max Mustermann",
            "doctor": "Dr. Anna Schmidt",
            "facility": "Praxis Schmidt",
            "address": "Hauptstraße 1, 1010 Wien",
            "date": "15.03.2024",
            "time": "14:30",
            "duration": "45",
        }
        result = postprocess_arztbesuch(raw)
        assert result["document_type"] == "arztbesuchsbestaetigung"
        assert result["patient_name"] == "Max Mustermann"
        assert result["doctor_name"] == "Dr. Anna Schmidt"
        assert result["facility_name"] == "Praxis Schmidt"
        assert result["facility_address"] == "Hauptstraße 1, 1010 Wien"
        assert result["visit_date"] == "2024-03-15"
        assert result["visit_time"] == "14:30"
        assert result["duration_minutes"] == 45

    def test_date_parsing_dot_format(self):
        result = postprocess_arztbesuch({"date": "01.06.2023"})
        assert result["visit_date"] == "2023-06-01"

    def test_date_parsing_iso_passthrough(self):
        result = postprocess_arztbesuch({"date": "2024-01-15"})
        assert result["visit_date"] == "2024-01-15"

    def test_time_parsing_dot_format(self):
        result = postprocess_arztbesuch({"time": "14.30"})
        assert result["visit_time"] == "14:30"

    def test_time_parsing_uhr_format(self):
        result = postprocess_arztbesuch({"time": "9 Uhr 30"})
        assert result["visit_time"] == "09:30"

    def test_duration_parsing(self):
        result = postprocess_arztbesuch({"duration": "45 min"})
        assert result["duration_minutes"] == 45

    def test_minimal_fields(self):
        result = postprocess_arztbesuch({})
        assert result == {"document_type": "arztbesuchsbestaetigung"}

    def test_schema_valid_output(self):
        raw = {
            "patient": "Max Mustermann",
            "doctor": "Dr. Schmidt",
            "facility": "Praxis Wien",
            "date": "15.03.2024",
        }
        result = postprocess_arztbesuch(raw)
        schema = _load_schema("arztbesuchsbestaetigung")
        jsonschema.validate(result, schema)


# ---------------------------------------------------------------------------
# postprocess_reisekosten
# ---------------------------------------------------------------------------


class TestPostprocessReisekosten:
    """Test Reisekostenbeleg postprocessor."""

    def test_full_fields(self):
        raw = {
            "vendor": "Hotel Sacher",
            "vaddress": "Philharmonikerstr. 4, 1010 Wien",
            "date": "20.05.2024",
            "amount": "199,50",
            "currency": "EUR",
            "vat_rate": "10",
            "vat_amount": "19,95",
            "category": "Hotel",
            "desc": "Übernachtung Einzelzimmer",
            "receipt_num": "R-2024-001",
        }
        result = postprocess_reisekosten(raw)
        assert result["document_type"] == "reisekostenbeleg"
        assert result["vendor_name"] == "Hotel Sacher"
        assert result["vendor_address"] == "Philharmonikerstr. 4, 1010 Wien"
        assert result["date"] == "2024-05-20"
        assert result["amount"] == 199.50
        assert result["currency"] == "EUR"
        assert result["vat_rate"] == 10.0
        assert result["vat_amount"] == 19.95
        assert result["category"] == "hotel"
        assert result["description"] == "Übernachtung Einzelzimmer"
        assert result["receipt_number"] == "R-2024-001"

    def test_amount_with_euro_sign(self):
        result = postprocess_reisekosten({"amount": "€ 49,90"})
        assert result["amount"] == 49.90

    def test_amount_with_german_thousands(self):
        result = postprocess_reisekosten({"amount": "1.234,56"})
        assert result["amount"] == 1234.56

    def test_default_currency_eur(self):
        result = postprocess_reisekosten({})
        assert result["currency"] == "EUR"

    def test_invalid_category_defaults_to_other(self):
        result = postprocess_reisekosten({"category": "FLUG"})
        assert result["category"] == "other"

    def test_valid_categories(self):
        for cat in ["hotel", "restaurant", "transport", "other"]:
            result = postprocess_reisekosten({"category": cat})
            assert result["category"] == cat

    def test_schema_valid_output(self):
        raw = {
            "vendor": "Gasthof Adler",
            "date": "10.04.2024",
            "amount": "35,00",
            "currency": "EUR",
        }
        result = postprocess_reisekosten(raw)
        schema = _load_schema("reisekostenbeleg")
        jsonschema.validate(result, schema)


# ---------------------------------------------------------------------------
# postprocess_lieferschein
# ---------------------------------------------------------------------------


class TestPostprocessLieferschein:
    """Test Lieferschein postprocessor."""

    def test_full_fields(self):
        raw = {
            "delnr": "LS-12345",
            "deldate": "01.06.2024",
            "sender": "Firma ABC GmbH",
            "saddr": "Industriestr. 5, 4020 Linz",
            "recip": "Firma XYZ AG",
            "raddr": "Handelskai 10, 1020 Wien",
            "ordnr": "PO-9999",
            "item_desc": "Schrauben M8",
            "item_qty": "500",
            "item_unit": "Stk",
            "weight": "25 kg",
        }
        result = postprocess_lieferschein(raw)
        assert result["document_type"] == "lieferschein"
        assert result["delivery_note_number"] == "LS-12345"
        assert result["delivery_date"] == "2024-06-01"
        assert result["sender"] == {"name": "Firma ABC GmbH", "address": "Industriestr. 5, 4020 Linz"}
        assert result["recipient"] == {"name": "Firma XYZ AG", "address": "Handelskai 10, 1020 Wien"}
        assert result["order_number"] == "PO-9999"
        assert result["items"] == [{"description": "Schrauben M8", "quantity": 500.0, "unit": "Stk"}]
        assert result["total_weight"] == "25 kg"

    def test_sender_name_only(self):
        result = postprocess_lieferschein({"sender": "Test GmbH"})
        assert result["sender"] == {"name": "Test GmbH"}

    def test_recipient_with_address(self):
        result = postprocess_lieferschein({"recip": "Empfänger AG", "raddr": "Weg 1"})
        assert result["recipient"] == {"name": "Empfänger AG", "address": "Weg 1"}

    def test_item_defaults(self):
        result = postprocess_lieferschein({"item_desc": "Widget"})
        assert result["items"] == [{"description": "Widget", "quantity": 0, "unit": "Stk"}]

    def test_no_items_when_no_desc(self):
        result = postprocess_lieferschein({"delnr": "LS-001"})
        assert "items" not in result

    def test_schema_valid_output(self):
        raw = {
            "delnr": "LS-99999",
            "deldate": "2024-06-01",
            "sender": "Sender GmbH",
            "recip": "Empfänger AG",
        }
        result = postprocess_lieferschein(raw)
        schema = _load_schema("lieferschein")
        jsonschema.validate(result, schema)

    def test_minimal_fields(self):
        result = postprocess_lieferschein({})
        assert result == {"document_type": "lieferschein"}


# ---------------------------------------------------------------------------
# POSTPROCESSORS dict
# ---------------------------------------------------------------------------


class TestPostprocessors:
    """Test the POSTPROCESSORS mapping."""

    def test_all_document_types_present(self):
        assert "arztbesuchsbestaetigung" in POSTPROCESSORS
        assert "reisekostenbeleg" in POSTPROCESSORS
        assert "lieferschein" in POSTPROCESSORS

    def test_callables(self):
        for name, func in POSTPROCESSORS.items():
            assert callable(func), f"{name} is not callable"

    def test_postprocessors_accept_empty_dict(self):
        for name, func in POSTPROCESSORS.items():
            result = func({})
            assert "document_type" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases across postprocessing."""

    def test_bio_to_fields_then_arztbesuch(self):
        """End-to-end: BIO tags → raw fields → postprocessed output."""
        tokens = ["Patient", ":", "Max", "Mustermann", "Datum", ":", "15.03.2024"]
        tags = ["O", "O", "B-PATIENT", "I-PATIENT", "O", "O", "B-DATE"]
        raw = bio_tags_to_fields(tokens, tags)
        result = postprocess_arztbesuch(raw)
        assert result["patient_name"] == "Max Mustermann"
        assert result["visit_date"] == "2024-03-15"

    def test_bio_to_fields_then_reisekosten(self):
        tokens = ["Gasthof", "Adler", "35,00", "EUR"]
        tags = ["B-VENDOR", "I-VENDOR", "B-AMOUNT", "B-CURRENCY"]
        raw = bio_tags_to_fields(tokens, tags)
        result = postprocess_reisekosten(raw)
        assert result["vendor_name"] == "Gasthof Adler"
        assert result["amount"] == 35.0
        assert result["currency"] == "EUR"

    def test_bio_to_fields_then_lieferschein(self):
        tokens = ["LS-001", "Firma", "ABC"]
        tags = ["B-DELNR", "B-SENDER", "I-SENDER"]
        raw = bio_tags_to_fields(tokens, tags)
        result = postprocess_lieferschein(raw)
        assert result["delivery_note_number"] == "LS-001"
        assert result["sender"] == {"name": "Firma ABC"}

    def test_date_slash_format(self):
        result = postprocess_arztbesuch({"date": "15/03/2024"})
        assert result["visit_date"] == "2024-03-15"

    def test_unparseable_amount_skipped(self):
        result = postprocess_reisekosten({"amount": "abc"})
        assert "amount" not in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_schema(doc_type: str) -> dict:
    """Load a JSON schema from data/schemas/."""
    schema_path = Path(__file__).parent.parent.parent / "data" / "schemas" / f"{doc_type}.json"
    with open(schema_path) as f:
        return json.load(f)
