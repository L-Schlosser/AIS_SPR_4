"""Tests for the schema validation utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_model.inference.validator import SchemaValidator

SCHEMAS_DIR = Path("data/schemas")


@pytest.fixture()
def validator() -> SchemaValidator:
    return SchemaValidator(schemas_dir=SCHEMAS_DIR)


@pytest.fixture()
def valid_arztbesuch() -> dict:
    return {
        "document_type": "arztbesuchsbestaetigung",
        "patient_name": "Max Mustermann",
        "doctor_name": "Dr. Anna Schmidt",
        "facility_name": "Praxis Schmidt",
        "visit_date": "2024-06-15",
    }


@pytest.fixture()
def valid_reisekosten() -> dict:
    return {
        "document_type": "reisekostenbeleg",
        "vendor_name": "Hotel Sacher",
        "date": "2024-03-10",
        "amount": 159.90,
        "currency": "EUR",
    }


@pytest.fixture()
def valid_lieferschein() -> dict:
    return {
        "document_type": "lieferschein",
        "delivery_note_number": "LS-12345",
        "delivery_date": "2024-01-20",
        "sender": {"name": "Firma ABC GmbH"},
        "recipient": {"name": "Firma XYZ AG"},
    }


# --- Schema loading ---

class TestSchemaLoading:
    def test_loads_all_three_schemas(self, validator: SchemaValidator) -> None:
        for doc_type in ("arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"):
            schema = validator.get_schema(doc_type)
            assert schema["type"] == "object"

    def test_empty_directory(self, tmp_path: Path) -> None:
        v = SchemaValidator(schemas_dir=tmp_path)
        is_valid, errors = v.validate({}, "anything")
        assert not is_valid
        assert "No schema found" in errors[0]

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        v = SchemaValidator(schemas_dir=tmp_path / "does_not_exist")
        is_valid, errors = v.validate({}, "anything")
        assert not is_valid

    def test_get_schema_missing_raises(self, validator: SchemaValidator) -> None:
        with pytest.raises(ValueError, match="No schema found"):
            validator.get_schema("nonexistent_type")

    def test_loads_custom_schema(self, tmp_path: Path) -> None:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        (tmp_path / "custom.json").write_text(json.dumps(schema))
        v = SchemaValidator(schemas_dir=tmp_path)
        assert v.get_schema("custom") == schema


# --- Validation of correct data ---

class TestValidData:
    def test_valid_arztbesuch(self, validator: SchemaValidator, valid_arztbesuch: dict) -> None:
        is_valid, errors = validator.validate(valid_arztbesuch, "arztbesuchsbestaetigung")
        assert is_valid
        assert errors == []

    def test_valid_reisekosten(self, validator: SchemaValidator, valid_reisekosten: dict) -> None:
        is_valid, errors = validator.validate(valid_reisekosten, "reisekostenbeleg")
        assert is_valid
        assert errors == []

    def test_valid_lieferschein(self, validator: SchemaValidator, valid_lieferschein: dict) -> None:
        is_valid, errors = validator.validate(valid_lieferschein, "lieferschein")
        assert is_valid
        assert errors == []

    def test_valid_with_optional_fields(self, validator: SchemaValidator) -> None:
        data = {
            "document_type": "arztbesuchsbestaetigung",
            "patient_name": "Max Mustermann",
            "doctor_name": "Dr. Anna Schmidt",
            "facility_name": "Praxis Schmidt",
            "facility_address": "Hauptstraße 1, 1010 Wien",
            "visit_date": "2024-06-15",
            "visit_time": "14:30",
            "duration_minutes": 30,
            "confidence": 0.95,
        }
        is_valid, errors = validator.validate(data, "arztbesuchsbestaetigung")
        assert is_valid
        assert errors == []


# --- Validation of incorrect data ---

class TestInvalidData:
    def test_missing_required_fields(self, validator: SchemaValidator) -> None:
        data = {"document_type": "arztbesuchsbestaetigung"}
        is_valid, errors = validator.validate(data, "arztbesuchsbestaetigung")
        assert not is_valid
        assert len(errors) > 0
        assert any("required" in e.lower() for e in errors)

    def test_wrong_type_for_field(self, validator: SchemaValidator, valid_arztbesuch: dict) -> None:
        valid_arztbesuch["duration_minutes"] = "thirty"
        is_valid, errors = validator.validate(valid_arztbesuch, "arztbesuchsbestaetigung")
        assert not is_valid

    def test_extra_fields_rejected(self, validator: SchemaValidator, valid_arztbesuch: dict) -> None:
        valid_arztbesuch["unexpected_field"] = "surprise"
        is_valid, errors = validator.validate(valid_arztbesuch, "arztbesuchsbestaetigung")
        assert not is_valid

    def test_invalid_date_format(self, validator: SchemaValidator, valid_arztbesuch: dict) -> None:
        valid_arztbesuch["visit_date"] = "15.06.2024"
        is_valid, errors = validator.validate(valid_arztbesuch, "arztbesuchsbestaetigung")
        assert not is_valid

    def test_amount_must_be_positive(self, validator: SchemaValidator, valid_reisekosten: dict) -> None:
        valid_reisekosten["amount"] = 0
        is_valid, errors = validator.validate(valid_reisekosten, "reisekostenbeleg")
        assert not is_valid

    def test_invalid_currency(self, validator: SchemaValidator, valid_reisekosten: dict) -> None:
        valid_reisekosten["currency"] = "USD"
        is_valid, errors = validator.validate(valid_reisekosten, "reisekostenbeleg")
        assert not is_valid

    def test_empty_sender_name(self, validator: SchemaValidator, valid_lieferschein: dict) -> None:
        valid_lieferschein["sender"] = {"name": ""}
        is_valid, errors = validator.validate(valid_lieferschein, "lieferschein")
        assert not is_valid


# --- Wrong document type ---

class TestWrongDocumentType:
    def test_unknown_document_type(self, validator: SchemaValidator) -> None:
        is_valid, errors = validator.validate({}, "unknown_type")
        assert not is_valid
        assert "No schema found" in errors[0]

    def test_arztbesuch_data_against_reisekosten_schema(
        self, validator: SchemaValidator, valid_arztbesuch: dict
    ) -> None:
        is_valid, _ = validator.validate(valid_arztbesuch, "reisekostenbeleg")
        assert not is_valid
