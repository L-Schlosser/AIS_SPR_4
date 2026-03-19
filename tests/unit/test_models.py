"""Tests for api/models.py — Pydantic models and DocumentType enum."""

import pytest
from pydantic import ValidationError

from api.models import (
    ArztbesuchsbestaetigungResult,
    DeliveryItem,
    DocumentType,
    LieferscheinResult,
    ProcessingResult,
    ReisekostenbelegResult,
    SenderRecipient,
)

# --- DocumentType enum ---


class TestDocumentType:
    def test_enum_values(self):
        assert DocumentType.arztbesuchsbestaetigung.value == "arztbesuchsbestaetigung"
        assert DocumentType.reisekostenbeleg.value == "reisekostenbeleg"
        assert DocumentType.lieferschein.value == "lieferschein"

    def test_enum_has_three_members(self):
        assert len(DocumentType) == 3

    def test_enum_from_string(self):
        assert DocumentType("arztbesuchsbestaetigung") == DocumentType.arztbesuchsbestaetigung


# --- ArztbesuchsbestaetigungResult ---


class TestArztbesuchsbestaetigungResult:
    def test_valid_full(self):
        result = ArztbesuchsbestaetigungResult(
            document_type="arztbesuchsbestaetigung",
            patient_name="Max Mustermann",
            doctor_name="Dr. Schmidt",
            facility_name="Praxis Schmidt",
            facility_address="Hauptstr. 1, 1010 Wien",
            visit_date="2024-05-15",
            visit_time="09:30",
            duration_minutes=30,
            confidence=0.95,
        )
        assert result.patient_name == "Max Mustermann"
        assert result.duration_minutes == 30

    def test_valid_minimal(self):
        result = ArztbesuchsbestaetigungResult(
            document_type="arztbesuchsbestaetigung",
            patient_name="Anna",
            doctor_name="Dr. Müller",
            facility_name="Klinik",
            visit_date="2024-01-01",
        )
        assert result.facility_address is None
        assert result.visit_time is None

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="arztbesuchsbestaetigung",
                patient_name="Max",
                # doctor_name missing
                facility_name="Praxis",
                visit_date="2024-01-01",
            )

    def test_invalid_date_format(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="arztbesuchsbestaetigung",
                patient_name="Max",
                doctor_name="Dr. X",
                facility_name="Praxis",
                visit_date="15.05.2024",
            )

    def test_invalid_time_format(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="arztbesuchsbestaetigung",
                patient_name="Max",
                doctor_name="Dr. X",
                facility_name="Praxis",
                visit_date="2024-01-01",
                visit_time="9:30",  # must be HH:MM
            )

    def test_duration_out_of_range(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="arztbesuchsbestaetigung",
                patient_name="Max",
                doctor_name="Dr. X",
                facility_name="Praxis",
                visit_date="2024-01-01",
                duration_minutes=500,
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="arztbesuchsbestaetigung",
                patient_name="Max",
                doctor_name="Dr. X",
                facility_name="Praxis",
                visit_date="2024-01-01",
                unknown_field="value",
            )

    def test_wrong_document_type(self):
        with pytest.raises(ValidationError):
            ArztbesuchsbestaetigungResult(
                document_type="lieferschein",
                patient_name="Max",
                doctor_name="Dr. X",
                facility_name="Praxis",
                visit_date="2024-01-01",
            )


# --- ReisekostenbelegResult ---


class TestReisekostenbelegResult:
    def test_valid_full(self):
        result = ReisekostenbelegResult(
            document_type="reisekostenbeleg",
            vendor_name="Hotel Austria",
            vendor_address="Ringstr. 5, 1010 Wien",
            date="2024-03-20",
            amount=125.50,
            currency="EUR",
            vat_rate=20.0,
            vat_amount=20.92,
            category="hotel",
            description="Übernachtung",
            receipt_number="R-2024-001",
            confidence=0.88,
        )
        assert result.amount == 125.50
        assert result.category == "hotel"

    def test_valid_minimal(self):
        result = ReisekostenbelegResult(
            document_type="reisekostenbeleg",
            vendor_name="Taxi Wien",
            date="2024-06-01",
            amount=35.00,
            currency="EUR",
        )
        assert result.vat_rate is None

    def test_missing_required_amount(self):
        with pytest.raises(ValidationError):
            ReisekostenbelegResult(
                document_type="reisekostenbeleg",
                vendor_name="Taxi",
                date="2024-01-01",
                currency="EUR",
            )

    def test_amount_must_be_positive(self):
        with pytest.raises(ValidationError):
            ReisekostenbelegResult(
                document_type="reisekostenbeleg",
                vendor_name="Taxi",
                date="2024-01-01",
                amount=0,
                currency="EUR",
            )

    def test_invalid_currency(self):
        with pytest.raises(ValidationError):
            ReisekostenbelegResult(
                document_type="reisekostenbeleg",
                vendor_name="Taxi",
                date="2024-01-01",
                amount=10.0,
                currency="USD",
            )

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ReisekostenbelegResult(
                document_type="reisekostenbeleg",
                vendor_name="Taxi",
                date="2024-01-01",
                amount=10.0,
                currency="EUR",
                category="flight",
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ReisekostenbelegResult(
                document_type="reisekostenbeleg",
                vendor_name="Taxi",
                date="2024-01-01",
                amount=10.0,
                currency="EUR",
                extra="nope",
            )


# --- LieferscheinResult ---


class TestLieferscheinResult:
    def test_valid_full(self):
        result = LieferscheinResult(
            document_type="lieferschein",
            delivery_note_number="LS-12345",
            delivery_date="2024-07-10",
            sender=SenderRecipient(name="Firma A", address="Industriestr. 1"),
            recipient=SenderRecipient(name="Firma B", address="Lagerstr. 2"),
            order_number="PO-9999",
            items=[
                DeliveryItem(description="Schrauben M8", quantity=100, unit="Stück"),
                DeliveryItem(description="Muttern M8", quantity=100, unit="Stück"),
            ],
            total_weight="5.2 kg",
            confidence=0.92,
        )
        assert result.delivery_note_number == "LS-12345"
        assert len(result.items) == 2

    def test_valid_minimal(self):
        result = LieferscheinResult(
            document_type="lieferschein",
            delivery_note_number="LS-00001",
            delivery_date="2024-01-01",
            sender=SenderRecipient(name="Sender GmbH"),
            recipient=SenderRecipient(name="Empfänger AG"),
        )
        assert result.items is None
        assert result.order_number is None

    def test_missing_sender(self):
        with pytest.raises(ValidationError):
            LieferscheinResult(
                document_type="lieferschein",
                delivery_note_number="LS-00001",
                delivery_date="2024-01-01",
                recipient=SenderRecipient(name="Empfänger"),
            )

    def test_sender_requires_name(self):
        with pytest.raises(ValidationError):
            SenderRecipient(address="Some address")

    def test_item_requires_fields(self):
        with pytest.raises(ValidationError):
            DeliveryItem(description="Widget")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            LieferscheinResult(
                document_type="lieferschein",
                delivery_note_number="LS-1",
                delivery_date="2024-01-01",
                sender=SenderRecipient(name="A"),
                recipient=SenderRecipient(name="B"),
                bonus="nope",
            )


# --- ProcessingResult ---


class TestProcessingResult:
    def test_valid(self):
        result = ProcessingResult(
            document_type=DocumentType.arztbesuchsbestaetigung,
            fields={"patient_name": "Max"},
            confidence=0.9,
        )
        assert result.document_type == DocumentType.arztbesuchsbestaetigung
        assert result.raw_text is None

    def test_with_raw_text(self):
        result = ProcessingResult(
            document_type=DocumentType.reisekostenbeleg,
            fields={},
            confidence=0.5,
            raw_text="OCR output text",
        )
        assert result.raw_text == "OCR output text"

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            ProcessingResult(
                document_type=DocumentType.lieferschein,
                confidence=0.9,
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            ProcessingResult(
                document_type=DocumentType.lieferschein,
                fields={},
                confidence=1.5,
            )
