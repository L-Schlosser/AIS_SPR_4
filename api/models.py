"""Pydantic models for document processing results."""

from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""

    arztbesuchsbestaetigung = "arztbesuchsbestaetigung"
    reisekostenbeleg = "reisekostenbeleg"
    lieferschein = "lieferschein"


class ArztbesuchsbestaetigungResult(BaseModel):
    """Medical visit confirmation document result."""

    model_config = {"extra": "forbid"}

    document_type: str = Field(..., description="Document type identifier", pattern="^arztbesuchsbestaetigung$")
    patient_name: str = Field(..., description="Name of the patient", min_length=1)
    doctor_name: str = Field(..., description="Name of the doctor", min_length=1)
    facility_name: str = Field(..., description="Name of the medical facility", min_length=1)
    facility_address: str | None = Field(default=None, description="Address of the medical facility")
    visit_date: str = Field(..., description="Date of the visit in ISO format", pattern=r"^\d{4}-\d{2}-\d{2}$")
    visit_time: str | None = Field(default=None, description="Time of the visit", pattern=r"^[0-2][0-9]:[0-5][0-9]$")
    duration_minutes: int | None = Field(default=None, description="Duration of the visit in minutes", ge=1, le=480)
    confidence: float | None = Field(default=None, description="Confidence score", ge=0, le=1)


class ReisekostenbelegResult(BaseModel):
    """Business travel expense receipt result."""

    model_config = {"extra": "forbid"}

    document_type: str = Field(..., description="Document type identifier", pattern="^reisekostenbeleg$")
    vendor_name: str = Field(..., description="Name of the vendor", min_length=1)
    vendor_address: str | None = Field(default=None, description="Address of the vendor")
    date: str = Field(..., description="Date of the expense in ISO format", pattern=r"^\d{4}-\d{2}-\d{2}$")
    amount: float = Field(..., description="Total amount", gt=0)
    currency: str = Field(..., description="Currency code", pattern="^EUR$")
    vat_rate: float | None = Field(default=None, description="VAT rate percentage", ge=0, le=100)
    vat_amount: float | None = Field(default=None, description="VAT amount", ge=0)
    category: str | None = Field(
        default=None, description="Expense category", pattern="^(hotel|restaurant|transport|other)$"
    )
    description: str | None = Field(default=None, description="Description of the expense")
    receipt_number: str | None = Field(default=None, description="Receipt number")
    confidence: float | None = Field(default=None, description="Confidence score", ge=0, le=1)


class SenderRecipient(BaseModel):
    """Sender or recipient information."""

    model_config = {"extra": "forbid"}

    name: str = Field(..., description="Name of the sender or recipient", min_length=1)
    address: str | None = Field(default=None, description="Address")


class DeliveryItem(BaseModel):
    """Item in a delivery note."""

    model_config = {"extra": "forbid"}

    description: str = Field(..., description="Item description")
    quantity: float = Field(..., description="Item quantity", ge=0)
    unit: str = Field(..., description="Unit of measurement")


class LieferscheinResult(BaseModel):
    """Delivery note document result."""

    model_config = {"extra": "forbid"}

    document_type: str = Field(..., description="Document type identifier", pattern="^lieferschein$")
    delivery_note_number: str = Field(..., description="Delivery note number", min_length=1)
    delivery_date: str = Field(..., description="Delivery date in ISO format", pattern=r"^\d{4}-\d{2}-\d{2}$")
    sender: SenderRecipient = Field(..., description="Sender information")
    recipient: SenderRecipient = Field(..., description="Recipient information")
    order_number: str | None = Field(default=None, description="Order reference number")
    items: list[DeliveryItem] | None = Field(default=None, description="List of delivered items")
    total_weight: str | None = Field(default=None, description="Total weight of delivery")
    confidence: float | None = Field(default=None, description="Confidence score", ge=0, le=1)


class ProcessingResult(BaseModel):
    """Result of document processing pipeline."""

    document_type: DocumentType = Field(..., description="Classified document type")
    fields: dict = Field(..., description="Extracted fields")
    confidence: float = Field(..., description="Classification confidence score", ge=0, le=1)
    raw_text: str | None = Field(default=None, description="Raw OCR text")
