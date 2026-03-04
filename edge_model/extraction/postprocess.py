"""NER postprocessing: convert BIO tags to structured field dicts."""

from __future__ import annotations

import re
from collections.abc import Callable


def bio_tags_to_fields(tokens: list[str], tags: list[str]) -> dict[str, str]:
    """Merge B-/I- tagged tokens into field values.

    Args:
        tokens: List of text tokens (may include ## subword prefixes).
        tags: List of BIO tags aligned with tokens.

    Returns:
        Dict mapping field_name (lowercase) to concatenated value string.
        For duplicate fields, the first occurrence is kept.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag length mismatch: {len(tokens)} tokens vs {len(tags)} tags")

    fields: dict[str, list[str]] = {}
    current_field: str | None = None

    for token, tag in zip(tokens, tags):
        clean_token = _clean_subword(token)

        if tag.startswith("B-"):
            field_name = tag[2:].lower()
            if field_name not in fields:
                fields[field_name] = []
                current_field = field_name
            else:
                # Duplicate B- tag for same field — keep first occurrence
                current_field = None
        elif tag.startswith("I-") and current_field is not None:
            field_name = tag[2:].lower()
            if field_name == current_field:
                fields[current_field].append(clean_token)
            else:
                current_field = None
        else:
            current_field = None

        if tag.startswith("B-"):
            field_name = tag[2:].lower()
            if len(fields.get(field_name, [])) == 0:
                fields[field_name] = [clean_token]

    return {k: " ".join(v).strip() for k, v in fields.items() if v}


def _clean_subword(token: str) -> str:
    """Remove ## subword prefix from tokenizer artifacts."""
    if token.startswith("##"):
        return token[2:]
    return token


def _parse_date_to_iso(date_str: str) -> str:
    """Try to parse a date string into YYYY-MM-DD format.

    Handles: DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY.
    """
    date_str = date_str.strip()

    # Already ISO format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # DD.MM.YYYY or DD/MM/YYYY
    match = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    # DD-MM-YYYY
    match = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return date_str


def _parse_time(time_str: str) -> str:
    """Parse time string into HH:MM format.

    Handles: HH:MM, H:MM, HHhMM, HH.MM, HH Uhr MM.
    """
    time_str = time_str.strip()

    # Already correct format
    if re.match(r"^[0-2]\d:[0-5]\d$", time_str):
        return time_str

    # HH.MM or HHhMM
    match = re.match(r"^(\d{1,2})[.h](\d{2})$", time_str)
    if match:
        hour, minute = match.groups()
        return f"{int(hour):02d}:{minute}"

    # HH Uhr MM or HH Uhr
    match = re.match(r"^(\d{1,2})\s*Uhr\s*(\d{2})?$", time_str, re.IGNORECASE)
    if match:
        hour = match.group(1)
        minute = match.group(2) or "00"
        return f"{int(hour):02d}:{minute}"

    return time_str


def _parse_int(value_str: str) -> int | None:
    """Parse an integer from a string, stripping non-numeric chars."""
    cleaned = re.sub(r"[^\d]", "", value_str.strip())
    if cleaned:
        return int(cleaned)
    return None


def _parse_float(value_str: str) -> float | None:
    """Parse a float from a string, handling German number format (comma as decimal)."""
    cleaned = value_str.strip()
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[€$£\s]", "", cleaned)
    # Replace German comma decimal with dot
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    elif "," in cleaned and "." in cleaned:
        # 1.234,56 → 1234.56
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def postprocess_arztbesuch(raw_fields: dict[str, str]) -> dict:
    """Postprocess Arztbesuchsbestätigung NER fields into schema-valid dict."""
    result: dict = {"document_type": "arztbesuchsbestaetigung"}

    if "patient" in raw_fields:
        result["patient_name"] = raw_fields["patient"]
    if "doctor" in raw_fields:
        result["doctor_name"] = raw_fields["doctor"]
    if "facility" in raw_fields:
        result["facility_name"] = raw_fields["facility"]
    if "address" in raw_fields:
        result["facility_address"] = raw_fields["address"]
    if "date" in raw_fields:
        result["visit_date"] = _parse_date_to_iso(raw_fields["date"])
    if "time" in raw_fields:
        result["visit_time"] = _parse_time(raw_fields["time"])
    if "duration" in raw_fields:
        parsed = _parse_int(raw_fields["duration"])
        if parsed is not None:
            result["duration_minutes"] = parsed

    return result


def postprocess_reisekosten(raw_fields: dict[str, str]) -> dict:
    """Postprocess Reisekostenbeleg NER fields into schema-valid dict."""
    result: dict = {"document_type": "reisekostenbeleg"}

    if "vendor" in raw_fields:
        result["vendor_name"] = raw_fields["vendor"]
    if "vaddress" in raw_fields:
        result["vendor_address"] = raw_fields["vaddress"]
    if "date" in raw_fields:
        result["date"] = _parse_date_to_iso(raw_fields["date"])
    if "amount" in raw_fields:
        parsed = _parse_float(raw_fields["amount"])
        if parsed is not None:
            result["amount"] = parsed
    if "currency" in raw_fields:
        result["currency"] = raw_fields["currency"].upper().strip()
    else:
        result["currency"] = "EUR"
    if "vat_rate" in raw_fields:
        parsed = _parse_float(raw_fields["vat_rate"])
        if parsed is not None:
            result["vat_rate"] = parsed
    if "vat_amount" in raw_fields:
        parsed = _parse_float(raw_fields["vat_amount"])
        if parsed is not None:
            result["vat_amount"] = parsed
    if "category" in raw_fields:
        cat = raw_fields["category"].lower().strip()
        valid_cats = {"hotel", "restaurant", "transport", "other"}
        result["category"] = cat if cat in valid_cats else "other"
    if "desc" in raw_fields:
        result["description"] = raw_fields["desc"]
    if "receipt_num" in raw_fields:
        result["receipt_number"] = raw_fields["receipt_num"]

    return result


def postprocess_lieferschein(raw_fields: dict[str, str]) -> dict:
    """Postprocess Lieferschein NER fields into schema-valid dict."""
    result: dict = {"document_type": "lieferschein"}

    if "delnr" in raw_fields:
        result["delivery_note_number"] = raw_fields["delnr"]
    if "deldate" in raw_fields:
        result["delivery_date"] = _parse_date_to_iso(raw_fields["deldate"])

    # Build sender object
    sender: dict = {}
    if "sender" in raw_fields:
        sender["name"] = raw_fields["sender"]
    if "saddr" in raw_fields:
        sender["address"] = raw_fields["saddr"]
    if sender:
        result["sender"] = sender

    # Build recipient object
    recipient: dict = {}
    if "recip" in raw_fields:
        recipient["name"] = raw_fields["recip"]
    if "raddr" in raw_fields:
        recipient["address"] = raw_fields["raddr"]
    if recipient:
        result["recipient"] = recipient

    if "ordnr" in raw_fields:
        result["order_number"] = raw_fields["ordnr"]

    # Group items from individual ITEM_DESC/ITEM_QTY/ITEM_UNIT fields
    # These come as single fields from BIO, so we put them in items array
    if "item_desc" in raw_fields:
        item: dict = {"description": raw_fields["item_desc"]}
        if "item_qty" in raw_fields:
            parsed = _parse_float(raw_fields["item_qty"])
            item["quantity"] = parsed if parsed is not None else 0
        else:
            item["quantity"] = 0
        if "item_unit" in raw_fields:
            item["unit"] = raw_fields["item_unit"]
        else:
            item["unit"] = "Stk"
        result["items"] = [item]

    if "weight" in raw_fields:
        result["total_weight"] = raw_fields["weight"]

    return result


POSTPROCESSORS: dict[str, Callable[[dict[str, str]], dict]] = {
    "arztbesuchsbestaetigung": postprocess_arztbesuch,
    "reisekostenbeleg": postprocess_reisekosten,
    "lieferschein": postprocess_lieferschein,
}
