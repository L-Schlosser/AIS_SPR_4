"""NER BIO label definitions for each document type."""

ARZTBESUCH_LABELS: list[str] = [
    "O",
    "B-PATIENT", "I-PATIENT",
    "B-DOCTOR", "I-DOCTOR",
    "B-FACILITY", "I-FACILITY",
    "B-ADDRESS", "I-ADDRESS",
    "B-DATE", "I-DATE",
    "B-TIME", "I-TIME",
    "B-DURATION",
]

REISEKOSTEN_LABELS: list[str] = [
    "O",
    "B-VENDOR", "I-VENDOR",
    "B-VADDRESS", "I-VADDRESS",
    "B-DATE", "I-DATE",
    "B-AMOUNT",
    "B-CURRENCY",
    "B-VAT_RATE",
    "B-VAT_AMOUNT",
    "B-CATEGORY",
    "B-DESC", "I-DESC",
    "B-RECEIPT_NUM",
]

LIEFERSCHEIN_LABELS: list[str] = [
    "O",
    "B-DELNR",
    "B-DELDATE",
    "B-SENDER", "I-SENDER",
    "B-SADDR", "I-SADDR",
    "B-RECIP", "I-RECIP",
    "B-RADDR", "I-RADDR",
    "B-ORDNR",
    "B-ITEM_DESC", "I-ITEM_DESC",
    "B-ITEM_QTY",
    "B-ITEM_UNIT",
    "B-WEIGHT",
]

LABEL_SETS: dict[str, list[str]] = {
    "arztbesuchsbestaetigung": ARZTBESUCH_LABELS,
    "reisekostenbeleg": REISEKOSTEN_LABELS,
    "lieferschein": LIEFERSCHEIN_LABELS,
}


def get_label2id(labels: list[str]) -> dict[str, int]:
    """Map label strings to integer IDs."""
    return {label: idx for idx, label in enumerate(labels)}


def get_id2label(labels: list[str]) -> dict[int, str]:
    """Map integer IDs to label strings."""
    return {idx: label for idx, label in enumerate(labels)}
