"""Aggregate BIO token predictions into structured JSON.

Strategy (hybrid, no field is ever a concatenation of duplicate spans):

* Deterministic fields (dates by role, amounts, IBAN, UID, email, phone,
  document numbers, currency, address) → German context-anchored regex is the
  PRIMARY signal; the NER model is a fallback.
* Fuzzy/semantic fields (company, patient, doctor, customer, diagnosis) → the
  NER model is PRIMARY; regex / layout heuristics are the fallback.

For every field exactly ONE best value is chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EntitySpan:
    label: str
    text: str
    score: float
    start: int
    end: int


# --- shared sub-patterns ----------------------------------------------------
# Dates allow optional spaces around separators (OCR: "28. 12. 2026").
_DATE = r"(\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*\d{2,4})"
_AMOUNT = r"(\d{1,3}(?:[.\s]\d{3})*[,.]\d{2}|\d+[,.]\d{2})"
_RATE = r"(\d{1,2}\s*%)"

# Fields where regex is trusted over the model (deterministic / closed sets).
_REGEX_PRIMARY = {
    "date", "issue_date", "start_date", "end_date", "delivery_date", "due_date",
    "birth_date", "time", "total", "subtotal", "vat", "vat_rate",
    "invoice_number", "receipt_number", "order_number", "customer_number",
    "delivery_number", "currency", "email", "phone", "fax", "website",
    "iban", "bic", "vat_id", "company_register_number", "insurance_number",
    "payment_method", "insurer", "bank_name",
}

# Address-style fields: choose the longer of NER vs regex (most complete).
_ADDRESS_FIELDS = {"address", "customer_address", "doctor_address"}

# Fields whose captured value is a date (spaces normalised out).
_DATE_FIELDS = {
    "date", "issue_date", "start_date", "end_date", "delivery_date",
    "due_date", "birth_date",
}

_FLAGS = re.IGNORECASE | re.DOTALL


def _re(pat: str) -> re.Pattern[str]:
    return re.compile(pat, _FLAGS)


# Each value is an ordered list of patterns; first match wins. The LAST captured
# group is taken as the value.
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "date": [
        _re(rf"(?:rechnungsdatum|belegdatum|datum)\s*[:/]?\s*{_DATE}"),
    ],
    "issue_date": [
        _re(rf"ausstellungsdatum\s*[:/]?\s*{_DATE}"),
        _re(rf"ausgestellt\s*am\s*[:/]?\s*{_DATE}"),
    ],
    "start_date": [
        _re(rf"arbeitsunf[äa]hig\s*von\s*[:/]?\s*{_DATE}"),
        _re(rf"pflege(?:freistellung)?\s*(?:notwendig\s*)?von\s*[:/]?\s*{_DATE}"),
        _re(rf"(?:^|\n)\s*von\s+{_DATE}"),
        _re(rf"\bab\s+{_DATE}"),
    ],
    "end_date": [
        _re(rf"letzter\s*tag\s*der\s*.{{0,40}}?arbeitsunf[äa]higkeit\s*[:/]?\s*{_DATE}"),
        _re(rf"\bbis\s+{_DATE}"),
        _re(rf"voraussichtliches?\s*ende\s*.{{0,40}}?{_DATE}"),
    ],
    "delivery_date": [
        _re(rf"lieferdatum\s*(?:/\s*date)?\s*[:/]?\s*{_DATE}"),
    ],
    "due_date": [
        _re(rf"(?:zahlbar\s*bis|f[äa]llig(?:keitsdatum)?(?:\s*am)?|zahlungsziel)\s*[:/]?\s*{_DATE}"),
    ],
    "birth_date": [
        _re(rf"(?:geburtsdatum|geboren\s*am|geb\.?)\s*[:/]?\s*{_DATE}"),
    ],
    "time": [
        _re(r"(?:uhrzeit|zeit)\s*[:.]?\s*(\d{1,2}[:.]\d{2})"),
    ],
    "total": [
        _re(rf"(?:gesamtbetrag|rechnungsbetrag|zu\s*zahlen)\s*[:.]?\s*(?:EUR\s*)?{_AMOUNT}"),
        _re(rf"summe\s+EUR\s+{_AMOUNT}"),
        _re(rf"(?:gesamt|summe|total|bezahlt|gegeben)\s*[:.]?\s*(?:EUR\s*)?{_AMOUNT}"),
    ],
    "subtotal": [
        _re(rf"(?:zwischensumme|nettobetrag|netto|entgelt)\s*[:.]?\s*(?:EUR\s*)?{_AMOUNT}"),
    ],
    "vat": [
        # Austrian receipt line "B: 10% MwSt von 8.04 = 0.80" -> take the result.
        _re(rf"(?:mwst|ust|umsatzsteuer)\s*(?:\d{{1,2}}\s*%)?\s*von\s+{_AMOUNT}\s*=\s*{_AMOUNT}"),
        # "MwSt 20%: 73,90" / "MwSt 0,80" / "USt 20% 474,48" / "MwSt-Betrag 73,90"
        _re(rf"(?:mwst|ust|umsatzsteuer|vat)(?:-?\s*betrag)?\s*(?:\d{{1,2}}\s*%)?\s*[:.=]?\s*(?:EUR\s*)?{_AMOUNT}"),
    ],
    "vat_rate": [
        _re(rf"(?:steuersatz|mwst-?satz|ust-?satz)\s*[:.]?\s*{_RATE}"),
        _re(rf"(?:mwst|ust|umsatzsteuer)\s*[:.]?\s*{_RATE}"),
        _re(rf"{_RATE}\s*(?:mwst|ust)"),
    ],
    "invoice_number": [
        _re(r"(?:rechnungs-?\s*nr|rechnungsnummer|re-?nr)\.?\s*[:.]?\s*([A-Za-z0-9][\w./\-]{2,28})"),
    ],
    "receipt_number": [
        _re(r"(?:beleg-?\s*nr|beleg\s*nr|bon-?nr|kassenbon\s*nr)\.?\s*[:.]?\s*([A-Za-z0-9][\w./\-]{2,28})"),
    ],
    "order_number": [
        _re(r"(?:bestellnummer|bestell-?nr|auftragsnummer|auftrags-?nr)\.?\s*[:.]?\s*([A-Za-z0-9][\w./\-]{2,28})"),
    ],
    "customer_number": [
        _re(r"(?:kundennummer|kunden-?nr|kd-?nr)\.?\s*[:.]?\s*([A-Za-z0-9][\w./\-]{2,28})"),
    ],
    "delivery_number": [
        _re(r"(?:lieferschein-?\s*nr|lieferschein\s*nr|lieferschein|ls)\.?\s*[:.#]?\s*([A-Za-z0-9][\w./\-]{2,24})"),
    ],
    "currency": [_re(r"\b(EUR|USD|GBP|CHF|€)\b")],
    "payment_method": [
        _re(r"\b(MasterCard|Maestro|Bankomat|Kreditkarte|EC-Karte|Überweisung|PayLife|Visa|Bar)\b"),
    ],
    "email": [_re(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")],
    "phone": [
        _re(r"(?:tel\.?|telefon|fon)\s*[:.]?\s*((?:\+?\d[\d\s./\-]{6,20}\d))"),
    ],
    "fax": [
        _re(r"(?:fax|telefax)\s*[:.]?\s*((?:\+?\d[\d\s./\-]{6,20}\d))"),
    ],
    "website": [
        _re(r"\b((?:https?://)?www\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"),
    ],
    "iban": [_re(r"\b(AT\d{2}(?:\s?\d{4}){4}|AT\d{18}|DE\d{20})\b")],
    "bic": [
        _re(r"(?:bic|swift)\s*[:.]?\s*([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)"),
        _re(r"\b([A-Z]{4}AT[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b"),
    ],
    "bank_name": [
        _re(r"\b(Raiffeisenlandesbank|Raiffeisenbank|Erste\s*Bank|BAWAG\s*P\.?S\.?K\.?|Bank\s*Austria|Sparkasse|Volksbank|Oberbank|Hypo\s*Tirol\s*Bank)\b"),
        _re(r"(?:bank|kreditinstitut)\s*[:.]?\s*([A-ZÄÖÜ][\wÄÖÜäöüß.\- ]{2,28}?)(?:\n|$)"),
    ],
    "vat_id": [
        _re(r"\b(ATU\s?\d{8})\b"),
        _re(r"(?:uid(?:-?nr)?|ust-?id)\.?\s*[:.]?\s*(ATU?\s?\d{6,9})"),
    ],
    "company_register_number": [
        _re(r"(?:firmenbuchnummer|firmenbuch-?nr)\.?\s*[:.]?\s*(FN\s?\d{4,7}\s?[a-z]?)"),
        _re(r"\b(FN\s?\d{4,7}\s?[a-z]?)\b"),
    ],
    "insurance_number": [
        _re(r"(?:versicherungsnummer|sv-?nr|svnr)\s*[:.]?\s*\n*\s*(\d{4}\s*\d{2}\s*\d{2}\s*\d{2})"),
        _re(r"\b(\d{4}\s\d{2}\s\d{2}\s\d{2})\b"),
    ],
    "insurer": [
        _re(r"(?:versicherungstr[äa]ger|kasse)\s*[:.]?\s*((?:ÖGK|SVS|BVAEB|KFA)(?:\s+(?:Wien|Nieder[öo]sterreich|Ober[öo]sterreich|Steiermark|Tirol|K[äa]rnten|Salzburg|Vorarlberg|Burgenland))?)"),
        _re(r"\b((?:ÖGK|SVS|BVAEB|KFA)(?:\s+(?:Wien|Nieder[öo]sterreich|Ober[öo]sterreich|Steiermark|Tirol|K[äa]rnten|Salzburg|Vorarlberg|Burgenland))?)\b"),
    ],
    "patient": [
        _re(r"familienname,?\s*vorname\(?n?\)?\s*[:]?\s*\n*\s*([A-ZÄÖÜ][\wäöüß]+\s+[A-ZÄÖÜ][\wäöüß]+)"),
        _re(r"(?:patient(?:/in|in)?|name\s*der/des\s*erkrankten|erkrankte\s*person)\s*[:]?\s*([A-ZÄÖÜ][\wäöüß]+\s+[A-ZÄÖÜ][\wäöüß]+)"),
    ],
    "doctor": [
        _re(r"((?:Univ\.-Prof\.\s*)?Dr\.?(?:\s*med\.?)?\s+[A-ZÄÖÜ][\wäöüß]+(?:\s*&\s*Dr\.?\s+[A-ZÄÖÜ][\wäöüß]+)?)"),
    ],
    "diagnosis": [
        _re(r"(?:diagnose|befund|grund)\s*[:]?\s*([A-ZÄÖÜ][\wäöüß/\- ]{2,40}?)(?:\n|$)"),
        _re(r"\b(Krankheit/Unfall|Berufskrankheit|Krankheit|Unfall)\b"),
    ],
    "company": [
        _re(r"([A-ZÄÖÜ][\wÄÖÜäöüß.\-]*(?:\s+[A-ZÄÖÜ&][\wÄÖÜäöüß.\-]*){0,3}\s+(?:AG|GmbH|GesmbH|KG|OG|e\.U\.|GmbH\s*&\s*Co\s*KG))"),
    ],
    "customer": [
        _re(r"(?:empf[äa]nger|kunde|rechnungsempf[äa]nger)\s*[:]?\s*\n*\s*([A-ZÄÖÜ][\wÄÖÜäöüß.&\-]+(?:\s+[A-ZÄÖÜ&][\wÄÖÜäöüß.\-]+){0,3})"),
    ],
    "contact_person": [
        _re(r"(?:ansprechpartner|kontaktperson|kontakt)\s*[:]?\s*([A-ZÄÖÜ][\wäöüß]+\s+[A-ZÄÖÜ][\wäöüß]+)"),
    ],
    "address": [
        _re(r"\b(\d{4}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+)*\s+\d{1,4})"),
        _re(r"\b(\d{4}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\- ]{3,40})"),
    ],
    "customer_address": [
        _re(r"\b(\d{4}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+)*\s+\d{1,4})"),
    ],
    "doctor_address": [
        _re(r"\b(\d{4}\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.\-]+)*\s+\d{1,4})"),
    ],
}


def aggregate_bio_spans(
    words: list[str],
    word_tags: list[str],
    word_scores: list[float] | None = None,
) -> list[EntitySpan]:
    if word_scores is None:
        word_scores = [1.0] * len(words)

    spans: list[EntitySpan] = []
    cur_label: str | None = None
    cur_words: list[str] = []
    cur_scores: list[float] = []
    start_idx = 0

    def flush() -> None:
        nonlocal cur_label, cur_words, cur_scores, start_idx
        if cur_label and cur_words:
            spans.append(
                EntitySpan(
                    label=cur_label,
                    text=" ".join(cur_words),
                    score=sum(cur_scores) / len(cur_scores),
                    start=start_idx,
                    end=start_idx + len(cur_words),
                )
            )
        cur_label = None
        cur_words = []
        cur_scores = []

    for i, (word, tag) in enumerate(zip(words, word_tags)):
        score = word_scores[i] if i < len(word_scores) else 1.0
        if not tag or tag == "O":
            flush()
            continue
        if tag.startswith("B-"):
            flush()
            cur_label, cur_words, cur_scores, start_idx = tag[2:], [word], [score], i
        elif tag.startswith("I-"):
            ent = tag[2:]
            if cur_label == ent:
                cur_words.append(word)
                cur_scores.append(score)
            else:
                flush()
                cur_label, cur_words, cur_scores, start_idx = ent, [word], [score], i
    flush()
    return spans


def _best_ner_value(label: str, spans: list[EntitySpan], min_score: float) -> tuple[str, float] | None:
    """Pick the single best span for a label: highest score, then longest."""
    cands = [s for s in spans if s.label == label and s.score >= min_score]
    if not cands:
        return None
    best = max(cands, key=lambda s: (round(s.score, 3), len(s.text)))
    return _clean(best.text), best.score


def _regex_value(field: str, text: str) -> str | None:
    for pat in _PATTERNS.get(field, []):
        m = pat.search(text)
        if m:
            val = m.group(m.lastindex) if m.lastindex else m.group(0)
            val = _clean(val)
            if val:
                return val
    return None


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip(" .,:;-")
    return value


def _normalize_amount(value: str) -> str:
    """1.234,50 -> 1234.50 ; 154,06 -> 154.06 ; keep 108.21 as-is."""
    v = value.replace(" ", "")
    if "," in v:
        v = v.replace(".", "").replace(",", ".")
    return v


def _normalize_date(value: str) -> str:
    """28. 12. 2026 -> 28.12.2026 (strip spaces around separators)."""
    return re.sub(r"\s*([./\-])\s*", r"\1", value).strip()


def decode_entities(
    document_type: str,
    raw_text: str,
    words: list[str],
    word_tags: list[str],
    word_scores: list[float] | None = None,
    fields_by_type: dict[str, list[str]] | None = None,
    min_entity_score: float = 0.35,
    enable_regex_fallback: bool = True,
    display_map: dict[str, str] | None = None,
) -> dict[str, str]:
    fields_by_type = fields_by_type or {}
    allowed = fields_by_type.get(document_type, [])
    spans = aggregate_bio_spans(words, word_tags, word_scores)

    result: dict[str, str] = {}
    for field in allowed:
        regex_val = _regex_value(field, raw_text) if enable_regex_fallback else None
        ner = _best_ner_value(field, spans, min_entity_score)
        ner_val = ner[0] if ner else None

        if field in _ADDRESS_FIELDS:
            # Prefer the more complete address (NER may stop early on line breaks).
            value = max([v for v in (ner_val, regex_val) if v], key=len, default=None)
        elif field in _REGEX_PRIMARY:
            value = regex_val or ner_val
        else:
            value = ner_val or regex_val

        if not value:
            continue
        if field in ("total", "subtotal", "vat"):
            value = _normalize_amount(value)
        elif field in _DATE_FIELDS:
            value = _normalize_date(value)
        elif field == "currency" and value == "€":
            value = "EUR"
        result[field] = value

    return to_display_fields(result, display_map)


def to_display_fields(
    fields: dict[str, str],
    display_map: dict[str, str] | None = None,
) -> dict[str, str]:
    if not display_map:
        return {k: v for k, v in fields.items() if v}
    return {display_map.get(k, k): v for k, v in fields.items() if v}
