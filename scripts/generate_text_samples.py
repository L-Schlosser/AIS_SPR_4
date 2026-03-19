"""Generate BIO-tagged text samples simulating OCR output for NER training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from faker import Faker

fake = Faker("de_DE")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _tokenize_and_tag(text: str, tag: str) -> tuple[list[str], list[str]]:
    """Split text into tokens and assign BIO tags.

    The first token gets B-{tag}, subsequent tokens get I-{tag}.
    """
    tokens = text.split()
    if not tokens:
        return [], []
    tags = [f"B-{tag}"] + [f"I-{tag}"] * (len(tokens) - 1)
    return tokens, tags


def _add_o_tokens(tokens: list[str], tags: list[str], words: list[str]) -> None:
    """Append words tagged as O (outside) to the token/tag lists."""
    tokens.extend(words)
    tags.extend(["O"] * len(words))


def _german_date_str(date_obj) -> str:
    """Format a date as DD.MM.YYYY (German convention)."""
    return date_obj.strftime("%d.%m.%Y")


# ---------------------------------------------------------------------------
# Arztbesuchsbestätigung NER generator
# ---------------------------------------------------------------------------


def _generate_one_arztbesuch_ner() -> dict:
    """Generate a single BIO-tagged Arztbesuch OCR text sample."""
    tokens: list[str] = []
    tags: list[str] = []

    patient_name = fake.name()
    doctor_name = f"Dr. {fake.last_name()}"
    facility_name = f"Praxis {fake.last_name()}"
    facility_address = fake.address().replace("\n", ", ")
    visit_date = fake.date_between(start_date="-2y", end_date="today")
    visit_hour = random.randint(8, 17)
    visit_minute = random.choice([0, 15, 30, 45])
    visit_time = f"{visit_hour:02d}:{visit_minute:02d}"
    duration = str(random.choice([15, 20, 30, 45, 60, 90, 120]))

    # Vary the document layout/order to simulate different OCR outputs
    variant = random.choice(["standard", "compact", "verbose", "reordered"])

    if variant == "standard":
        _add_o_tokens(tokens, tags, ["Bestätigung", "Arztbesuch"])

        t, tg = _tokenize_and_tag(facility_name, "FACILITY")
        _add_o_tokens(tokens, tags, random.choice([[], ["Praxis:"], ["Einrichtung:"]]))
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(facility_address, "ADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Patient:"])
        t, tg = _tokenize_and_tag(patient_name, "PATIENT")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Behandelnder", "Arzt:"])
        t, tg = _tokenize_and_tag(doctor_name, "DOCTOR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Datum", "des", "Besuchs:"])
        t, tg = _tokenize_and_tag(_german_date_str(visit_date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Uhrzeit:"])
        t, tg = _tokenize_and_tag(visit_time, "TIME")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Dauer", "(Minuten):"])
        t, tg = _tokenize_and_tag(duration, "DURATION")
        tokens.extend(t)
        tags.extend(tg)

    elif variant == "compact":
        _add_o_tokens(tokens, tags, ["BESTÄTIGUNG", "ARZTBESUCH"])

        t, tg = _tokenize_and_tag(facility_name, "FACILITY")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(facility_address, "ADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Pat.:"])
        t, tg = _tokenize_and_tag(patient_name, "PATIENT")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Arzt:"])
        t, tg = _tokenize_and_tag(doctor_name, "DOCTOR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Dat.:"])
        t, tg = _tokenize_and_tag(_german_date_str(visit_date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(visit_time, "TIME")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(duration, "DURATION")
        _add_o_tokens(tokens, tags, ["Min."])
        tokens.extend(t)
        tags.extend(tg)

    elif variant == "verbose":
        _add_o_tokens(tokens, tags, ["Bestätigung", "über", "einen", "Arztbesuch"])
        _add_o_tokens(tokens, tags, ["Hiermit", "wird", "bestätigt,", "dass"])

        t, tg = _tokenize_and_tag(patient_name, "PATIENT")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["am"])
        t, tg = _tokenize_and_tag(_german_date_str(visit_date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["um"])
        t, tg = _tokenize_and_tag(visit_time, "TIME")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Uhr", "in", "der"])
        t, tg = _tokenize_and_tag(facility_name, "FACILITY")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(facility_address, "ADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["bei"])
        t, tg = _tokenize_and_tag(doctor_name, "DOCTOR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["vorstellig", "war.", "Dauer:"])
        t, tg = _tokenize_and_tag(duration, "DURATION")
        tokens.extend(t)
        tags.extend(tg)
        _add_o_tokens(tokens, tags, ["Minuten."])

    else:  # reordered
        _add_o_tokens(tokens, tags, ["Arztbesuch", "-", "Bestätigung"])

        _add_o_tokens(tokens, tags, ["Arzt:"])
        t, tg = _tokenize_and_tag(doctor_name, "DOCTOR")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(facility_name, "FACILITY")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(facility_address, "ADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Patient/in:"])
        t, tg = _tokenize_and_tag(patient_name, "PATIENT")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Besuchsdatum:"])
        t, tg = _tokenize_and_tag(_german_date_str(visit_date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Zeit:"])
        t, tg = _tokenize_and_tag(visit_time, "TIME")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Behandlungsdauer:"])
        t, tg = _tokenize_and_tag(duration, "DURATION")
        tokens.extend(t)
        tags.extend(tg)
        _add_o_tokens(tokens, tags, ["Min"])

    # Optionally add trailing noise tokens (simulating OCR artifacts)
    if random.random() < 0.3:
        noise = random.choice([
            ["Diese", "Bestätigung", "dient", "als", "Nachweis."],
            ["Stempel", "Unterschrift"],
            ["Seite", "1", "von", "1"],
        ])
        _add_o_tokens(tokens, tags, noise)

    return {"tokens": tokens, "ner_tags": tags, "document_type": "arztbesuchsbestaetigung"}


# ---------------------------------------------------------------------------
# Reisekostenbeleg NER generator
# ---------------------------------------------------------------------------


def _generate_one_reisekosten_ner() -> dict:
    """Generate a single BIO-tagged Reisekostenbeleg OCR text sample."""
    tokens: list[str] = []
    tags: list[str] = []

    vendor_name = fake.company()
    vendor_address = fake.address().replace("\n", ", ")
    date = fake.date_between(start_date="-2y", end_date="today")
    amount = f"{random.uniform(5, 500):.2f}"
    currency = "EUR"
    vat_rate_num = random.choice([10.0, 13.0, 20.0])
    vat_rate = f"{vat_rate_num:.0f}%"
    vat_amount = f"{float(amount) * vat_rate_num / (100 + vat_rate_num):.2f}"
    category = random.choice(["hotel", "restaurant", "transport", "other"])
    receipt_number = f"RE-{random.randint(10000, 99999)}"

    descriptions = {
        "hotel": f"Übernachtung {fake.city()}",
        "restaurant": f"Geschäftsessen {fake.city()}",
        "transport": f"Fahrt {fake.city()} - {fake.city()}",
        "other": f"Geschäftsausgabe {fake.word()}",
    }
    description = descriptions[category]

    variant = random.choice(["standard", "compact", "verbose", "reordered"])

    if variant == "standard":
        header = random.choice(["RECHNUNG", "QUITTUNG", "BELEG"])
        _add_o_tokens(tokens, tags, [header])

        t, tg = _tokenize_and_tag(vendor_name, "VENDOR")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(vendor_address, "VADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Belegnummer:"])
        t, tg = _tokenize_and_tag(receipt_number, "RECEIPT_NUM")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Datum:"])
        t, tg = _tokenize_and_tag(_german_date_str(date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Beschreibung:"])
        t, tg = _tokenize_and_tag(description, "DESC")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Kategorie:"])
        t, tg = _tokenize_and_tag(category, "CATEGORY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Nettobetrag:"])
        net = f"{float(amount) - float(vat_amount):.2f}"
        _add_o_tokens(tokens, tags, [net, currency])

        _add_o_tokens(tokens, tags, ["MwSt"])
        t, tg = _tokenize_and_tag(vat_rate, "VAT_RATE")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(vat_amount, "VAT_AMOUNT")
        tokens.extend(t)
        tags.extend(tg)
        _add_o_tokens(tokens, tags, [currency])

        _add_o_tokens(tokens, tags, ["Gesamtbetrag:"])
        t, tg = _tokenize_and_tag(amount, "AMOUNT")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(currency, "CURRENCY")
        tokens.extend(t)
        tags.extend(tg)

    elif variant == "compact":
        _add_o_tokens(tokens, tags, [random.choice(["Rechnung", "Quittung", "Beleg"])])

        t, tg = _tokenize_and_tag(vendor_name, "VENDOR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Nr.:"])
        t, tg = _tokenize_and_tag(receipt_number, "RECEIPT_NUM")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(_german_date_str(date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(description, "DESC")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Kat.:"])
        t, tg = _tokenize_and_tag(category, "CATEGORY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Betrag:"])
        t, tg = _tokenize_and_tag(amount, "AMOUNT")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(currency, "CURRENCY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, [f"({vat_rate}"])
        t, tg = _tokenize_and_tag(vat_rate, "VAT_RATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["MwSt:"])
        t, tg = _tokenize_and_tag(vat_amount, "VAT_AMOUNT")
        tokens.extend(t)
        tags.extend(tg)
        _add_o_tokens(tokens, tags, [")"])

    elif variant == "verbose":
        _add_o_tokens(tokens, tags, ["Reisekostenbeleg"])

        t, tg = _tokenize_and_tag(vendor_name, "VENDOR")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(vendor_address, "VADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Rechnungsnummer:"])
        t, tg = _tokenize_and_tag(receipt_number, "RECEIPT_NUM")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Rechnungsdatum:"])
        t, tg = _tokenize_and_tag(_german_date_str(date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Leistungsbeschreibung:"])
        t, tg = _tokenize_and_tag(description, "DESC")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Kategorie:"])
        t, tg = _tokenize_and_tag(category, "CATEGORY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Gesamtbetrag:"])
        t, tg = _tokenize_and_tag(amount, "AMOUNT")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(currency, "CURRENCY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Mehrwertsteuer:"])
        t, tg = _tokenize_and_tag(vat_rate, "VAT_RATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["MwSt-Betrag:"])
        t, tg = _tokenize_and_tag(vat_amount, "VAT_AMOUNT")
        tokens.extend(t)
        tags.extend(tg)
        _add_o_tokens(tokens, tags, [currency])

    else:  # reordered
        t, tg = _tokenize_and_tag(_german_date_str(date), "DATE")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(vendor_name, "VENDOR")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(vendor_address, "VADDRESS")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Beleg-Nr:"])
        t, tg = _tokenize_and_tag(receipt_number, "RECEIPT_NUM")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(description, "DESC")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(category, "CATEGORY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Summe:"])
        t, tg = _tokenize_and_tag(amount, "AMOUNT")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(currency, "CURRENCY")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["inkl."])
        t, tg = _tokenize_and_tag(vat_rate, "VAT_RATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["MwSt"])
        t, tg = _tokenize_and_tag(vat_amount, "VAT_AMOUNT")
        tokens.extend(t)
        tags.extend(tg)

    # Trailing noise
    if random.random() < 0.3:
        noise = random.choice([
            ["Vielen", "Dank", "für", "Ihren", "Besuch!"],
            ["Bitte", "aufbewahren."],
            ["MwSt-Nr:", fake.bothify("ATU########")],
        ])
        _add_o_tokens(tokens, tags, noise)

    return {"tokens": tokens, "ner_tags": tags, "document_type": "reisekostenbeleg"}


# ---------------------------------------------------------------------------
# Lieferschein NER generator
# ---------------------------------------------------------------------------


def _generate_one_lieferschein_ner() -> dict:
    """Generate a single BIO-tagged Lieferschein OCR text sample."""
    tokens: list[str] = []
    tags: list[str] = []

    delivery_note_number = f"LS-{random.randint(10000, 99999)}"
    delivery_date = fake.date_between(start_date="-2y", end_date="today")
    order_number = f"BE-{random.randint(10000, 99999)}"
    sender_name = fake.company()
    sender_address = fake.address().replace("\n", ", ")
    recipient_name = fake.company()
    recipient_address = fake.address().replace("\n", ", ")
    total_weight = f"{random.uniform(5, 500):.1f} kg"

    # Generate items
    units = ["Stk", "kg", "m", "Packung", "Karton", "Palette", "Liter"]
    item_descs = [
        "Schrauben M8x50", "Stahlblech 2mm", "Kupferrohr 15mm", "Dichtungsring",
        "Hydraulikschlauch", "Kabelbinder 200mm", "LED-Panel 60x60", "Sicherung 16A",
        "Montageschiene", "Winkelverbinder", "Isoliermaterial", "Werkzeugset",
    ]
    num_items = random.randint(1, 4)

    variant = random.choice(["standard", "compact", "verbose", "reordered"])

    if variant == "standard":
        _add_o_tokens(tokens, tags, ["LIEFERSCHEIN"])

        _add_o_tokens(tokens, tags, ["Lieferschein-Nr.:"])
        t, tg = _tokenize_and_tag(delivery_note_number, "DELNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Lieferdatum:"])
        t, tg = _tokenize_and_tag(_german_date_str(delivery_date), "DELDATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Bestellnummer:"])
        t, tg = _tokenize_and_tag(order_number, "ORDNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Absender:"])
        t, tg = _tokenize_and_tag(sender_name, "SENDER")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(sender_address, "SADDR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Empfänger:"])
        t, tg = _tokenize_and_tag(recipient_name, "RECIP")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(recipient_address, "RADDR")
        tokens.extend(t)
        tags.extend(tg)

        # Items
        _add_o_tokens(tokens, tags, ["Pos", "Beschreibung", "Menge", "Einheit"])
        for i in range(num_items):
            _add_o_tokens(tokens, tags, [str(i + 1)])
            desc = random.choice(item_descs)
            t, tg = _tokenize_and_tag(desc, "ITEM_DESC")
            tokens.extend(t)
            tags.extend(tg)

            qty = f"{random.uniform(1, 500):.1f}"
            t, tg = _tokenize_and_tag(qty, "ITEM_QTY")
            tokens.extend(t)
            tags.extend(tg)

            unit = random.choice(units)
            t, tg = _tokenize_and_tag(unit, "ITEM_UNIT")
            tokens.extend(t)
            tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Gesamtgewicht:"])
        t, tg = _tokenize_and_tag(total_weight, "WEIGHT")
        tokens.extend(t)
        tags.extend(tg)

    elif variant == "compact":
        _add_o_tokens(tokens, tags, ["Lieferschein"])

        _add_o_tokens(tokens, tags, ["Nr:"])
        t, tg = _tokenize_and_tag(delivery_note_number, "DELNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Dat:"])
        t, tg = _tokenize_and_tag(_german_date_str(delivery_date), "DELDATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Best-Nr:"])
        t, tg = _tokenize_and_tag(order_number, "ORDNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Von:"])
        t, tg = _tokenize_and_tag(sender_name, "SENDER")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(sender_address, "SADDR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["An:"])
        t, tg = _tokenize_and_tag(recipient_name, "RECIP")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(recipient_address, "RADDR")
        tokens.extend(t)
        tags.extend(tg)

        for _ in range(num_items):
            desc = random.choice(item_descs)
            t, tg = _tokenize_and_tag(desc, "ITEM_DESC")
            tokens.extend(t)
            tags.extend(tg)

            qty = f"{random.uniform(1, 500):.1f}"
            t, tg = _tokenize_and_tag(qty, "ITEM_QTY")
            tokens.extend(t)
            tags.extend(tg)

            unit = random.choice(units)
            t, tg = _tokenize_and_tag(unit, "ITEM_UNIT")
            tokens.extend(t)
            tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Gewicht:"])
        t, tg = _tokenize_and_tag(total_weight, "WEIGHT")
        tokens.extend(t)
        tags.extend(tg)

    elif variant == "verbose":
        _add_o_tokens(tokens, tags, ["Lieferschein", "/", "Warenbegleitschein"])

        _add_o_tokens(tokens, tags, ["Lieferscheinnummer:"])
        t, tg = _tokenize_and_tag(delivery_note_number, "DELNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Datum", "der", "Lieferung:"])
        t, tg = _tokenize_and_tag(_german_date_str(delivery_date), "DELDATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Bezug", "auf", "Bestellnummer:"])
        t, tg = _tokenize_and_tag(order_number, "ORDNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Versender", "/", "Absender:"])
        t, tg = _tokenize_and_tag(sender_name, "SENDER")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Adresse:"])
        t, tg = _tokenize_and_tag(sender_address, "SADDR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Warenempfänger:"])
        t, tg = _tokenize_and_tag(recipient_name, "RECIP")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Lieferadresse:"])
        t, tg = _tokenize_and_tag(recipient_address, "RADDR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Warenliste:"])
        for i in range(num_items):
            _add_o_tokens(tokens, tags, ["Position", f"{i + 1}:"])
            desc = random.choice(item_descs)
            t, tg = _tokenize_and_tag(desc, "ITEM_DESC")
            tokens.extend(t)
            tags.extend(tg)

            _add_o_tokens(tokens, tags, ["Anzahl:"])
            qty = f"{random.uniform(1, 500):.1f}"
            t, tg = _tokenize_and_tag(qty, "ITEM_QTY")
            tokens.extend(t)
            tags.extend(tg)

            unit = random.choice(units)
            t, tg = _tokenize_and_tag(unit, "ITEM_UNIT")
            tokens.extend(t)
            tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Gesamtgewicht", "der", "Sendung:"])
        t, tg = _tokenize_and_tag(total_weight, "WEIGHT")
        tokens.extend(t)
        tags.extend(tg)

    else:  # reordered
        t, tg = _tokenize_and_tag(sender_name, "SENDER")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(sender_address, "SADDR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["LIEFERSCHEIN"])

        _add_o_tokens(tokens, tags, ["LS-Nr:"])
        t, tg = _tokenize_and_tag(delivery_note_number, "DELNR")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(_german_date_str(delivery_date), "DELDATE")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Bestell-Nr:"])
        t, tg = _tokenize_and_tag(order_number, "ORDNR")
        tokens.extend(t)
        tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Lieferung", "an:"])
        t, tg = _tokenize_and_tag(recipient_name, "RECIP")
        tokens.extend(t)
        tags.extend(tg)

        t, tg = _tokenize_and_tag(recipient_address, "RADDR")
        tokens.extend(t)
        tags.extend(tg)

        for _ in range(num_items):
            desc = random.choice(item_descs)
            t, tg = _tokenize_and_tag(desc, "ITEM_DESC")
            tokens.extend(t)
            tags.extend(tg)

            qty = f"{random.uniform(1, 500):.1f}"
            t, tg = _tokenize_and_tag(qty, "ITEM_QTY")
            tokens.extend(t)
            tags.extend(tg)

            unit = random.choice(units)
            t, tg = _tokenize_and_tag(unit, "ITEM_UNIT")
            tokens.extend(t)
            tags.extend(tg)

        _add_o_tokens(tokens, tags, ["Gew.:"])
        t, tg = _tokenize_and_tag(total_weight, "WEIGHT")
        tokens.extend(t)
        tags.extend(tg)

    # Trailing noise
    if random.random() < 0.3:
        noise = random.choice([
            ["Ware", "erhalten", "am:", _german_date_str(delivery_date)],
            ["Unterschrift", "Empfänger"],
            ["Seite", "1/1"],
        ])
        _add_o_tokens(tokens, tags, noise)

    return {"tokens": tokens, "ner_tags": tags, "document_type": "lieferschein"}


# ---------------------------------------------------------------------------
# Generation orchestrators
# ---------------------------------------------------------------------------


def generate_arztbesuch_ner(count: int = 1500) -> list[dict]:
    """Generate BIO-tagged text samples for Arztbesuchsbestätigung."""
    return [_generate_one_arztbesuch_ner() for _ in range(count)]


def generate_reisekosten_ner(count: int = 1500) -> list[dict]:
    """Generate BIO-tagged text samples for Reisekostenbeleg."""
    return [_generate_one_reisekosten_ner() for _ in range(count)]


def generate_lieferschein_ner(count: int = 1500) -> list[dict]:
    """Generate BIO-tagged text samples for Lieferschein."""
    return [_generate_one_lieferschein_ner() for _ in range(count)]


def _write_jsonl(samples: list[dict], path: Path) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _split_train_val(samples: list[dict], val_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split samples into train and validation sets (80/20)."""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]


def main() -> None:
    """Generate NER text samples for all document types."""
    parser = argparse.ArgumentParser(description="Generate BIO-tagged text samples for NER training.")
    parser.add_argument("--count", type=int, default=1500, help="Number of samples per document type (default: 1500)")
    parser.add_argument(
        "--output-dir", type=str, default="data/samples", help="Output directory (default: data/samples)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    count = args.count

    generators = {
        "arztbesuchsbestaetigung": generate_arztbesuch_ner,
        "reisekostenbeleg": generate_reisekosten_ner,
        "lieferschein": generate_lieferschein_ner,
    }

    for doc_type, generator in generators.items():
        print(f"  Generating {count} {doc_type} NER samples... ", end="", flush=True)
        samples = generator(count)
        train, val = _split_train_val(samples)

        _write_jsonl(train, output_dir / f"{doc_type}_ner_train.jsonl")
        _write_jsonl(val, output_dir / f"{doc_type}_ner_val.jsonl")
        print(f"done ({len(train)} train, {len(val)} val)")

    print(f"\nGenerated {count * 3} total NER samples across {len(generators)} document types.")


if __name__ == "__main__":
    main()
