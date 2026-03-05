"""Generate synthetic document images with ground-truth labels for training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

fake = Faker("de_DE")

# Canvas dimensions (A4 ratio)
WIDTH = 800
HEIGHT = 1130


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to default if unavailable."""
    names = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    if bold:
        names = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] + names
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont, fill: str = "black",
) -> int:
    """Draw text and return the y position after the text."""
    draw.text((x, y), text, fill=fill, font=font)
    bbox = font.getbbox(text)
    return y + (bbox[3] - bbox[1]) + 4


def _apply_augmentations(image: Image.Image) -> Image.Image:
    """Apply random augmentations: rotation, noise, brightness, blur."""
    # Slight rotation (±3°)
    angle = random.uniform(-3, 3)
    image = image.rotate(angle, fillcolor="white", expand=False)

    arr = np.array(image, dtype=np.float32)

    # Brightness jitter (±15%)
    factor = random.uniform(0.85, 1.15)
    arr = np.clip(arr * factor, 0, 255)

    # Gaussian noise
    noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    image = Image.fromarray(arr.astype(np.uint8))

    # Slight blur
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    return image


# ---------------------------------------------------------------------------
# Arztbesuchsbestätigung
# ---------------------------------------------------------------------------


def _generate_one_arztbesuch() -> tuple[Image.Image, dict]:
    """Generate a single medical visit confirmation document."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    font_title = _get_font(28, bold=True)
    font_label = _get_font(16, bold=True)
    font_value = _get_font(16)
    font_small = _get_font(12)

    patient_name = fake.name()
    doctor_name = f"Dr. {fake.last_name()}"
    facility_name = f"Praxis {fake.last_name()}"
    facility_address = fake.address().replace("\n", ", ")
    visit_date = fake.date_between(start_date="-2y", end_date="today")
    visit_hour = random.randint(8, 17)
    visit_minute = random.choice([0, 15, 30, 45])
    visit_time = f"{visit_hour:02d}:{visit_minute:02d}"
    duration = random.choice([15, 20, 30, 45, 60, 90, 120])

    # Draw document
    y = 40
    title = "BESTÄTIGUNG ARZTBESUCH"
    draw.text(((WIDTH - draw.textlength(title, font=font_title)) / 2, y), title, fill="black", font=font_title)
    y += 50

    # Horizontal line
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=2)
    y += 20

    # Facility header
    y = _draw_text(draw, 60, y, facility_name, font_label)
    y = _draw_text(draw, 60, y, facility_address, font_small)
    y += 15

    # Fields
    fields = [
        ("Patient:", patient_name),
        ("Behandelnder Arzt:", doctor_name),
        ("Datum des Besuchs:", visit_date.strftime("%d.%m.%Y")),
        ("Uhrzeit:", visit_time),
        ("Dauer (Minuten):", str(duration)),
    ]
    for label, value in fields:
        y = _draw_text(draw, 60, y, label, font_label)
        y = _draw_text(draw, 80, y, value, font_value)
        y += 8

    # Footer
    y += 30
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 10
    _draw_text(draw, 60, y, "Diese Bestätigung dient als Nachweis des Arztbesuchs.", font_small)

    label_data = {
        "document_type": "arztbesuchsbestaetigung",
        "patient_name": patient_name,
        "doctor_name": doctor_name,
        "facility_name": facility_name,
        "facility_address": facility_address,
        "visit_date": visit_date.strftime("%Y-%m-%d"),
        "visit_time": visit_time,
        "duration_minutes": duration,
        "confidence": 1.0,
    }

    return img, label_data


def generate_arztbesuch(output_dir: Path, count: int = 250) -> None:
    """Generate arztbesuchsbestaetigung document images."""
    out = output_dir / "arztbesuchsbestaetigung"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, label = _generate_one_arztbesuch()
        if random.random() < 0.5:
            img = _apply_augmentations(img)
        img.save(out / f"arztbesuch_{i:04d}.png")
        with open(out / f"arztbesuch_{i:04d}_label.json", "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Reisekostenbeleg
# ---------------------------------------------------------------------------


def _generate_one_reisekosten() -> tuple[Image.Image, dict]:
    """Generate a single travel expense receipt."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    font_title = _get_font(28, bold=True)
    font_label = _get_font(16, bold=True)
    font_value = _get_font(16)
    font_small = _get_font(12)

    header = random.choice(["RECHNUNG", "QUITTUNG", "BELEG"])
    category = random.choice(["hotel", "restaurant", "transport", "other"])
    vendor_name = fake.company()
    vendor_address = fake.address().replace("\n", ", ")
    date = fake.date_between(start_date="-2y", end_date="today")
    amount = round(random.uniform(5, 500), 2)
    vat_rate = random.choice([10.0, 13.0, 20.0])
    vat_amount = round(amount * vat_rate / (100 + vat_rate), 2)
    net_amount = round(amount - vat_amount, 2)
    receipt_number = f"RE-{random.randint(10000, 99999)}"

    # Category-specific descriptions
    descriptions = {
        "hotel": f"Übernachtung {fake.city()}, {random.randint(1, 5)} Nacht/Nächte",
        "restaurant": f"Geschäftsessen {fake.city()}",
        "transport": random.choice([
            f"Fahrt {fake.city()} - {fake.city()}",
            f"Taxi {fake.city()}", f"Parkgebühren {fake.city()}",
        ]),
        "other": f"Geschäftsausgabe: {fake.bs()}",
    }
    description = descriptions[category]

    # Draw document
    y = 40
    draw.text(((WIDTH - draw.textlength(header, font=font_title)) / 2, y), header, fill="black", font=font_title)
    y += 50

    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=2)
    y += 20

    # Vendor info
    y = _draw_text(draw, 60, y, vendor_name, font_label)
    y = _draw_text(draw, 60, y, vendor_address, font_small)
    y += 10

    # Receipt number and date
    y = _draw_text(draw, 60, y, f"Belegnummer: {receipt_number}", font_value)
    y = _draw_text(draw, 60, y, f"Datum: {date.strftime('%d.%m.%Y')}", font_value)
    y += 15

    # Description
    y = _draw_text(draw, 60, y, "Beschreibung:", font_label)
    y = _draw_text(draw, 80, y, description, font_value)
    y += 15

    # Amount table
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 10
    y = _draw_text(draw, 60, y, f"Nettobetrag:    EUR {net_amount:.2f}", font_value)
    y = _draw_text(draw, 60, y, f"MwSt ({vat_rate:.0f}%):     EUR {vat_amount:.2f}", font_value)
    draw.line([(350, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 5
    y = _draw_text(draw, 60, y, f"Gesamtbetrag:   EUR {amount:.2f}", font_label)

    label_data = {
        "document_type": "reisekostenbeleg",
        "vendor_name": vendor_name,
        "vendor_address": vendor_address,
        "date": date.strftime("%Y-%m-%d"),
        "amount": amount,
        "currency": "EUR",
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "category": category,
        "description": description,
        "receipt_number": receipt_number,
        "confidence": 1.0,
    }

    return img, label_data


def generate_reisekosten(output_dir: Path, count: int = 250) -> None:
    """Generate reisekostenbeleg document images."""
    out = output_dir / "reisekostenbeleg"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, label = _generate_one_reisekosten()
        if random.random() < 0.5:
            img = _apply_augmentations(img)
        img.save(out / f"reisekosten_{i:04d}.png")
        with open(out / f"reisekosten_{i:04d}_label.json", "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Lieferschein
# ---------------------------------------------------------------------------


def _generate_one_lieferschein() -> tuple[Image.Image, dict]:
    """Generate a single delivery note document."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    font_title = _get_font(28, bold=True)
    font_label = _get_font(16, bold=True)
    font_value = _get_font(16)
    font_small = _get_font(12)
    font_table_hdr = _get_font(14, bold=True)
    font_table = _get_font(14)

    delivery_note_number = f"LS-{random.randint(10000, 99999)}"
    delivery_date = fake.date_between(start_date="-2y", end_date="today")
    order_number = f"BE-{random.randint(10000, 99999)}"

    sender_name = fake.company()
    sender_address = fake.address().replace("\n", ", ")
    recipient_name = fake.company()
    recipient_address = fake.address().replace("\n", ", ")

    # Generate items
    num_items = random.randint(1, 6)
    units = ["Stk", "kg", "m", "Packung", "Karton", "Palette", "Liter"]
    item_descs = [
        "Schrauben M8x50", "Stahlblech 2mm", "Kupferrohr 15mm", "Dichtungsring",
        "Hydraulikschlauch", "Kabelbinder 200mm", "LED-Panel 60x60", "Sicherung 16A",
        "Montageschiene", "Winkelverbinder", "Isoliermaterial", "Werkzeugset",
        "Verpackungsmaterial", "Klebstoff Industrial", "Filterpatrone",
    ]
    items = []
    for _ in range(num_items):
        items.append({
            "description": random.choice(item_descs),
            "quantity": round(random.uniform(1, 500), 1),
            "unit": random.choice(units),
        })

    total_weight = f"{random.uniform(5, 500):.1f} kg"

    # Draw document
    y = 40
    title = "LIEFERSCHEIN"
    draw.text(((WIDTH - draw.textlength(title, font=font_title)) / 2, y), title, fill="black", font=font_title)
    y += 50

    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=2)
    y += 20

    # Delivery note number and date
    y = _draw_text(draw, 60, y, f"Lieferschein-Nr.: {delivery_note_number}", font_label)
    y = _draw_text(draw, 60, y, f"Lieferdatum: {delivery_date.strftime('%d.%m.%Y')}", font_value)
    y = _draw_text(draw, 60, y, f"Bestellnummer: {order_number}", font_value)
    y += 15

    # Sender / Recipient side by side
    y = _draw_text(draw, 60, y, "Absender:", font_label)
    y = _draw_text(draw, 80, y, sender_name, font_value)
    y = _draw_text(draw, 80, y, sender_address, font_small)
    y += 10

    y = _draw_text(draw, 60, y, "Empfänger:", font_label)
    y = _draw_text(draw, 80, y, recipient_name, font_value)
    y = _draw_text(draw, 80, y, recipient_address, font_small)
    y += 15

    # Items table header
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 5
    _draw_text(draw, 60, y, "Pos", font_table_hdr)
    _draw_text(draw, 110, y, "Beschreibung", font_table_hdr)
    _draw_text(draw, 450, y, "Menge", font_table_hdr)
    y = _draw_text(draw, 560, y, "Einheit", font_table_hdr)
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 5

    # Items
    for idx, item in enumerate(items, 1):
        _draw_text(draw, 60, y, str(idx), font_table)
        _draw_text(draw, 110, y, item["description"], font_table)
        _draw_text(draw, 450, y, f"{item['quantity']:.1f}", font_table)
        y = _draw_text(draw, 560, y, item["unit"], font_table)
        y += 2

    # Total weight
    y += 10
    draw.line([(50, y), (WIDTH - 50, y)], fill="gray", width=1)
    y += 10
    _draw_text(draw, 60, y, f"Gesamtgewicht: {total_weight}", font_label)

    label_data = {
        "document_type": "lieferschein",
        "delivery_note_number": delivery_note_number,
        "delivery_date": delivery_date.strftime("%Y-%m-%d"),
        "sender": {"name": sender_name, "address": sender_address},
        "recipient": {"name": recipient_name, "address": recipient_address},
        "order_number": order_number,
        "items": items,
        "total_weight": total_weight,
        "confidence": 1.0,
    }

    return img, label_data


def generate_lieferschein(output_dir: Path, count: int = 250) -> None:
    """Generate lieferschein document images."""
    out = output_dir / "lieferschein"
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, label = _generate_one_lieferschein()
        if random.random() < 0.5:
            img = _apply_augmentations(img)
        img.save(out / f"lieferschein_{i:04d}.png")
        with open(out / f"lieferschein_{i:04d}_label.json", "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate synthetic document images for all three document types."""
    parser = argparse.ArgumentParser(description="Generate synthetic document images for training.")
    parser.add_argument("--count", type=int, default=250, help="Number of images per document type (default: 250)")
    parser.add_argument(
        "--output-dir", type=str, default="data/samples", help="Output directory (default: data/samples)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    count = args.count

    print(f"Generating {count} images per document type in {output_dir}/")

    print("  Arztbesuchsbestätigung... ", end="", flush=True)
    generate_arztbesuch(output_dir, count)
    print("done")

    print("  Reisekostenbeleg... ", end="", flush=True)
    generate_reisekosten(output_dir, count)
    print("done")

    print("  Lieferschein... ", end="", flush=True)
    generate_lieferschein(output_dir, count)
    print("done")

    total = count * 3
    print(f"\nGenerated {total} images + {total} label files.")


if __name__ == "__main__":
    main()
