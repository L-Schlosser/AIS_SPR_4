"""Tests for data generation scripts — image labels, NER samples, augmentations."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
from PIL import Image

from scripts.generate_samples import (
    _apply_augmentations,
    _generate_one_arztbesuch,
    _generate_one_lieferschein,
    _generate_one_reisekosten,
    generate_arztbesuch,
    generate_lieferschein,
    generate_reisekosten,
)
from scripts.generate_text_samples import (
    generate_arztbesuch_ner,
    generate_lieferschein_ner,
    generate_reisekosten_ner,
)

# ---------------------------------------------------------------------------
# Label files match their JSON schemas
# ---------------------------------------------------------------------------


class TestLabelSchemaValidation:
    """Verify that generated label data validates against JSON schemas."""

    def test_arztbesuch_label_matches_schema(self, all_schemas: dict[str, dict]) -> None:
        schema = all_schemas["arztbesuchsbestaetigung"]
        for _ in range(5):
            _, label = _generate_one_arztbesuch()
            jsonschema.validate(label, schema)

    def test_reisekosten_label_matches_schema(self, all_schemas: dict[str, dict]) -> None:
        schema = all_schemas["reisekostenbeleg"]
        for _ in range(5):
            _, label = _generate_one_reisekosten()
            jsonschema.validate(label, schema)

    def test_lieferschein_label_matches_schema(self, all_schemas: dict[str, dict]) -> None:
        schema = all_schemas["lieferschein"]
        for _ in range(5):
            _, label = _generate_one_lieferschein()
            jsonschema.validate(label, schema)

    def test_generated_files_match_schema(self, tmp_path: Path, all_schemas: dict[str, dict]) -> None:
        """Generate actual files and verify labels on disk match schemas."""
        generate_arztbesuch(tmp_path, count=2)
        generate_reisekosten(tmp_path, count=2)
        generate_lieferschein(tmp_path, count=2)

        for doc_type in ("arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"):
            schema = all_schemas[doc_type]
            label_files = list((tmp_path / doc_type).glob("*_label.json"))
            assert len(label_files) == 2
            for lf in label_files:
                with open(lf, encoding="utf-8") as f:
                    data = json.load(f)
                jsonschema.validate(data, schema)


# ---------------------------------------------------------------------------
# Token/tag length alignment in NER samples
# ---------------------------------------------------------------------------


class TestNERTokenTagAlignment:
    """Ensure tokens and ner_tags lists always have the same length."""

    def test_arztbesuch_alignment(self, sample_ner_data: dict[str, list[dict]]) -> None:
        for sample in sample_ner_data["arztbesuchsbestaetigung"]:
            assert len(sample["tokens"]) == len(sample["ner_tags"]), (
                f"Token count {len(sample['tokens'])} != tag count {len(sample['ner_tags'])}"
            )

    def test_reisekosten_alignment(self, sample_ner_data: dict[str, list[dict]]) -> None:
        for sample in sample_ner_data["reisekostenbeleg"]:
            assert len(sample["tokens"]) == len(sample["ner_tags"])

    def test_lieferschein_alignment(self, sample_ner_data: dict[str, list[dict]]) -> None:
        for sample in sample_ner_data["lieferschein"]:
            assert len(sample["tokens"]) == len(sample["ner_tags"])

    def test_alignment_at_scale(self) -> None:
        """Generate a larger batch and verify alignment across all."""
        for gen in (generate_arztbesuch_ner, generate_reisekosten_ner, generate_lieferschein_ner):
            samples = gen(count=50)
            for s in samples:
                assert len(s["tokens"]) == len(s["ner_tags"])


# ---------------------------------------------------------------------------
# All expected BIO tags appear in the generated data
# ---------------------------------------------------------------------------


ARZTBESUCH_EXPECTED_TAGS = {"O", "B-PATIENT", "I-PATIENT", "B-DOCTOR", "I-DOCTOR", "B-FACILITY", "I-FACILITY",
                            "B-ADDRESS", "I-ADDRESS", "B-DATE", "B-TIME", "B-DURATION"}
REISEKOSTEN_EXPECTED_TAGS = {"O", "B-VENDOR", "I-VENDOR", "B-DATE", "B-AMOUNT", "B-CURRENCY", "B-VAT_RATE",
                             "B-VAT_AMOUNT", "B-CATEGORY", "B-DESC", "I-DESC", "B-RECEIPT_NUM"}
LIEFERSCHEIN_EXPECTED_TAGS = {"O", "B-DELNR", "B-DELDATE", "B-SENDER", "I-SENDER", "B-SADDR", "I-SADDR",
                              "B-RECIP", "I-RECIP", "B-RADDR", "I-RADDR", "B-ORDNR", "B-ITEM_DESC",
                              "B-ITEM_QTY", "B-ITEM_UNIT", "B-WEIGHT"}


class TestExpectedBIOTags:
    """Verify that all expected BIO tag types appear in generated data."""

    def test_arztbesuch_tags_present(self) -> None:
        samples = generate_arztbesuch_ner(count=100)
        all_tags = set()
        for s in samples:
            all_tags.update(s["ner_tags"])
        missing = ARZTBESUCH_EXPECTED_TAGS - all_tags
        assert not missing, f"Missing tags: {missing}"

    def test_reisekosten_tags_present(self) -> None:
        samples = generate_reisekosten_ner(count=100)
        all_tags = set()
        for s in samples:
            all_tags.update(s["ner_tags"])
        missing = REISEKOSTEN_EXPECTED_TAGS - all_tags
        assert not missing, f"Missing tags: {missing}"

    def test_lieferschein_tags_present(self) -> None:
        samples = generate_lieferschein_ner(count=100)
        all_tags = set()
        for s in samples:
            all_tags.update(s["ner_tags"])
        missing = LIEFERSCHEIN_EXPECTED_TAGS - all_tags
        assert not missing, f"Missing tags: {missing}"

    def test_all_tags_are_valid_bio_format(self) -> None:
        """Every tag must be 'O' or start with 'B-' or 'I-'."""
        for gen in (generate_arztbesuch_ner, generate_reisekosten_ner, generate_lieferschein_ner):
            for s in gen(count=20):
                for tag in s["ner_tags"]:
                    assert tag == "O" or tag.startswith("B-") or tag.startswith("I-"), f"Invalid tag: {tag}"


# ---------------------------------------------------------------------------
# Augmentation produces visually different images
# ---------------------------------------------------------------------------


class TestAugmentation:
    """Verify that augmentations modify pixel arrays."""

    def test_augmentation_changes_pixels(self) -> None:
        img, _ = _generate_one_arztbesuch()
        original_arr = np.array(img)
        augmented = _apply_augmentations(img.copy())
        augmented_arr = np.array(augmented)
        assert original_arr.shape == augmented_arr.shape, "Augmentation must not change image dimensions"
        assert not np.array_equal(original_arr, augmented_arr), "Augmented image should differ from original"

    def test_augmentation_preserves_dimensions(self) -> None:
        for gen in (_generate_one_arztbesuch, _generate_one_reisekosten, _generate_one_lieferschein):
            img, _ = gen()
            augmented = _apply_augmentations(img)
            assert augmented.size == img.size

    def test_augmentation_pixel_range(self) -> None:
        img, _ = _generate_one_reisekosten()
        augmented = _apply_augmentations(img)
        arr = np.array(augmented)
        assert arr.min() >= 0 and arr.max() <= 255


# ---------------------------------------------------------------------------
# Faker generates German-locale data
# ---------------------------------------------------------------------------


class TestGermanLocale:
    """Verify that generated data uses German locale patterns."""

    def test_arztbesuch_uses_german_dates(self) -> None:
        _, label = _generate_one_arztbesuch()
        assert label["visit_date"].count("-") == 2, "Date should be ISO format YYYY-MM-DD"
        parts = label["visit_date"].split("-")
        assert len(parts[0]) == 4, "Year should be 4 digits"

    def test_arztbesuch_doctor_has_dr_prefix(self) -> None:
        for _ in range(10):
            _, label = _generate_one_arztbesuch()
            assert label["doctor_name"].startswith("Dr."), "Doctor name should start with 'Dr.'"

    def test_arztbesuch_facility_has_praxis_prefix(self) -> None:
        for _ in range(10):
            _, label = _generate_one_arztbesuch()
            assert label["facility_name"].startswith("Praxis"), "Facility should start with 'Praxis'"

    def test_lieferschein_delivery_note_format(self) -> None:
        _, label = _generate_one_lieferschein()
        assert label["delivery_note_number"].startswith("LS-"), "Delivery note number should start with 'LS-'"

    def test_reisekosten_currency_is_eur(self) -> None:
        for _ in range(10):
            _, label = _generate_one_reisekosten()
            assert label["currency"] == "EUR"

    def test_reisekosten_vat_is_austrian(self) -> None:
        rates_seen: set[float] = set()
        for _ in range(100):
            _, label = _generate_one_reisekosten()
            rates_seen.add(label["vat_rate"])
        assert rates_seen.issubset({10.0, 13.0, 20.0}), f"Unexpected VAT rates: {rates_seen}"


# ---------------------------------------------------------------------------
# NER document_type field is correct
# ---------------------------------------------------------------------------


class TestNERDocumentType:
    """Ensure each NER generator sets the correct document_type."""

    def test_arztbesuch_document_type(self) -> None:
        for s in generate_arztbesuch_ner(count=5):
            assert s["document_type"] == "arztbesuchsbestaetigung"

    def test_reisekosten_document_type(self) -> None:
        for s in generate_reisekosten_ner(count=5):
            assert s["document_type"] == "reisekostenbeleg"

    def test_lieferschein_document_type(self) -> None:
        for s in generate_lieferschein_ner(count=5):
            assert s["document_type"] == "lieferschein"


# ---------------------------------------------------------------------------
# Generated images are valid
# ---------------------------------------------------------------------------


class TestImageGeneration:
    """Verify that generated images are valid PIL Images."""

    def test_arztbesuch_image_dimensions(self, sample_arztbesuch_image: Path) -> None:
        img = Image.open(sample_arztbesuch_image)
        assert img.size == (800, 1130)
        assert img.mode == "RGB"

    def test_reisekosten_image_dimensions(self, sample_reisekosten_image: Path) -> None:
        img = Image.open(sample_reisekosten_image)
        assert img.size == (800, 1130)
        assert img.mode == "RGB"

    def test_lieferschein_image_dimensions(self, sample_lieferschein_image: Path) -> None:
        img = Image.open(sample_lieferschein_image)
        assert img.size == (800, 1130)
        assert img.mode == "RGB"
