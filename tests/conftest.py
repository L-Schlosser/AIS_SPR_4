"""Shared test fixtures for the edge-doc test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_samples import _generate_one_arztbesuch, _generate_one_lieferschein, _generate_one_reisekosten
from scripts.generate_text_samples import (
    generate_arztbesuch_ner,
    generate_lieferschein_ner,
    generate_reisekosten_ner,
)

SCHEMAS_DIR = Path("data/schemas")


@pytest.fixture()
def sample_arztbesuch_image(tmp_path: Path) -> Path:
    """Generate a single arztbesuchsbestaetigung image in a temp directory and return its path."""
    img, label = _generate_one_arztbesuch()
    img_path = tmp_path / "arztbesuch_0000.png"
    img.save(img_path)
    label_path = tmp_path / "arztbesuch_0000_label.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    return img_path


@pytest.fixture()
def sample_reisekosten_image(tmp_path: Path) -> Path:
    """Generate a single reisekostenbeleg image in a temp directory and return its path."""
    img, label = _generate_one_reisekosten()
    img_path = tmp_path / "reisekosten_0000.png"
    img.save(img_path)
    label_path = tmp_path / "reisekosten_0000_label.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    return img_path


@pytest.fixture()
def sample_lieferschein_image(tmp_path: Path) -> Path:
    """Generate a single lieferschein image in a temp directory and return its path."""
    img, label = _generate_one_lieferschein()
    img_path = tmp_path / "lieferschein_0000.png"
    img.save(img_path)
    label_path = tmp_path / "lieferschein_0000_label.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    return img_path


@pytest.fixture()
def sample_ner_data() -> dict[str, list[dict]]:
    """Generate 5 NER samples in memory for each document type."""
    return {
        "arztbesuchsbestaetigung": generate_arztbesuch_ner(count=5),
        "reisekostenbeleg": generate_reisekosten_ner(count=5),
        "lieferschein": generate_lieferschein_ner(count=5),
    }


@pytest.fixture()
def schemas_dir() -> Path:
    """Return the path to the JSON schemas directory."""
    return SCHEMAS_DIR


@pytest.fixture()
def all_schemas() -> dict[str, dict]:
    """Load and return all three JSON schemas as dicts."""
    schemas = {}
    for schema_file in SCHEMAS_DIR.glob("*.json"):
        with open(schema_file, encoding="utf-8") as f:
            schemas[schema_file.stem] = json.load(f)
    return schemas
