"""Schema validation utility for document extraction results."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


class SchemaValidator:
    """Validates extracted document data against JSON schemas."""

    def __init__(self, schemas_dir: str | Path = "data/schemas") -> None:
        self._schemas_dir = Path(schemas_dir)
        self._schemas: dict[str, dict] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all JSON schemas from the schemas directory."""
        if not self._schemas_dir.is_dir():
            return
        for schema_file in self._schemas_dir.glob("*.json"):
            with open(schema_file, encoding="utf-8") as f:
                schema = json.load(f)
            document_type = schema_file.stem
            self._schemas[document_type] = schema

    def validate(self, data: dict, document_type: str) -> tuple[bool, list[str]]:
        """Validate data against the schema for the given document type.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        schema = self._schemas.get(document_type)
        if schema is None:
            return False, [f"No schema found for document type: {document_type}"]

        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if not errors:
            return True, []
        return False, [e.message for e in errors]

    def get_schema(self, document_type: str) -> dict:
        """Return the raw schema for a document type.

        Raises:
            ValueError: If the document type has no schema.
        """
        schema = self._schemas.get(document_type)
        if schema is None:
            raise ValueError(f"No schema found for document type: {document_type}")
        return schema
