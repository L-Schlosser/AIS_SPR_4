"""NER extractor inference wrapper for ONNX models."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

from edge_model.extraction.labels import get_id2label
from edge_model.extraction.postprocess import POSTPROCESSORS


class ExtractorInference:
    """Runs NER field extraction inference using an ONNX model."""

    def __init__(self, model_path: str, tokenizer_path: str, labels: list[str]) -> None:
        """Initialize the extractor with an ONNX model and tokenizer.

        Args:
            model_path: Path to the ONNX model file.
            tokenizer_path: Path to directory containing the tokenizer files.
            labels: List of BIO label strings (index-aligned with model output).
        """
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        self._id2label = get_id2label(labels)
        self._max_length = 256

    def extract(self, text: str) -> dict[str, str]:
        """Extract raw NER fields from text.

        Matches training tokenization by pre-splitting into words and using
        is_split_into_words=True. Uses word_ids() to collapse subword
        predictions back to word-level, then groups B-/I- tagged words.

        Args:
            text: Input text (typically OCR output).

        Returns:
            Dict mapping field names to extracted values.
        """
        # Pre-split into words to match training tokenization
        words = text.split()
        if not words:
            return {}

        encoding = self._tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        logits = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]

        tag_ids = np.argmax(logits, axis=-1)[0]

        # Collapse subword predictions to word-level using word_ids
        # (same alignment as training: first subword carries the label)
        word_ids = encoding.word_ids(batch_index=0)
        word_tags: list[str] = ["O"] * len(words)
        seen_words: set[int | None] = set()

        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            if word_id < len(words):
                word_tags[word_id] = self._id2label.get(int(tag_ids[idx]), "O")

        # Group consecutive B-/I- tagged words into field values
        fields: dict[str, str] = {}
        current_field: str | None = None
        current_words: list[str] = []

        for word, tag in zip(words, word_tags):
            if tag.startswith("B-"):
                # Save previous field
                if current_field and current_field not in fields and current_words:
                    fields[current_field] = " ".join(current_words)
                current_field = tag[2:].lower()
                current_words = [word]
            elif tag.startswith("I-") and current_field:
                if tag[2:].lower() == current_field:
                    current_words.append(word)
                else:
                    if current_field not in fields and current_words:
                        fields[current_field] = " ".join(current_words)
                    current_field = None
                    current_words = []
            else:
                if current_field and current_field not in fields and current_words:
                    fields[current_field] = " ".join(current_words)
                current_field = None
                current_words = []

        # Save last field
        if current_field and current_field not in fields and current_words:
            fields[current_field] = " ".join(current_words)

        return fields

    def extract_and_postprocess(self, text: str, document_type: str) -> dict:
        """Extract fields and apply document-type-specific postprocessing.

        Args:
            text: Input text (typically OCR output).
            document_type: One of the supported document types.

        Returns:
            Postprocessed field dict ready for schema validation.

        Raises:
            ValueError: If document_type has no registered postprocessor.
        """
        raw_fields = self.extract(text)

        if document_type not in POSTPROCESSORS:
            raise ValueError(f"Unknown document type: {document_type}")

        return POSTPROCESSORS[document_type](raw_fields)
