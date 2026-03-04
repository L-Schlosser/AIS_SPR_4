"""NER extractor inference wrapper for ONNX models."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

from edge_model.extraction.labels import get_id2label
from edge_model.extraction.postprocess import POSTPROCESSORS, bio_tags_to_fields


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

        Args:
            text: Input text (typically OCR output).

        Returns:
            Dict mapping field names to extracted values.
        """
        encoding = self._tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        logits = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })[0]

        tag_ids = np.argmax(logits, axis=-1)[0]

        # Convert IDs to BIO tags, skipping special tokens
        tokens: list[str] = []
        tags: list[str] = []
        offset_mapping = encoding["offset_mapping"][0]

        for idx, (tag_id, offset) in enumerate(zip(tag_ids, offset_mapping)):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if offset[0] == 0 and offset[1] == 0:
                continue
            token = self._tokenizer.convert_ids_to_tokens(int(input_ids[0, idx]))
            tokens.append(token)
            tags.append(self._id2label.get(int(tag_id), "O"))

        return bio_tags_to_fields(tokens, tags)

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
