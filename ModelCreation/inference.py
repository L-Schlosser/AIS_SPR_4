"""End-to-end document field extraction from OCR text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer

from entity_decoder import decode_entities

ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class DocumentExtractor:
    """Run ONNX NER model + entity decoder on OCR text."""

    def __init__(
        self,
        onnx_dir: Path | None = None,
        config: dict | None = None,
        use_gpu: bool = False,
    ):
        self.config = config or load_config()
        onnx_dir = onnx_dir or ROOT / self.config["paths"]["onnx_model_dir"]
        model_path = onnx_dir / self.config["paths"]["onnx_model_file"]
        tokenizer_path = onnx_dir / "tokenizer"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. Run export_onnx.py first."
            )

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        meta_path = onnx_dir / "model_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            labels = self.config["labels"]
            self.meta = {
                "labels": labels,
                "id2label": {str(i): lb for i, lb in enumerate(labels)},
                "max_length": self.config["model"]["max_length"],
                "fields_by_type": self.config["fields_by_type"],
                "field_display_de": self.config.get("field_display_de", {}),
            }

        self.id2label = {int(k): v for k, v in self.meta["id2label"].items()}
        self.max_length = self.meta["max_length"]
        self.fields_by_type = self.meta["fields_by_type"]
        self.display_map = self.meta.get("field_display_de", {})
        inf_cfg = self.config.get("inference", {})
        self.min_entity_score = inf_cfg.get("min_entity_score", 0.35)
        self.enable_regex_fallback = inf_cfg.get("enable_regex_fallback", True)

    def _preprocess(self, raw_text: str) -> tuple[list[str], dict]:
        words = raw_text.split()
        if not words:
            words = [raw_text] if raw_text.strip() else [""]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
        )
        return words, {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
            "word_ids": encoding.word_ids(batch_index=0),
        }

    def _postprocess(
        self,
        logits: np.ndarray,
        word_ids: list[int | None],
    ) -> tuple[list[str], list[float]]:
        probs = _softmax(logits[0])
        pred_ids = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)

        word_tags: list[str] = []
        word_scores: list[float] = []
        seen: set[int] = set()

        for pred_id, conf, word_id in zip(pred_ids, confidences, word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            word_tags.append(self.id2label.get(int(pred_id), "O"))
            word_scores.append(float(conf))

        return word_tags, word_scores

    def extract(self, document_type: str, raw_text: str) -> dict[str, str]:
        words, batch = self._preprocess(raw_text)
        outputs = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
        )
        logits = outputs[0]
        word_tags, word_scores = self._postprocess(logits, batch["word_ids"])

        return decode_entities(
            document_type=document_type,
            raw_text=raw_text,
            words=words,
            word_tags=word_tags,
            word_scores=word_scores,
            fields_by_type=self.fields_by_type,
            min_entity_score=self.min_entity_score,
            enable_regex_fallback=self.enable_regex_fallback,
            display_map=self.display_map,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract document fields from OCR text")
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--text", default=None, help="OCR raw text")
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()

    if args.text_file:
        raw_text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        raw_text = args.text
    else:
        raw_text = (
            "GUARDIAN HEALTH AND BEAUTY SDN BHD\n"
            "LOT B-005-006, BASEMENT LEVEL 1 THE STARLING MALL JALAN SS21/60, DAMANSARA UTAMA\n"
            "Date: 16/08/17\nTotal: 108.21"
        )

    extractor = DocumentExtractor(config=load_config(args.config))
    result = extractor.extract(args.document_type, raw_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
