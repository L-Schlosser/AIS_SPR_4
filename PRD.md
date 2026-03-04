# Edge-AI Document Processing — PRD

## Overview
On-device document classification and field extraction pipeline for the BMD Go mobile app.
Three document types: Arztbesuchsbestätigung, Reisekostenbeleg, Lieferschein.
Architecture: EfficientNet-Lite0 → PaddleOCR → DistilBERT NER → JSON output.
All models ONNX-exported with INT8 quantization.

## Technical Constraints
- Python 3.12, dependency management via uv + pyproject.toml
- Inference dependencies ONLY: onnxruntime, numpy, Pillow, pydantic, jsonschema
- Training dependencies: torch, torchvision, timm, transformers, optimum, datasets, scikit-learn
- OCR: rapidocr-onnxruntime (bundles PaddleOCR ONNX models)
- All outputs validated against JSON schemas
- Every task MUST include tests. Run `uv run pytest` after each task.
- Use type hints everywhere. Run `uv run ruff check .` after each task.

## Commit Convention
- Lowercase with underscores: `add_document_classifier`
- One commit per completed task

---

## Tasks

### Phase 0: Project Scaffolding

- [x] **Task 0.1: Create pyproject.toml**
  Create `pyproject.toml` at project root with:
  ```toml
  [project]
  name = "ais-spr4-edge-doc"
  version = "0.1.0"
  description = "Edge-AI Based Document Processing for BMD Go"
  requires-python = ">=3.12,<3.13"
  dependencies = [
      "onnxruntime>=1.17.0",
      "numpy>=1.26.0",
      "Pillow>=10.0.0",
      "pydantic>=2.5.0",
      "jsonschema>=4.20.0",
  ]

  [dependency-groups]
  train = [
      "torch>=2.2.0",
      "torchvision>=0.17.0",
      "transformers>=4.38.0",
      "timm>=0.9.12",
      "datasets>=2.17.0",
      "onnx>=1.15.0",
      "optimum>=1.17.0",
      "scikit-learn>=1.4.0",
      "matplotlib>=3.8.0",
  ]
  ocr = [
      "rapidocr-onnxruntime>=1.3.0",
  ]
  viz = [
      "matplotlib>=3.8.0",
      "seaborn>=0.13.0",
  ]
  dev = [
      "pytest>=8.0.0",
      "pytest-cov>=4.1.0",
      "ruff>=0.3.0",
  ]

  [tool.ruff]
  line-length = 120
  target-version = "py312"

  [tool.ruff.lint]
  select = ["E", "F", "I", "N", "W"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  pythonpath = ["."]
  markers = [
      "integration: requires ONNX models",
      "e2e: full pipeline test",
  ]
  ```
  Then run `uv sync --group dev` to install base + dev dependencies.
  **Validation**: `uv run python -c "import onnxruntime; print(onnxruntime.__version__)"` succeeds.

- [x] **Task 0.2: Create directory structure and __init__.py files**
  Create ALL of these directories and files:
  ```
  data/samples/arztbesuchsbestaetigung/.gitkeep
  data/samples/reisekostenbeleg/.gitkeep
  data/samples/lieferschein/.gitkeep
  data/schemas/.gitkeep
  edge_model/__init__.py
  edge_model/classification/__init__.py
  edge_model/extraction/__init__.py
  edge_model/inference/__init__.py
  ocr/__init__.py
  api/__init__.py
  mobile_app/__init__.py
  mobile_app/src/__init__.py
  scripts/.gitkeep
  tests/__init__.py
  tests/unit/__init__.py
  tests/integration/__init__.py
  tests/e2e/__init__.py
  docs/.gitkeep
  ```
  IMPORTANT: Use `edge_model` (underscore) not `edge-model` (hyphen) for Python package names. Same for `mobile_app`.
  Each `__init__.py` can be empty or contain a module docstring.
  **Validation**: `uv run python -c "import edge_model; import ocr; import api"` succeeds.

- [x] **Task 0.3: Create .gitignore**
  Create `.gitignore` at project root:
  ```
  __pycache__/
  *.pyc
  .venv/
  .python-version
  *.egg-info/
  dist/
  build/
  .ruff_cache/
  .mypy_cache/
  .pytest_cache/

  # Model artifacts
  *.onnx
  *.pt
  *.pth
  *.bin
  *.ort
  edge_model/classification/models/
  edge_model/extraction/models/

  # Data (keep schemas)
  data/samples/**/*.png
  data/samples/**/*.jpg
  data/samples/**/*.jpeg
  data/samples/**/*.jsonl
  !data/schemas/*.json

  # IDE
  .vscode/
  .idea/

  # OS
  .DS_Store
  Thumbs.db
  ```
  **Validation**: File exists and contains `__pycache__`.

- [x] **Task 0.4: Create progress.txt**
  Create `progress.txt` at project root with content:
  ```
  # Progress Log
  This file tracks learnings across iterations.
  ```
  **Validation**: File exists.

### Phase 1: JSON Schemas and Data Models

- [x] **Task 1.1: Create Arztbesuchsbestätigung JSON schema**
  Create `data/schemas/arztbesuchsbestaetigung.json`:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Arztbesuchsbestaetigung",
    "description": "Medical visit confirmation document",
    "type": "object",
    "required": ["document_type", "patient_name", "doctor_name", "facility_name", "visit_date"],
    "properties": {
      "document_type": { "const": "arztbesuchsbestaetigung" },
      "patient_name": { "type": "string", "minLength": 1 },
      "doctor_name": { "type": "string", "minLength": 1 },
      "facility_name": { "type": "string", "minLength": 1 },
      "facility_address": { "type": "string" },
      "visit_date": { "type": "string", "format": "date", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
      "visit_time": { "type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]$" },
      "duration_minutes": { "type": "integer", "minimum": 1, "maximum": 480 },
      "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
    },
    "additionalProperties": false
  }
  ```
  **Validation**: `uv run python -c "import json, jsonschema; s=json.load(open('data/schemas/arztbesuchsbestaetigung.json')); jsonschema.Draft7Validator.check_schema(s); print('valid')"` prints "valid".

- [x] **Task 1.2: Create Reisekostenbeleg JSON schema**
  Create `data/schemas/reisekostenbeleg.json`:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Reisekostenbeleg",
    "description": "Business travel expense receipt",
    "type": "object",
    "required": ["document_type", "vendor_name", "date", "amount", "currency"],
    "properties": {
      "document_type": { "const": "reisekostenbeleg" },
      "vendor_name": { "type": "string", "minLength": 1 },
      "vendor_address": { "type": "string" },
      "date": { "type": "string", "format": "date", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
      "amount": { "type": "number", "exclusiveMinimum": 0 },
      "currency": { "type": "string", "enum": ["EUR"] },
      "vat_rate": { "type": "number", "minimum": 0, "maximum": 100 },
      "vat_amount": { "type": "number", "minimum": 0 },
      "category": { "type": "string", "enum": ["hotel", "restaurant", "transport", "other"] },
      "description": { "type": "string" },
      "receipt_number": { "type": "string" },
      "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
    },
    "additionalProperties": false
  }
  ```
  **Validation**: Same jsonschema check as Task 1.1 but for this file.

- [x] **Task 1.3: Create Lieferschein JSON schema**
  Create `data/schemas/lieferschein.json`:
  ```json
  {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Lieferschein",
    "description": "Delivery note document",
    "type": "object",
    "required": ["document_type", "delivery_note_number", "delivery_date", "sender", "recipient"],
    "properties": {
      "document_type": { "const": "lieferschein" },
      "delivery_note_number": { "type": "string", "minLength": 1 },
      "delivery_date": { "type": "string", "format": "date", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
      "sender": {
        "type": "object",
        "required": ["name"],
        "properties": {
          "name": { "type": "string", "minLength": 1 },
          "address": { "type": "string" }
        },
        "additionalProperties": false
      },
      "recipient": {
        "type": "object",
        "required": ["name"],
        "properties": {
          "name": { "type": "string", "minLength": 1 },
          "address": { "type": "string" }
        },
        "additionalProperties": false
      },
      "order_number": { "type": "string" },
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["description", "quantity", "unit"],
          "properties": {
            "description": { "type": "string" },
            "quantity": { "type": "number", "minimum": 0 },
            "unit": { "type": "string" }
          },
          "additionalProperties": false
        }
      },
      "total_weight": { "type": "string" },
      "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
    },
    "additionalProperties": false
  }
  ```
  **Validation**: Same jsonschema check.

- [x] **Task 1.4: Create Pydantic models and DocumentType enum**
  Create `api/models.py`:
  - `DocumentType` enum with values: `arztbesuchsbestaetigung`, `reisekostenbeleg`, `lieferschein`
  - `ArztbesuchsbestaetigungResult(BaseModel)` with all fields from schema 1.1
  - `ReisekostenbelegResult(BaseModel)` with all fields from schema 1.2
  - `LieferscheinResult(BaseModel)` with all fields from schema 1.3
  - `ProcessingResult(BaseModel)` with: `document_type: DocumentType`, `fields: dict`, `confidence: float`, `raw_text: str | None = None`
  - Use `Field(...)` with descriptions for all required fields.
  - All date fields should be `str` with pattern validation (not datetime).
  Create `tests/unit/test_models.py`:
  - Test valid instantiation of each model with sample data
  - Test that required fields raise ValidationError when missing
  - Test enum values
  - Test that extra fields are rejected
  **Validation**: `uv run pytest tests/unit/test_models.py -v` — all tests pass.

- [x] **Task 1.5: Create schema validation utility**
  Create `edge_model/inference/validator.py`:
  - `SchemaValidator` class that loads JSON schemas from `data/schemas/` directory
  - `validate(data: dict, document_type: str) -> tuple[bool, list[str]]` — validates data against the matching schema, returns (is_valid, error_messages)
  - `get_schema(document_type: str) -> dict` — returns the raw schema
  - Handle missing schema files gracefully with clear error messages.
  Create `tests/unit/test_validator.py`:
  - Test validation of correct data passes
  - Test validation of incorrect data fails with meaningful errors
  - Test missing required fields
  - Test wrong document_type
  - Test schema loading from directory
  **Validation**: `uv run pytest tests/unit/test_validator.py -v` — all tests pass.

### Phase 2: Synthetic Data Generation

- [x] **Task 2.1: Create document image generator**
  First run: `uv add faker --group train`
  Create `scripts/generate_samples.py`:
  - Uses Pillow + Faker (de_DE locale) to generate synthetic document images
  - `generate_arztbesuch(output_dir, count=250)`:
    - White canvas 800x1130px (A4 ratio)
    - Header: "BESTÄTIGUNG ARZTBESUCH" centered, bold
    - Fields rendered in German document layout: Praxis name/address, Patient name, Date, Time, Duration
    - Random German doctor names, facility names, addresses via Faker
    - Random dates (2023-2025), times (08:00-18:00), durations (15-120 min)
  - `generate_reisekosten(output_dir, count=250)`:
    - Header varies: "RECHNUNG", "QUITTUNG", "BELEG"
    - Fields: Company name/address, date, items with prices, subtotal, MwSt (VAT), total
    - Random amounts (5-500 EUR), VAT rates (10%, 13%, 20% — Austrian rates)
    - Categories: hotel/restaurant/transport reflected in layout
  - `generate_lieferschein(output_dir, count=250)`:
    - Header: "LIEFERSCHEIN" with delivery note number (LS-XXXXX)
    - Sender/recipient blocks with company names and addresses
    - Items table with description, quantity, unit columns
    - Order reference, delivery date, total weight
  - Apply augmentations to 50% of images: slight rotation (±3°), Gaussian noise, brightness jitter (±15%), slight blur
  - Each image saved as PNG with companion `_label.json` containing ground-truth extracted fields matching the JSON schemas
  - `main()` generates all three types with `argparse` for count/output_dir
  **Validation**: `uv run python scripts/generate_samples.py --count 10 --output-dir data/samples` generates 30 images (10 per type) + 30 label files. Spot-check: open one image to confirm it looks like a document.

- [x] **Task 2.2: Create NER text sample generator**
  Create `scripts/generate_text_samples.py`:
  - Generates BIO-tagged text simulating OCR output for NER training
  - For each document type, creates realistic text sequences with BIO tags
  - `generate_arztbesuch_ner(count=1500)`:
    - Tokens like: ["Bestätigung", "Arztbesuch", "Patient", ":", "Max", "Mustermann", ...]
    - BIO tags: ["O", "O", "O", "O", "B-PATIENT", "I-PATIENT", ...]
    - Uses Faker de_DE for names, addresses, dates
    - Varies word order and formatting to simulate real OCR output
  - `generate_reisekosten_ner(count=1500)`:
    - Tags: B-VENDOR, I-VENDOR, B-DATE, B-AMOUNT, B-CURRENCY, B-VAT_RATE, B-VAT_AMOUNT, B-CATEGORY, B-DESC, I-DESC, B-RECEIPT_NUM
  - `generate_lieferschein_ner(count=1500)`:
    - Tags: B-DELNR, B-DELDATE, B-SENDER, I-SENDER, B-SADDR, I-SADDR, B-RECIP, I-RECIP, B-RADDR, I-RADDR, B-ORDNR, B-ITEM_DESC, I-ITEM_DESC, B-ITEM_QTY, B-ITEM_UNIT, B-WEIGHT
  - Output format: JSONL files at `data/samples/{doc_type}_ner_train.jsonl`
    Each line: `{"tokens": [...], "ner_tags": [...], "document_type": "..."}`
  - Include train/val split (80/20) — generate `_train.jsonl` and `_val.jsonl`
  **Validation**: `uv run python scripts/generate_text_samples.py --count 50` generates 6 JSONL files (train+val for each type). Verify: `uv run python -c "import json; data=[json.loads(l) for l in open('data/samples/arztbesuchsbestaetigung_ner_train.jsonl')]; print(len(data), 'samples'); assert len(data[0]['tokens'])==len(data[0]['ner_tags'])"`.

- [x] **Task 2.3: Create data loading and verification tests**
  Create `tests/unit/test_data_generation.py`:
  - Test that generated label files match their JSON schemas
  - Test token/tag length alignment in NER samples
  - Test that all expected BIO tags appear in the generated data
  - Test augmentation produces visually different images (compare pixel arrays)
  - Test Faker generates German-locale data (names, addresses)
  Create `tests/conftest.py`:
  - `@pytest.fixture` for sample image path (generates 1 image in tmp dir)
  - `@pytest.fixture` for sample NER data (generates 5 samples in memory)
  - `@pytest.fixture` for schema directory path
  - `@pytest.fixture` for all three loaded schemas as dicts
  **Validation**: `uv run pytest tests/unit/test_data_generation.py tests/conftest.py -v` — all pass.

### Phase 3: Document Classification

- [x] **Task 3.1: Create classification dataset and config**
  Create `edge_model/classification/config.py`:
  - `ClassificationConfig` dataclass with: image_size=224, num_classes=3, batch_size=16, lr_frozen=1e-4, lr_unfrozen=1e-5, epochs_frozen=5, epochs_unfrozen=10, model_name="tf_efficientnet_lite0", class_names=["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
  Create `edge_model/classification/dataset.py`:
  - `DocumentDataset(torch.utils.data.Dataset)`:
    - `__init__(self, root_dir, transform=None)` — loads images from subdirectories, using directory name as label
    - `__getitem__` returns (image_tensor, label_index)
    - `__len__` returns total count
  - `get_transforms(image_size, is_training=True)` — returns torchvision transforms:
    - Training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Normalize (ImageNet stats)
    - Eval: Resize, CenterCrop, Normalize
  - `create_dataloaders(root_dir, config, val_split=0.2)` — returns train_loader, val_loader with stratified split
  Create `tests/unit/test_classifier_dataset.py`:
  - Test dataset loading with mock directory structure
  - Test transforms output correct shape (3, 224, 224)
  - Test label mapping is correct
  - Test train/val split maintains class proportions
  **Validation**: `uv run pytest tests/unit/test_classifier_dataset.py -v` — all pass.

- [x] **Task 3.2: Create classification training script**
  Run: `uv sync --group train`
  Create `edge_model/classification/train.py`:
  - `create_model(config) -> nn.Module` — loads timm `tf_efficientnet_lite0` pretrained, replaces classifier head for 3 classes
  - `train_epoch(model, loader, criterion, optimizer, device)` — single epoch, returns avg loss + accuracy
  - `validate_epoch(model, loader, criterion, device)` — eval mode, returns avg loss + accuracy + per-class metrics
  - `train(config, data_dir, output_dir)`:
    1. Create dataloaders from data_dir
    2. Create model, move to device (cuda if available, else cpu)
    3. Phase 1: Freeze all except classifier head, train for config.epochs_frozen
    4. Phase 2: Unfreeze all, train for config.epochs_unfrozen with lower LR
    5. Save best model (by val accuracy) to output_dir/best_model.pt
    6. Save training metrics to output_dir/metrics.json
    7. Print final accuracy and per-class precision/recall
  - `if __name__ == "__main__":` with argparse for data_dir, output_dir
  Create `tests/unit/test_classifier_train.py`:
  - Test model creation has correct output shape (batch, 3)
  - Test frozen phase: verify backbone params have requires_grad=False
  - Test unfrozen phase: verify all params have requires_grad=True
  - Test train_epoch runs without error on tiny dataset (2 images per class)
  **Validation**: `uv run pytest tests/unit/test_classifier_train.py -v` — all pass.

- [x] **Task 3.3: Create ONNX export for classifier**
  Create `edge_model/classification/export_onnx.py`:
  - `export_to_onnx(model_path, output_path, config)`:
    1. Load PyTorch model from model_path
    2. Create dummy input: torch.randn(1, 3, 224, 224)
    3. `torch.onnx.export(model, dummy_input, output_path, input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)`
    4. Run `onnx.checker.check_model()` on exported model
    5. Print model size in MB
  - `quantize_model(onnx_path, output_path)`:
    1. `from onnxruntime.quantization import quantize_dynamic, QuantType`
    2. `quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)`
    3. Print quantized model size
  - `verify_onnx(onnx_path, sample_image_path)`:
    1. Load ONNX via `ort.InferenceSession`
    2. Run inference on sample image
    3. Print predicted class and confidence
    4. Assert output shape is (1, 3)
  - `if __name__ == "__main__":` runs export → quantize → verify
  Create `tests/unit/test_classifier_export.py`:
  - Test export produces valid ONNX file (using a tiny random model)
  - Test quantized model is smaller than original
  - Test ONNX inference produces correct output shape
  **Validation**: `uv run pytest tests/unit/test_classifier_export.py -v` — all pass.

- [x] **Task 3.4: Create classification validation script**
  Create `edge_model/classification/validate.py`:
  - `validate_onnx_model(model_path, data_dir, config)`:
    1. Load ONNX model via onnxruntime
    2. Load validation images from data_dir
    3. Run inference on each, collect predictions
    4. Compute: overall accuracy, per-class precision/recall/F1, confusion matrix
    5. Print results as formatted table
    6. Save results to JSON file
    7. Return dict with all metrics
  - `if __name__ == "__main__":` with argparse
  Create `tests/unit/test_classifier_validate.py`:
  - Test metric computation with known predictions/labels
  - Test confusion matrix shape is (3, 3)
  **Validation**: `uv run pytest tests/unit/test_classifier_validate.py -v` — all pass.

### Phase 4: OCR Module

- [x] **Task 4.1: Create OCR engine**
  Run: `uv sync --group ocr`
  Create `ocr/engine.py`:
  - `class OCREngine`:
    - `__init__(self, use_gpu=False)` — initializes RapidOCR with config
    - `extract_text(self, image: np.ndarray) -> OCRResult`:
      - Runs RapidOCR on image
      - Returns `OCRResult` with: `text` (full text), `regions` (list of `TextRegion` with text, bbox, confidence), `processing_time_ms`
    - `extract_text_from_file(self, file_path: str) -> OCRResult`
  - `@dataclass OCRResult`: text: str, regions: list[TextRegion], processing_time_ms: float
  - `@dataclass TextRegion`: text: str, bbox: tuple[float, float, float, float], confidence: float
  Create `ocr/preprocessing.py`:
  - `preprocess_for_ocr(image: np.ndarray) -> np.ndarray`:
    - Convert to grayscale if colored
    - Apply adaptive thresholding for better text contrast
    - Deskew if rotation detected (using Hough transform or similar)
    - Return preprocessed image
  Create `ocr/postprocessing.py`:
  - `sort_regions_by_position(regions: list[TextRegion]) -> list[TextRegion]`:
    - Sort by y-coordinate first (top to bottom), then x (left to right)
  - `merge_text(regions: list[TextRegion]) -> str`:
    - Combine sorted regions into a single text string with line breaks where vertical gaps exist
  Create `tests/unit/test_ocr.py`:
  - Test OCREngine initialization
  - Test preprocessing converts image correctly
  - Test postprocessing sorts regions correctly
  - Test text merging produces readable output
  - Mock RapidOCR for unit tests (don't require actual OCR model)
  **Validation**: `uv run pytest tests/unit/test_ocr.py -v` — all pass.

- [ ] **Task 4.2: Create OCR integration test**
  Create `tests/integration/test_ocr_engine.py`:
  - Generate a simple test image with known text using Pillow (draw "Hello World 123" on white background)
  - Run OCR engine on it
  - Assert that extracted text contains "Hello" and "123"
  - Measure and print processing time
  - Mark with `@pytest.mark.integration`
  **Validation**: `uv run pytest tests/integration/test_ocr_engine.py -v -m integration` — passes (requires OCR models downloaded).

### Phase 5: Field Extraction (NER)

- [ ] **Task 5.1: Create NER label definitions**
  Create `edge_model/extraction/labels.py`:
  - `ARZTBESUCH_LABELS`: list of BIO tags = ["O", "B-PATIENT", "I-PATIENT", "B-DOCTOR", "I-DOCTOR", "B-FACILITY", "I-FACILITY", "B-ADDRESS", "I-ADDRESS", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-DURATION"]
  - `REISEKOSTEN_LABELS`: ["O", "B-VENDOR", "I-VENDOR", "B-VADDRESS", "I-VADDRESS", "B-DATE", "I-DATE", "B-AMOUNT", "B-CURRENCY", "B-VAT_RATE", "B-VAT_AMOUNT", "B-CATEGORY", "B-DESC", "I-DESC", "B-RECEIPT_NUM"]
  - `LIEFERSCHEIN_LABELS`: ["O", "B-DELNR", "B-DELDATE", "B-SENDER", "I-SENDER", "B-SADDR", "I-SADDR", "B-RECIP", "I-RECIP", "B-RADDR", "I-RADDR", "B-ORDNR", "B-ITEM_DESC", "I-ITEM_DESC", "B-ITEM_QTY", "B-ITEM_UNIT", "B-WEIGHT"]
  - `LABEL_SETS: dict[str, list[str]]` mapping document_type to its labels
  - `get_label2id(labels) -> dict[str, int]` and `get_id2label(labels) -> dict[int, str]`
  Create `tests/unit/test_labels.py`:
  - Test all label lists start with "O"
  - Test every B- tag that has a matching I- tag
  - Test label2id and id2label are inverse of each other
  - Test LABEL_SETS contains all 3 document types
  **Validation**: `uv run pytest tests/unit/test_labels.py -v` — all pass.

- [ ] **Task 5.2: Create NER dataset and config**
  Create `edge_model/extraction/config.py`:
  - `ExtractionConfig` dataclass: model_name="distilbert-base-german-cased", max_length=256, batch_size=16, lr=5e-5, epochs=20, weight_decay=0.01
  Create `edge_model/extraction/dataset.py`:
  - `NERDataset`:
    - Loads JSONL files from data/samples/
    - Tokenizes with DistilBertTokenizerFast
    - Handles subword alignment: when a word is split into subwords, the first subword gets the original tag, subsequent subwords get -100 (ignored in loss)
    - Returns input_ids, attention_mask, labels tensors
  - `load_ner_data(jsonl_path) -> list[dict]` — loads and validates JSONL
  - `create_ner_dataloaders(train_path, val_path, config, labels)` — returns train/val dataloaders
  Create `tests/unit/test_ner_dataset.py`:
  - Test JSONL loading
  - Test subword alignment is correct
  - Test special tokens ([CLS], [SEP]) get -100 labels
  - Test batch collation works
  **Validation**: `uv run pytest tests/unit/test_ner_dataset.py -v` — all pass.

- [ ] **Task 5.3: Create NER training script**
  Create `edge_model/extraction/train.py`:
  - `train_ner_model(config, document_type, train_path, val_path, output_dir)`:
    1. Load labels from labels.py for given document_type
    2. Load tokenizer: `DistilBertTokenizerFast.from_pretrained(config.model_name)`
    3. Load model: `DistilBertForTokenClassification.from_pretrained(config.model_name, num_labels=len(labels))`
    4. Create datasets and dataloaders
    5. Use HuggingFace `Trainer` with `TrainingArguments`:
       - output_dir, num_train_epochs, per_device_train_batch_size, learning_rate, weight_decay
       - evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True
       - metric_for_best_model="f1"
    6. Compute metrics function: precision, recall, F1 (using seqeval-style evaluation, ignoring -100 and "O" tags)
    7. Save best model + tokenizer to output_dir
    8. Print per-entity-type F1 scores
  - `if __name__ == "__main__":` with argparse for document_type, train_path, val_path, output_dir
  - NOTE: Use `distilbert-base-german-cased` as base model for German text
  Create `tests/unit/test_ner_train.py`:
  - Test model creation with correct num_labels
  - Test compute_metrics function with known inputs
  - Test training runs for 1 epoch on tiny data without errors
  **Validation**: `uv run pytest tests/unit/test_ner_train.py -v` — all pass.

- [ ] **Task 5.4: Create NER postprocessing**
  Create `edge_model/extraction/postprocess.py`:
  - `bio_tags_to_fields(tokens: list[str], tags: list[str]) -> dict[str, str]`:
    - Merge B-/I- tagged tokens into field values
    - Handle multiple occurrences of same field (take first or highest confidence)
    - Clean up subword artifacts (## prefixes from tokenizer)
    - Return flat dict of field_name -> value
  - `postprocess_arztbesuch(raw_fields: dict) -> dict`:
    - Parse date strings into ISO format (YYYY-MM-DD)
    - Parse time strings into HH:MM format
    - Parse duration to integer
    - Add document_type field
  - `postprocess_reisekosten(raw_fields: dict) -> dict`:
    - Parse amount to float
    - Parse VAT fields to float
    - Normalize category to lowercase enum value
    - Add document_type and currency fields
  - `postprocess_lieferschein(raw_fields: dict) -> dict`:
    - Group ITEM_DESC/ITEM_QTY/ITEM_UNIT into items array
    - Structure sender/recipient as nested objects
    - Add document_type field
  - `POSTPROCESSORS: dict[str, Callable]` mapping document_type to its postprocessor
  Create `tests/unit/test_postprocess.py`:
  - Test bio_tags_to_fields with various tag sequences
  - Test subword cleaning (## removal)
  - Test each postprocessor produces schema-valid output
  - Test edge cases: empty tags, all O tags, missing fields
  - Test that postprocessed output validates against JSON schemas
  **Validation**: `uv run pytest tests/unit/test_postprocess.py -v` — all pass.

- [ ] **Task 5.5: Create NER ONNX export**
  Create `edge_model/extraction/export_onnx.py`:
  - `export_ner_to_onnx(model_dir, output_path)`:
    1. Load model using `optimum.onnxruntime.ORTModelForTokenClassification.from_pretrained(model_dir, export=True)`
    2. Save to output_path
    3. Validate with onnx.checker
    4. Print model size
  - `quantize_ner_model(onnx_dir, output_dir)`:
    1. Load and quantize to INT8
    2. Save quantized model
    3. Print size comparison
  - `verify_ner_onnx(model_dir, tokenizer_dir, sample_text)`:
    1. Load ONNX model via ORTModelForTokenClassification
    2. Tokenize sample text
    3. Run inference
    4. Print predicted tags
    5. Assert output shape matches (1, seq_len, num_labels)
  - `if __name__ == "__main__":` exports all three models
  Create `tests/unit/test_ner_export.py`:
  - Test that export function runs without error (using tiny model)
  - Test quantized model is smaller
  **Validation**: `uv run pytest tests/unit/test_ner_export.py -v` — all pass.

### Phase 6: Inference Pipeline

- [ ] **Task 6.1: Create classifier inference wrapper**
  Create `edge_model/inference/classifier_inference.py`:
  - `class ClassifierInference`:
    - `__init__(self, model_path: str, class_names: list[str])`:
      - Load ONNX model via `ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])`
      - Store class_names
    - `predict(self, image: np.ndarray) -> tuple[str, float]`:
      - Preprocess image (resize 224x224, normalize, HWC→CHW, add batch dim)
      - Run inference
      - Apply softmax to logits
      - Return (class_name, confidence)
    - `predict_batch(self, images: list[np.ndarray]) -> list[tuple[str, float]]`
  Create `edge_model/inference/preprocessor.py`:
  - `class ImagePreprocessor`:
    - `prepare_for_classification(image: np.ndarray, size=224) -> np.ndarray`:
      - Resize to (size, size) with Pillow
      - Convert to float32, normalize with ImageNet mean/std
      - Transpose to CHW, add batch dimension
      - Return numpy array ready for ONNX input
    - `load_image(file_path: str) -> np.ndarray`: Load and return as RGB numpy array
  Create `tests/unit/test_classifier_inference.py`:
  - Test preprocessor output shape is (1, 3, 224, 224)
  - Test preprocessor normalizes to approximately [-2.5, 2.5] range
  - Mock ONNX session to test predict returns correct format
  **Validation**: `uv run pytest tests/unit/test_classifier_inference.py -v` — all pass.

- [ ] **Task 6.2: Create extractor inference wrapper**
  Create `edge_model/inference/extractor_inference.py`:
  - `class ExtractorInference`:
    - `__init__(self, model_path: str, tokenizer_path: str, labels: list[str])`:
      - Load ONNX model via ort.InferenceSession
      - Load tokenizer via DistilBertTokenizerFast.from_pretrained
      - Build id2label mapping
    - `extract(self, text: str) -> dict[str, str]`:
      1. Tokenize text with padding/truncation to max_length=256
      2. Run ONNX inference
      3. Argmax on logits to get tag IDs
      4. Convert IDs to BIO tags (skip special tokens)
      5. Call bio_tags_to_fields from postprocess module
      6. Return raw field dict
    - `extract_and_postprocess(self, text: str, document_type: str) -> dict`:
      1. Call extract()
      2. Apply document-type-specific postprocessor
      3. Return processed fields
  Create `tests/unit/test_extractor_inference.py`:
  - Test tokenization produces correct input shapes
  - Mock ONNX session to test extract returns field dict
  - Test that postprocessing is applied correctly
  **Validation**: `uv run pytest tests/unit/test_extractor_inference.py -v` — all pass.

- [ ] **Task 6.3: Create main pipeline orchestrator**
  Create `edge_model/inference/config.py`:
  - `@dataclass PipelineConfig`:
    - classifier_model_path: str
    - extractor_model_paths: dict[str, str]  (document_type -> model_path)
    - extractor_tokenizer_paths: dict[str, str]
    - schemas_dir: str = "data/schemas"
    - confidence_threshold: float = 0.7
    - use_ocr: bool = True
  - `load_config(config_path: str) -> PipelineConfig` — loads from YAML
  Create `edge_model/inference/pipeline.py`:
  - `class DocumentPipeline`:
    - `__init__(self, config: PipelineConfig)`:
      - Initialize ClassifierInference
      - Initialize 3 ExtractorInference instances (one per doc type)
      - Initialize SchemaValidator
      - Optionally initialize OCREngine (if config.use_ocr)
    - `process(self, image: np.ndarray) -> ProcessingResult`:
      1. Classify document type → (doc_type, confidence)
      2. If confidence < threshold: return result with low_confidence flag
      3. Run OCR on image → text
      4. Run type-specific extractor on text → fields
      5. Validate fields against JSON schema
      6. Return ProcessingResult with all data
    - `process_file(self, file_path: str) -> ProcessingResult`:
      - Load image from file, call process()
    - `@classmethod from_config(cls, config_path: str) -> DocumentPipeline`
  Create `tests/unit/test_pipeline.py`:
  - Test pipeline initialization with mocked components
  - Test process flow: classifier → OCR → extractor → validator
  - Test low confidence handling
  - Test that correct extractor is chosen based on doc_type
  **Validation**: `uv run pytest tests/unit/test_pipeline.py -v` — all pass.

### Phase 7: API Service Layer

- [ ] **Task 7.1: Create document service**
  Create `api/service.py`:
  - `class DocumentService`:
    - `__init__(self, config_path: str = "config.yaml")`:
      - Load config and initialize DocumentPipeline
    - `process_image(self, image_bytes: bytes) -> ProcessingResult`:
      - Decode image from bytes (JPEG/PNG)
      - Run pipeline
      - Return ProcessingResult
    - `process_image_file(self, file_path: str) -> ProcessingResult`:
      - Load file, run pipeline
    - `get_supported_types(self) -> list[str]`:
      - Return list of supported document types
    - `get_schema(self, document_type: str) -> dict`:
      - Return JSON schema for given type
  Create `config.yaml` at project root:
  ```yaml
  classifier_model_path: "edge_model/classification/models/classifier_int8.onnx"
  extractor_model_paths:
    arztbesuchsbestaetigung: "edge_model/extraction/models/arztbesuch/model.onnx"
    reisekostenbeleg: "edge_model/extraction/models/reisekosten/model.onnx"
    lieferschein: "edge_model/extraction/models/lieferschein/model.onnx"
  extractor_tokenizer_paths:
    arztbesuchsbestaetigung: "edge_model/extraction/models/arztbesuch/"
    reisekostenbeleg: "edge_model/extraction/models/reisekosten/"
    lieferschein: "edge_model/extraction/models/lieferschein/"
  schemas_dir: "data/schemas"
  confidence_threshold: 0.7
  use_ocr: true
  ```
  Create `tests/unit/test_service.py`:
  - Test service initialization (mock pipeline)
  - Test process_image decodes bytes correctly
  - Test get_supported_types returns all 3 types
  - Test get_schema returns valid schema dict
  **Validation**: `uv run pytest tests/unit/test_service.py -v` — all pass.

### Phase 8: Mobile Demo App

- [ ] **Task 8.1: Create CLI demo application**
  Create `mobile_app/src/model_manager.py`:
  - `class ModelManager`:
    - `__init__(self, models_dir: str)`:
      - Scans directory for ONNX model files
    - `get_model_info(self) -> dict`: returns model names, sizes, paths
    - `check_models_exist(self) -> tuple[bool, list[str]]`: checks all required models exist, returns (all_present, missing_list)
    - `get_total_size_mb(self) -> float`: total size of all models in MB
  Create `mobile_app/src/app.py`:
  - CLI application using argparse:
    - `process` command: `python -m mobile_app.src.app process <image_path>` → runs full pipeline, prints JSON output
    - `info` command: shows model info, sizes, supported document types
    - `batch` command: process all images in a directory
    - `demo` command: generates a sample image, processes it, shows result (uses generate_samples internally)
  - Pretty-prints results with colors (using simple ANSI codes)
  - Shows confidence scores and processing time
  Create `mobile_app/README.md`:
  - Quick start instructions
  - ONNX Runtime Mobile integration guide for Android (Java/Kotlin) and iOS (Swift)
  - Model size overview
  - API reference for DocumentService
  Create `tests/unit/test_mobile_app.py`:
  - Test model_manager.check_models_exist with mock directory
  - Test app CLI argument parsing
  **Validation**: `uv run pytest tests/unit/test_mobile_app.py -v` — all pass.

### Phase 9: Integration & E2E Tests

- [ ] **Task 9.1: Create integration tests**
  Create `tests/integration/test_classification_pipeline.py`:
  - @pytest.mark.integration
  - Generate synthetic test images (3, one per type)
  - If classifier model exists: run classification, assert correct predictions
  - If model doesn't exist: skip with clear message
  - Test preprocessing + classification together
  Create `tests/integration/test_extraction_pipeline.py`:
  - @pytest.mark.integration
  - Use sample OCR text for each document type
  - If extractor models exist: run extraction, validate output against schemas
  - Test NER + postprocessing together
  **Validation**: `uv run pytest tests/integration/ -v -m integration` — passes (skips if models unavailable).

- [ ] **Task 9.2: Create end-to-end tests**
  Create `tests/e2e/test_full_pipeline.py`:
  - @pytest.mark.e2e
  - `test_arztbesuch_e2e`: Generate arztbesuch image → process → validate JSON against schema
  - `test_reisekosten_e2e`: Same for reisekostenbeleg
  - `test_lieferschein_e2e`: Same for lieferschein
  - `test_unknown_image`: Process a non-document image → should return low confidence
  - `test_pipeline_timing`: Assert total processing time < 2000ms on CPU
  Create `tests/e2e/test_all_document_types.py`:
  - @pytest.mark.e2e
  - Parametrized test across all 3 document types
  - For each: generate 5 images, process all, check >80% are correctly classified
  **Validation**: `uv run pytest tests/e2e/ -v -m e2e` — passes with trained models.

### Phase 10: Training Execution

- [ ] **Task 10.1: Generate all training data**
  Run the data generation scripts:
  ```bash
  uv run python scripts/generate_samples.py --count 250 --output-dir data/samples
  uv run python scripts/generate_text_samples.py --count 1500 --output-dir data/samples
  ```
  **Validation**: Verify file counts:
  - `data/samples/arztbesuchsbestaetigung/` has 250 PNG + 250 JSON files
  - `data/samples/reisekostenbeleg/` has 250 PNG + 250 JSON files
  - `data/samples/lieferschein/` has 250 PNG + 250 JSON files
  - 6 JSONL files exist (train + val for each type)

- [ ] **Task 10.2: Train classification model**
  Run: `uv sync --group train`
  Execute:
  ```bash
  uv run python -m edge_model.classification.train --data-dir data/samples --output-dir edge_model/classification/models
  ```
  **Validation**: `edge_model/classification/models/best_model.pt` exists and `metrics.json` shows val_accuracy > 0.85.

- [ ] **Task 10.3: Export classifier to ONNX**
  Execute:
  ```bash
  uv run python -m edge_model.classification.export_onnx --model-path edge_model/classification/models/best_model.pt --output-path edge_model/classification/models/classifier.onnx
  ```
  **Validation**: Both `classifier.onnx` and `classifier_int8.onnx` exist. INT8 model is < 10MB. `uv run python -m edge_model.classification.validate --model-path edge_model/classification/models/classifier_int8.onnx --data-dir data/samples` shows accuracy > 0.85.

- [ ] **Task 10.4: Train all three NER extraction models**
  Execute for each document type:
  ```bash
  uv run python -m edge_model.extraction.train --document-type arztbesuchsbestaetigung --train-path data/samples/arztbesuchsbestaetigung_ner_train.jsonl --val-path data/samples/arztbesuchsbestaetigung_ner_val.jsonl --output-dir edge_model/extraction/models/arztbesuch
  uv run python -m edge_model.extraction.train --document-type reisekostenbeleg --train-path data/samples/reisekostenbeleg_ner_train.jsonl --val-path data/samples/reisekostenbeleg_ner_val.jsonl --output-dir edge_model/extraction/models/reisekosten
  uv run python -m edge_model.extraction.train --document-type lieferschein --train-path data/samples/lieferschein_ner_train.jsonl --val-path data/samples/lieferschein_ner_val.jsonl --output-dir edge_model/extraction/models/lieferschein
  ```
  **Validation**: Each model dir has `config.json`, `model.safetensors`, `tokenizer.json`. Training logs show F1 > 0.80 on validation set.

- [ ] **Task 10.5: Export all NER models to ONNX**
  Execute:
  ```bash
  uv run python -m edge_model.extraction.export_onnx --model-dir edge_model/extraction/models/arztbesuch --output-dir edge_model/extraction/models/arztbesuch/onnx
  uv run python -m edge_model.extraction.export_onnx --model-dir edge_model/extraction/models/reisekosten --output-dir edge_model/extraction/models/reisekosten/onnx
  uv run python -m edge_model.extraction.export_onnx --model-dir edge_model/extraction/models/lieferschein --output-dir edge_model/extraction/models/lieferschein/onnx
  ```
  **Validation**: Each ONNX dir has quantized model < 25MB. Verification script prints correct tag predictions on sample text.

### Phase 11: Documentation & Final Validation

- [ ] **Task 11.1: Create architecture documentation**
  Create `docs/architecture.md`:
  - System overview with Mermaid diagram showing the full pipeline
  - Component descriptions (classifier, OCR, extractors, validator)
  - Data flow diagram
  - Model specifications table (name, size, input/output)
  - Dependency graph between modules
  - ONNX Runtime Mobile deployment notes
  Create `docs/model_pipeline.md`:
  - Training instructions for each model
  - Hyperparameter documentation
  - ONNX export and quantization procedures
  - How to add new document types (step-by-step guide)
  - Performance benchmarks (accuracy, speed, model sizes)
  **Validation**: Both files exist, contain Mermaid diagrams, and are well-structured markdown.

- [ ] **Task 11.2: Run full test suite and final validation**
  Execute all tests and quality checks:
  ```bash
  uv run ruff check .
  uv run pytest tests/ -v --tb=short
  uv run pytest tests/integration/ -v -m integration
  uv run pytest tests/e2e/ -v -m e2e
  ```
  Fix any failures.
  Run the demo app:
  ```bash
  uv run python -m mobile_app.src.app demo
  uv run python -m mobile_app.src.app info
  ```
  **Validation**: ALL tests pass. Ruff reports no errors. Demo app produces valid JSON output for all document types.

- [ ] **Task 11.3: Create comprehensive README.md**
  Create/update `README.md` at project root with:
  - Project title and description
  - Architecture overview diagram (Mermaid)
  - Quick start: `uv sync && uv run python -m mobile_app.src.app demo`
  - Full setup instructions (all dependency groups)
  - How to train models from scratch
  - How to run tests
  - Project structure explanation
  - Supported document types with example outputs
  - Team member roles (filled in)
  - Technology stack (ONNX, timm, transformers, RapidOCR)
  - License note
  **Validation**: README renders correctly in markdown preview.
