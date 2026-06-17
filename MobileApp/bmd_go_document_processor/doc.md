# BMD Go Document Processor

A Flutter mobile app for **automatic document capture and understanding**. You take a
photo (or several), upload an image/PDF, and the app turns it into structured data:
it reads the text, decides what kind of document it is, and pulls out the important
fields — all **fully on-device**, with no server or internet connection required.

The whole pipeline is built around small, exported ONNX / ML Kit models so that it
works offline and keeps document data private.

---

## What the app does

The processing happens in three stages, chained together when the user presses
**"Verarbeiten"** (Process):

```
Image / PDF ──▶ 1. OCR ──▶ raw text ──▶ 2. Classification ──▶ doc type
                                                  │
                                                  ▼
                                          3. NER extraction ──▶ structured fields
```

1. **OCR** – read all text out of the page(s).
2. **Classification** – decide the document type, one of:
   `invoice`, `receipt`, `doctor_note`, `care_leave`, `delivery_note`, `master_data`.
3. **NER (Named Entity Recognition)** – extract the relevant fields for that
   specific document type (company, total, IBAN, patient, dates, …).

The result is shown on a dedicated results screen with the detected type and a
list of all extracted fields (labelled in German).

---

## Features

- **Multiple input methods**
  - Take photos with the camera, one page at a time (multi-page scan).
  - Pick multiple images from the gallery.
  - Upload a file (`jpg`, `png`, `pdf`, `doc`, `docx`).
- **Multi-page documents** – images and multi-page PDFs are both supported; PDFs
  are rendered page by page and OCR'd individually.
- **Thumbnail preview** with per-page deletion, page counter, and PDF page-count
  badges before processing.
- **Six document categories** automatically classified.
- **Field extraction tailored per document type** – an invoice gets invoice fields,
  a doctor's note gets medical fields, etc.
- **Fully offline / on-device** – OCR via ML Kit, classification and NER via ONNX
  Runtime. Nothing leaves the phone.
- **German UI** with English-friendly field keys mapped to German display names.

---

## Project structure

```
lib/
├── main.dart                          # UI, navigation, orchestrates the 3-stage pipeline
├── services/
│   ├── image_picker_service.dart      # Camera / gallery / file picking
│   ├── ml_service_ocr.dart            # Stage 1: OCR (ML Kit) + PDF rendering
│   ├── ml_service_classifier.dart     # Stage 2: document-type classification (ONNX)
│   ├── ml_service_ner.dart            # Stage 3: field extraction (ONNX BERT + regex)
│   └── old_models/                    # Abandoned experiments (see end of doc)
├── screens/
│   └── classification_results_screen.dart  # Shows type + extracted fields
└── utils/
    └── doc_type_translation.dart      # English doc type -> German label
assets/
└── models/
    ├── classifier_v2.onnx             # Classification model
    ├── NER_Model/                     # NER model + tokenizer + metadata
    └── old/                           # Assets for the abandoned models
```

---

## How it works (the important parts)

### `main.dart` — UI and pipeline orchestration

This is the heart of the app. It holds the upload screen and wires the three
services together.

- `_UploadScreenState` manages the list of selected files (`_selectedFiles`),
  shows a thumbnail grid, and provides camera / gallery / upload entry points.
- PDFs are detected with a simple extension check (`_isPdf`) and previewed inline
  using `pdfx`. Page counts are cached (`_pdfPageCountCache`) so the UI doesn't
  re-open a PDF every rebuild.
- The **"Verarbeiten"** button runs the whole pipeline in order and times each
  stage (handy for debugging performance):

```494:520:lib/main.dart
final orcMlService = OCRMLService();
String extractedText;
if(_isPdf(_selectedFiles.first)){
  extractedText = await orcMlService.processPdf(_selectedFiles.first);
} else {
  extractedText = await orcMlService.processImages(_selectedFiles);
}

//CLASSIFICATION
final classifierService = MLServiceClassifier();
await classifierService.initialize();
final classificationResult = await classifierService.classify(extractedText);

//____ actual NER model:
final nerExtractor = MLService_NER();
await nerExtractor.initialize();
final extractedInfos = await nerExtractor.extract(
  classificationResult.documentType,
  extractedText,
);
classificationResult.infos.addAll(extractedInfos.felder);
```

The extracted fields are merged into the classification result and the user is
pushed to `ClassificationResultsScreen`.

> Why this design: each stage is an independent service with its own model. They
> are decoupled, so any one model can be swapped out without touching the others —
> which is exactly what happened during development (several extractors were tried,
> see the bottom of this doc).

### `image_picker_service.dart` — getting the document in

A thin wrapper around `image_picker` and `file_picker`:

- `takePhotoWithCamera()` – single rear-camera photo at high quality.
- `pickMultipleImagesForDocument()` – multi-select from the gallery.
- `pickDocumentFile()` – file manager picker restricted to common document types.

It deliberately stays dumb (just returns `File`s); all multi-page logic lives in
the UI.

### `ml_service_ocr.dart` — Stage 1: OCR

Uses **Google ML Kit Text Recognition** (on-device, free, no network).

- `processImages()` runs OCR over each image and concatenates the text.
- `processPdf()` first **renders each PDF page to a PNG** with `pdfx`, caps the
  resolution (`maxDimension = 1200`) for speed, OCRs each page, then **deletes the
  temp images** to save disk/IO.

It also defines `OCRClassificationResult`, the shared data object that carries the
document type, confidence, and the `infos` map of extracted fields through the
rest of the pipeline.

> Why cap resolution + delete temp files: PDF pages can render huge; capping size
> keeps OCR fast and avoids filling temp storage on mobile.

### `ml_service_classifier.dart` — Stage 2: classification

Loads `assets/models/classifier_v2.onnx` via **ONNX Runtime** and predicts the
document type from the raw OCR text.

- The model takes the **raw text string directly** as input (`[1, 1]` string
  tensor) — tokenization is baked into the exported model, so Dart just passes the
  text in.
- Output 1 is the predicted label string; Output 2 is a probability map, from
  which the max value becomes the confidence score.
- Returns an `OCRClassificationResult` with the type and confidence (fields are
  still empty at this point — they get filled in stage 3).

> Why a text-in ONNX model: keeping tokenization inside the model means the Dart
> side has zero preprocessing to maintain for classification — it's the simplest,
> most robust option.

### `ml_service_ner.dart` — Stage 3: field extraction (the most complex part)

This is the largest service. It runs a **German BERT token-classification model**
(`deepset/gbert-base`, exported to ONNX) and combines it with **regex fallbacks**
to extract concrete field values. The model metadata lives in
`assets/models/NER_Model/model_meta.json`.

The `extract(documentType, text)` flow:

1. **Type-aware filtering** – `model_meta.json` lists `fields_by_type`, so only the
   fields relevant to the detected document type are considered (an invoice won't
   try to extract a `diagnosis`).
2. **Tokenization in pure Dart** – the service re-implements BERT preprocessing:
   - `_basicTokenize` – whitespace/punctuation split, keeping character offsets so
     extracted spans can be mapped back to the original text.
   - `_wordPieceTokenize` – WordPiece subword tokenization against `vocab.txt`.
   - Special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`) are looked up from the
     actual vocab rather than hardcoded, so it stays correct across BERT variants.
   - Input is `[CLS] … [SEP]`, padded to `max_length = 384`, with an attention mask.
3. **ONNX inference** – feeds `input_ids` + `attention_mask`, gets per-token logits.
4. **Decoding** – argmax + softmax per token, then **BIO tag decoding** (`B-`/`I-`/
   `O`) to merge tokens into entity spans, tracking an averaged confidence per span.
   Low-confidence spans (`< 0.35`) are dropped.
5. **NER + regex merge** – for each field it picks the best value using a strategy
   per field category:
   - **Deterministic fields** (dates, totals, IBAN, email, invoice numbers, …) →
     **regex is primary** (more reliable for structured patterns).
   - **Address fields** → pick whichever (NER vs regex) is *longer/more complete*.
   - **Everything else** → prefer NER, fall back to regex.
   The regex patterns are German/Austrian-specific (e.g. `ATU` VAT IDs, `FN`
   company register numbers, Austrian IBANs, insurer names like `ÖGK`/`SVS`).
6. **Normalization** – amounts and dates are cleaned (e.g. `28. 12. 2026` →
   `28.12.2026`, `€` → `EUR`).
7. **German display keys** – internal keys (`company`, `total`) are mapped to
   German labels (`firma`, `gesamtbetrag`) via `field_display_de` for the UI.

> Why a hybrid NER + regex approach: the BERT model is good at *finding* entities
> in messy OCR text and understanding context, but structured values (IBANs, dates,
> amounts) are more reliably captured with precise regex. Combining both gives
> better, cleaner results than either alone — the model finds *where* things are,
> regex validates and normalizes *what* they are.

### `classification_results_screen.dart` — showing results

Displays the detected document type (translated to German, color-coded per type)
and renders every extracted field from the `infos` map as a labelled card. A
"Speichern" (Save) button exists but is not yet implemented.

### `doc_type_translation.dart`

A tiny lookup map turning the model's English type names into German UI labels
(e.g. `invoice` → `Rechnung`).

---

## Models & assets

| Asset | Purpose |
|-------|---------|
| `assets/models/classifier_v2.onnx` | Document-type classification |
| `assets/models/NER_Model/model.onnx` | German BERT NER model for field extraction |
| `assets/models/NER_Model/model_meta.json` | Labels, per-type field lists, German display names |
| `assets/models/NER_Model/tokenizer/vocab.txt` | WordPiece vocabulary for tokenization |

All active assets are registered in `pubspec.yaml`. Everything under
`assets/models/old/` belongs to abandoned experiments and is **not** bundled.

### Key dependencies (`pubspec.yaml`)

- `google_mlkit_text_recognition` – on-device OCR
- `onnxruntime` – runs the classifier and NER ONNX models
- `pdfx` / `flutter_pdfview` – render & preview PDFs
- `image_picker` / `file_picker` – camera, gallery, file selection
- `path_provider` – temp storage for rendered PDF pages

---

## Approaches that were tried but dropped

Several extraction strategies were experimented with before settling on the
BERT-NER + regex hybrid. They live in `lib/services/old_models/` (and their assets
in `assets/models/old/`) and are commented out in `main.dart`. A lot of the tries are already deleted, but we kept some which had working parts in it.

- **GLiNER** – a zero-shot NER model; flexible but too heavy/slow on-device.
- **Transformer / SmolLM / Qwen (GGUF) generative extractors** – LLM-based field
  extraction; too large and slow for mobile, and less reliable for structured fields.

These were replaced by the current approach because it is **smaller, faster, and
more accurate** for this fixed set of document types and fields.
