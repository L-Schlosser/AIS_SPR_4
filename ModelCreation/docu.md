# Document Information Extraction — Model Documentation

Offline NER (Named-Entity Recognition) pipeline that extracts structured fields
from OCR text of German/Austrian documents and runs **fully on-device** in a
Flutter app via ONNX Runtime.

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [The six document types & their fields](#2-the-six-document-types--their-fields)
3. [How the model works (the ML part)](#3-how-the-model-works-the-ml-part)
4. [The full architecture (end-to-end)](#4-the-full-architecture-end-to-end)
5. [Project structure](#5-project-structure)
6. [Setup / installation](#6-setup--installation)
7. [How to train, export, test (the pipeline)](#7-how-to-train-export-test-the-pipeline)
8. [Configuration reference (`config.yaml`)](#8-configuration-reference-configyaml)
9. [Using it in Flutter](#9-using-it-in-flutter)
10. [How to add or change a field](#10-how-to-add-or-change-a-field)
11. [Performance](#11-performance)
12. [Design decisions & FAQ](#12-design-decisions--faq)

---

## 1. What this is

Your app already **classifies** a scanned document into one of six types
(`invoice`, `receipt`, `doctor_note`, `care_leave`, `delivery_note`,
`master_data`). This project is the **next step**: given that document type plus
the raw **OCR text**, it returns a structured JSON of the important fields with
**German UI keys**, e.g.:

```json
{
  "firma": "Elektro Köck GmbH",
  "adresse": "4020 Linz Landstraße 42",
  "uid": "ATU45221087",
  "rechnungsnummer": "RE-2024-0815",
  "datum": "28.12.2026",
  "gesamtbetrag": "1440.00",
  "waehrung": "EUR"
}
```

Key properties:

- **Offline / on-device** — no server, no internet, runs on the phone.
- **Small & fast** — ~105 MB (INT8), ~40 ms per document on CPU.
- **Closed schema** — 39 predefined fields. It finds *values* freely in the
  text but only assigns them to known *field names*, which keeps the JSON
  output stable for the UI.
- **Hybrid** — a German NER model for fuzzy/semantic fields + a German regex
  layer for deterministic fields (dates, amounts, IBAN, …).
- **Two synced implementations** — a Python reference (`inference.py`) for
  training/testing, and the identical logic in Dart (`ml_service_newNER.dart`)
  for the app.

---

## 2. The six document types & their fields

The model knows **39 internal fields** (each returned under a German display
key). Only the fields relevant to the detected document type are returned. This
is configured in `config.yaml → fields_by_type` and `field_display_de`.

| Document type | Returned fields (German key) |
|---|---|
| **invoice** (Rechnung) | firma, adresse, uid, kunde, kundenadresse, kundennummer, datum, lieferdatum, faelligkeitsdatum, rechnungsnummer, bestellnummer, nettobetrag, mwst_betrag, mwst_satz, gesamtbetrag, waehrung, iban, bic, email, telefon |
| **receipt** (Kassenbon) | firma, adresse, uid, datum, uhrzeit, belegnummer, nettobetrag, mwst_betrag, mwst_satz, gesamtbetrag, waehrung, zahlungsart, telefon |
| **doctor_note** (Arbeitsunfähigkeitsmeldung) | patient, geburtsdatum, versicherungstraeger, versicherungsnummer, adresse, arzt, arztadresse, startdatum, enddatum, ausstellungsdatum, diagnose |
| **care_leave** (Pflegefreistellung) | patient, geburtsdatum, versicherungstraeger, versicherungsnummer, adresse, arzt, arztadresse, startdatum, enddatum, ausstellungsdatum, diagnose |
| **delivery_note** (Lieferschein) | firma, adresse, kunde, kundenadresse, kundennummer, datum, lieferdatum, lieferscheinnummer, bestellnummer, uid |
| **master_data** (Stammdaten) | firma, adresse, telefon, fax, email, webseite, ansprechpartner, iban, bic, bank, uid, firmenbuchnummer |

The field set is grounded in real Austrian standards (e.g. §11 UStG mandatory
invoice fields, the ÖGK Arbeitsunfähigkeitsmeldung layout, common Lieferschein
conventions).

---

## 3. How the model works (the ML part)

### 3.1 Token classification with BIO tagging

The model does **not** generate free text like a chatbot. It does something
simpler and far more reliable: it **labels every word** of the OCR text.

For each word it answers: *"Is this word part of a field I care about — and if
so, which field, and is it the start or a continuation?"* This is the **BIO
scheme**:

- `B-total` → **B**eginning of a "total amount" field
- `I-total` → **I**nside (continuation) of the same field
- `O` → **O**utside (not a field)

Example:

```
Gesamtbetrag   :     1.440,00     EUR
   O           O     B-total      B-currency
```

Every word gets exactly **one** of **79 labels** = `O` + (`B-` and `I-` for each
of the 39 fields). The "fields" are simply this fixed label list.

### 3.2 The neural network

- **Encoder:** `deepset/gbert-base` — a German BERT. It turns each token into a
  context-aware vector (it "understands" the surrounding words).
- **Classification head:** a single linear layer mapping each token vector to 79
  scores. The highest score = the predicted label; a softmax gives a confidence.

Because BERT works on **subword pieces** (WordPiece), a word like
`Gesamtbetrag` may split into `Gesamt` + `##betrag`. Each piece is converted to a
numeric ID via the vocabulary (`vocab.txt`).

### 3.3 How it is trained on the fields

Training = showing the model many examples where the correct label of every word
is already known, and adjusting the weights until predictions match.

1. **Generate labeled examples** (`synthetic_data.py`). Since there is no real
   labeled dataset, we synthesize realistic Austrian documents — ÖGK doctor
   notes, Billa/Spar receipts, §11-UStG invoices, Lieferscheine, Stammdaten —
   with **OCR-style noise** (blank lines, spacing jitter, character corruption,
   and dates written with spaces like `28. 12. 2026`). Each sample is a list of
   words + the correct BIO tag per word:

   ```python
   words = ["RECHNUNG", "Elektro", "Köck", "GmbH", "Gesamtbetrag", "1.440,00", "EUR"]
   tags  = ["O", "B-company", "I-company", "I-company", "O", "B-total", "B-currency"]
   ```

2. **Tokenize & align** (`download_model.py`). Words are split into subword
   pieces; each word's label is copied to its pieces; special tokens
   (`[CLS]`, `[SEP]`, padding) get the ignore label `-100`.

3. **Fine-tune.** The model predicts labels, a **cross-entropy loss** compares
   them to the truth, and **backpropagation** updates the weights. We train for
   several epochs over ~7,000 samples (1,200 per type × 6 types).

4. **Evaluate.** A held-out split reports token accuracy and entity-level
   precision/recall/F1.

**Important:** you teach it by *example*, not by hand-written rules. To support a
new field you add labeled examples for it and retrain.

### 3.4 How extraction works at runtime

1. **Input:** OCR text + the document type from your classifier.
2. **Tokenize** the text (same vocabulary) → `[CLS] … [SEP]`.
3. **Run the model** → for each token, 79 scores → argmax = BIO label + softmax
   confidence.
4. **Reassemble:** merge subword labels back to words, then join consecutive
   `B-/I-` of the same type into one entity. For each field keep the **single
   best** entity (highest confidence, then longest) — so a value appearing
   multiple times yields one clean result, not a concatenation.
5. **Output:** map internal field names → German keys, filtered to the fields
   allowed for the document type.

Worked example — OCR text `Arbeitsunfähig von: 08.07.2024`:

```
Arbeitsunfähig=O   von=O   :=O   08.07.2024=B-start_date
→ start_date = "08.07.2024" → "startdatum": "08.07.2024"
```

The model learned from many examples that *a date right after "Arbeitsunfähig
von" is a start date*, so it generalizes to layouts it never saw verbatim.

### 3.5 Models considered (and why an encoder/NER model was chosen)

Several model families were evaluated before settling on a German BERT token
classifier. The deciding constraints were: **text-only input** (OCR provides
text, not bounding boxes), **fully offline on mobile CPU**, **low latency**, and
**a stable, valid JSON contract**.

| Model family | Type | Verdict | Reason |
|---|---|---|---|
| **LayoutLMv3 / LayoutXLM** | Multimodal (text + image + bounding boxes) | ❌ Rejected | Needs the page image and per-token bounding boxes, which the OCR pipeline does not provide; 500 MB+ FP32; slow on mobile CPU. |
| **Donut / CORD-V2** | Vision encoder + autoregressive text decoder | ❌ Rejected | Seq2seq generation → prompt sensitivity, invalid JSON, slow token-by-token decoding. |
| **T5 / mT5 / FLAN-T5 / BART** | Generative seq2seq | ❌ Rejected | Previously deployed and failed: slow, unstable JSON, instruction echo, repetition. |
| **Pretrained invoice NER** (`KrisMadeMe/distilbert-…-invoice`) | Token classifier | ⚠️ Not usable | Gated Hugging Face repo, no authenticated access; also invoice-only. |
| **General NER** (`dslim/distilbert-NER`) | Token classifier | ⚠️ Partial | Only PER/ORG/LOC/MISC labels — cannot represent `total`, `iban`, `vat_id`, `diagnosis`, etc. Useful as a concept only. |
| **MiniLM / DistilBERT / MobileBERT token classification** | Encoder + classification head | ✅ Selected family | Single forward pass (`input_ids` + `attention_mask` → `logits` → BIO → JSON); no decoder, no prompts, no generation loop; small and quantizable; mature ONNX export. |

**Encoder size/speed trade-offs that were weighed** (INT8 estimates, 512 tokens):

| Encoder | Params | FP32 | INT8 | CPU latency |
|---|---|---|---|---|
| MiniLM-L6 | 22M | ~90 MB | ~23 MB | ~60–120 ms |
| MobileBERT | 25M | ~100 MB | ~25 MB | ~80–150 ms |
| DistilBERT | 66M | ~260 MB | ~67 MB | ~120–250 ms |

**Final choice: `deepset/gbert-base`** (a German BERT) with a custom
token-classification head.

- The first working version used **multilingual DistilBERT**; it was upgraded to
  **gbert-base** because a German-specific model understands German/Austrian
  documents substantially better, while still staying small and fast
  (~105 MB INT8, ~40 ms).
- For maximum quality at the cost of speed/size, `deepset/gbert-large`
  (~340 MB INT8) is a drop-in alternative — just change `model.base_model` in
  `config.yaml` and retrain.

**Why an encoder/NER model wins here:** it is a single, deterministic forward
pass that emits one label per token. That gives a guaranteed, schema-valid output
(no hallucinated keys, no malformed JSON), tiny size, and millisecond latency —
exactly what an offline mobile app needs. The generative models can invent
arbitrary fields but trade away all of those guarantees.

---

## 4. The full architecture (end-to-end)

The deployed system is a **hybrid** of the NER model and a German regex layer.
This is what actually produces the final field map (see `entity_decoder.py` and
the mirrored logic in `ml_service_newNER.dart`):

```
OCR raw_text + document_type
        │
        ▼
   WordPiece tokenizer  ──►  [CLS] … [SEP]  (token IDs)
        │
        ▼
   gbert-base encoder + classification head (ONNX, INT8)
        │
        ▼
   logits [tokens × 79]  ──►  argmax → BIO tag + confidence per token
        │
        ▼
   BIO aggregation → best entity per field   (the NER signal)
        │
        ├───────────────┬───────────────────────────────┐
        ▼               ▼                                 ▼
   Regex layer     Merge per field                  Normalization
  (German anchors)  rule:                            - amounts → 1234.50
                    • deterministic fields →          - dates  → 28.12.2026
                      REGEX primary, NER fallback     - € → EUR
                    • semantic fields →
                      NER primary, regex fallback
                    • address fields →
                      take the longer/more complete
        │
        ▼
   Map internal → German keys, filter to document type
        │
        ▼
   { "firma": "...", "gesamtbetrag": "...", ... }  →  Flutter UI
```

**Why hybrid?**

- The NER model is great at *semantic* fields (which name is the doctor? which
  is the company?) but, trained only on synthetic data, isn't perfect.
- Regex is rock-solid for *structured* fields (an IBAN is always an IBAN; a date
  after "Ausstellungsdatum" is the issue date) but blind to meaning.
- Combining them: **NER for meaning, rules for structure** → accurate, reliable
  output without a large/slow generative model.

**Field routing (`entity_decoder.py`):**

- **Regex-primary** (deterministic): all dates, `time`, `total`, `subtotal`,
  `vat`, `vat_rate`, `invoice_number`, `receipt_number`, `order_number`,
  `customer_number`, `delivery_number`, `currency`, `email`, `phone`, `fax`,
  `website`, `iban`, `bic`, `vat_id`, `company_register_number`,
  `insurance_number`, `payment_method`, `insurer`, `bank_name`.
- **NER-primary** (semantic): `company`, `customer`, `patient`, `doctor`,
  `diagnosis`, `contact_person`.
- **Address fields** (`address`, `customer_address`, `doctor_address`): take the
  longer of NER vs regex (NER often stops at a line break; regex captures the
  full line).

---

## 5. Project structure

```
ModelCreation/
├── config.yaml              # Single source of truth: labels, fields, hyperparams
├── synthetic_data.py        # Generates realistic Austrian training samples (BIO)
├── download_model.py        # Downloads gbert-base + fine-tunes on synthetic data
├── export_onnx.py           # PyTorch → ONNX + INT8 quantization
├── entity_decoder.py        # BIO → JSON, regex layer, normalization (Python ref)
├── inference.py             # End-to-end Python inference (load ONNX, extract)
├── benchmark.py             # Latency / size / RAM / throughput measurement
├── model_creation.py        # One-shot pipeline: download → export → benchmark
├── ml_service_newNER.dart   # The SAME pipeline in Dart for the Flutter app
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── pytorch/             # Fine-tuned PyTorch model + tokenizer + meta
│   └── onnx/
│       ├── model.onnx       # ← deployable INT8 model (copy to Flutter)
│       ├── model_meta.json  # ← labels, fields_by_type, display map (copy too)
│       └── tokenizer/
│           └── vocab.txt    # ← WordPiece vocabulary (copy too)
│
└── test_*.txt               # Example OCR texts for quick testing
```

---

## 6. Setup / installation

Requires Python 3.10+ (the project was developed with a conda env named `SPR`).
A CUDA GPU makes training take ~12 min; CPU also works but is slower.

```bash
# create / activate an environment, then:
pip install -r requirements.txt
```

`requirements.txt` includes `torch`, `transformers`, `optimum[onnxruntime]`,
`onnx`, `onnxruntime`, `numpy`, `pyyaml`, `datasets`, `accelerate`.

> Note for conda users: prefix commands with
> `conda run --no-capture-output -n <env> ...` so output streams live.

---

## 7. How to train, export, test (the pipeline)

### Option A — one command (everything)

```bash
python model_creation.py
```

This runs `download_model.py` → `export_onnx.py` → `benchmark.py` →
a sample `inference.py`.

### Option B — step by step

**1. Train** (download gbert-base + fine-tune on synthetic data):

```bash
python download_model.py
# outputs: models/pytorch/  (model, tokenizer, model_meta.json)
```

Options: `--skip-finetune` (only download the base model),
`--config <path>`.

**2. Export to ONNX + INT8 quantize:**

```bash
python export_onnx.py
# outputs: models/onnx/model.onnx (~105 MB), tokenizer/, model_meta.json
```

Options: `--no-quantize` (keep FP32), `--use-optimum` (alternative exporter).

**3. Benchmark** (size, latency, RAM):

```bash
python benchmark.py --output benchmark_results.json
```

**4. Test on a real document:**

```bash
python inference.py --document-type doctor_note --text-file test_doctor_note.txt
python inference.py --document-type invoice --text "RECHNUNG ... Gesamtbetrag: 1.440,00 EUR"
```

`--document-type` must be one of the six types; provide `--text` or
`--text-file`.

### Inspect the synthetic data

```bash
python synthetic_data.py     # prints sample documents with their BIO tags
```

---

## 8. Configuration reference (`config.yaml`)

Everything is driven by this one file.

| Section | Purpose |
|---|---|
| `model.base_model` | HF model id. `deepset/gbert-base` (default, fast). Switch to `deepset/gbert-large` for max quality (~340 MB INT8, slower). |
| `model.max_length` | Max tokens per document (384). Longer = slower. |
| `model.synthetic_epochs` | Training passes (6). |
| `model.synthetic_samples_per_type` | Synthetic samples per document type (1200). |
| `model.learning_rate` / `batch_size` / `warmup_ratio` / `weight_decay` | Training hyperparameters. |
| `field_display_de` | Maps each internal field → German UI key. |
| `document_types` | The six supported types. |
| `fields_by_type` | Which fields are returned for each document type. |
| `labels` | The full BIO label list (79 entries). **Order matters** (defines label IDs). |
| `onnx.quantize_int8` | Whether to INT8-quantize (true → ~105 MB). |
| `inference.min_entity_score` | Minimum NER confidence to accept a span (0.35). |
| `inference.enable_regex_fallback` | Enable the regex layer (true). |

---

## 9. Using it in Flutter

### 9.1 Copy the assets

After training/export, copy these four files into the app:

| Source | Flutter destination |
|---|---|
| `models/onnx/model.onnx` | `assets/models/model.onnx` |
| `models/onnx/model_meta.json` | `assets/models/model_meta.json` |
| `models/onnx/tokenizer/vocab.txt` | `assets/models/tokenizer/vocab.txt` |
| `ml_service_newNER.dart` | into your `lib/` |

Declare the assets in `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/model.onnx
    - assets/models/model_meta.json
    - assets/models/tokenizer/vocab.txt
```

Dependency: `onnxruntime` (Dart/Flutter package).

### 9.2 Use the service

```dart
print('Extract features');
final nerExtractor = MLServiceNewNER();
await nerExtractor.initialize();

// documentType from your classifier: invoice | receipt | doctor_note | ...
final extractedInfos = await nerExtractor.extract(
  classificationResult.documentType,
  extractedText,
);

classificationResult.infos.addAll(extractedInfos.felder);

nerExtractor.dispose();
```

`extractedInfos.felder` is a `Map<String, String>` of German keys → values,
e.g. `{ "firma": "Billa AG", "gesamtbetrag": "8.84", "waehrung": "EUR" }`.

> The Dart service reads the special token IDs (`[CLS]`, `[SEP]`, `[PAD]`,
> `[UNK]`) **from `vocab.txt`**, so it stays correct for any BERT model — just
> keep all four files from the same export together.

---

## 10. How to add or change a field

The field set is fixed at training time, so adding a field means a retrain. Steps:

1. **`config.yaml`**
   - Add `B-<field>` and `I-<field>` to `labels`.
   - Add `<field>` to the relevant `fields_by_type` entries.
   - Add `<field>: <german_key>` to `field_display_de`.
2. **`synthetic_data.py`** — add the field (with a realistic value generator and
   template context) so the model sees labeled examples of it.
3. **`entity_decoder.py`** — add a regex under `_PATTERNS` (and put the field in
   `_REGEX_PRIMARY`, `_ADDRESS_FIELDS`, or `_DATE_FIELDS` as appropriate).
4. **`ml_service_newNER.dart`** — mirror the regex in `_patternsFor`, and the
   sets `_regexPrimary` / `_addressFields` / `_dateFields`, plus
   `_defaultFieldDisplayDe`.
5. **Retrain & re-export:** `python download_model.py && python export_onnx.py`.
6. Copy the new `model.onnx`, `model_meta.json`, `vocab.txt`, and the Dart file
   into Flutter.

> To remove a field from the UI without retraining, just delete it from the
> relevant `fields_by_type` list and re-export the metadata.

---

## 11. Performance

Measured with `benchmark.py` on a development CPU (INT8 model, 384 max sequence):

| Metric | Value |
|---|---|
| Model size (INT8) | ~105 MB |
| Latency (mean) | ~42 ms |
| Latency (p95) | ~44 ms |
| Base model | deepset/gbert-base |
| Labels | 79 (39 fields) |

Well within a mobile budget (the project targets <500 MB and low latency). On a
mid-range Android phone, expect roughly 100–400 ms per document.

---

## 12. Design decisions & FAQ

**Why a token classifier and not a generative model (LLM / Donut / T5)?**
Generative models can invent arbitrary key–value pairs but are large, slow, and
produce unstable/invalid JSON (prompt sensitivity, repetition). For an on-device
app needing a stable JSON contract and low latency, token classification + rules
is far more robust. (See `ARCHITECTURE_REPORT.md` for the full comparison.)

**Can the model invent its own field names?**
No. It only outputs the 39 predefined fields (closed schema). This is by design —
it guarantees a stable, predictable structure for the UI. Add fields deliberately
via the steps in section 10.

**Why synthetic training data?**
No labeled real dataset exists for these specific Austrian documents and field
schema. The synthetic generator models the real layouts and adds OCR noise so the
model generalizes. The single biggest quality lever is making synthetic templates
match the documents you actually scan — when you hit a failure case, add that
layout and retrain.

**Why is there both Python and Dart code?**
Python (`inference.py` + `entity_decoder.py`) is the reference for training and
testing. Dart (`ml_service_newNER.dart`) is the exact same logic for on-device
use, so results match. Keep them in sync when you change patterns/fields.

**Dates with spaces (e.g. `28. 12. 2026`)?**
Supported. The date patterns allow spaces around separators, and all date fields
are normalized on output to `28.12.2026`.

**How are duplicate values handled?**
Per field, only the single best entity is kept (highest confidence, then
longest) — no concatenation of repeated mentions.
```
