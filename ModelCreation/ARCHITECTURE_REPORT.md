# Architecture Evaluation Report — Offline Document Information Extraction

**Date:** 2026-06-10  
**Target:** Flutter Android, CPU-only, ONNX Runtime Mobile, <1 s latency, <100 MB model

---

## 1. Requirements Summary

| Constraint | Value |
|---|---|
| Input | OCR `raw_text` + `document_type` from existing classifier |
| Output | Structured JSON fields per document type |
| Inference | Single-pass, no autoregressive decoding |
| Platform | Flutter Android, fully offline |
| Model format | ONNX INT8 (`model.onnx`) |

---

## 2. Models Considered

### 2.1 LayoutLMv3 / LayoutXLM — **REJECTED**

| Criterion | Assessment |
|---|---|
| Architecture | Multimodal encoder + token classification head |
| Strengths | State-of-the-art on FUNSD, SROIE, CORD with bounding boxes |
| Weaknesses | Requires image + per-token bounding boxes; 500 MB+ FP32; slow on mobile CPU |
| Mobile fit | Poor — needs layout tensors, large memory, complex Flutter preprocessing |

**Verdict:** Rejected. The existing pipeline provides OCR **text only**, not normalized bounding boxes. Adding layout extraction would duplicate OCR work and blow the 100 MB / 1 s budget.

### 2.2 Donut / CORD-V2 — **REJECTED**

| Criterion | Assessment |
|---|---|
| Architecture | Vision encoder + autoregressive text decoder |
| Weaknesses | Seq2seq generation, prompt sensitivity, invalid JSON, slow decoding |
| Mobile fit | Poor — same failure mode as the previous T5/decoder approach |

**Verdict:** Rejected per project constraints (no decoder, no token-by-token generation).

### 2.3 T5 / mT5 / FLAN-T5 / BART — **REJECTED**

Previously deployed and failed: slow, unstable JSON, instruction echo, repetition.

**Verdict:** Rejected.

### 2.4 MiniLM / DistilBERT / MobileBERT Token Classification — **SELECTED**

| Model | Params | FP32 Size | INT8 Est. | CPU Latency (512 tok) |
|---|---|---|---|---|
| MobileBERT | 25M | ~100 MB | ~25 MB | ~80–150 ms |
| **DistilBERT** | **66M** | **~260 MB** | **~67 MB** | **~120–250 ms** |
| MiniLM-L6 | 22M | ~90 MB | ~23 MB | ~60–120 ms |

**Selected: DistilBERT-base-uncased** with a custom token classification head.

**Rationale:**
- Best quality/speed trade-off for OCR-noisy text (66M params vs MobileBERT's 25M)
- Mature ONNX export path via Hugging Face Optimum
- INT8 quantization brings size to ~67 MB (under 100 MB target)
- Single forward pass: `input_ids` + `attention_mask` → `logits` → BIO tags → JSON
- No decoder, no prompts, no generation loop

### 2.5 Pretrained Invoice NER (KrisMadeMe/distilbert-base-uncased-invoice-trained) — **NOT USABLE**

Gated Hugging Face repository; requires authenticated access not available in this environment.

### 2.6 General NER (dslim/distilbert-NER) — **PARTIAL**

Labels: PER, ORG, LOC, MISC only. Cannot extract `total`, `invoice_number`, `iban`, `vat_id`, etc.

**Verdict:** Useful encoder initialization concept, but label schema mismatch. Used as architectural reference only; custom 43-label schema implemented instead.

---

## 3. Kaggle Dataset Analysis

**Dataset:** `suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning`

| Property | Finding |
|---|---|
| Total files | 3,482 |
| File types | `.jpg` only (3,482 images) + 10 `.db` thumbnails |
| JSON / CSV / TXT annotations | **None** |
| Token-level NER labels | **None** |
| Field-level labels | **None** |
| Categories | ADVE, HWDB, ICDAR, etc. (image folders only) |

**Conclusion:** This dataset contains scanned document **images only**. It is suitable for OCR or VLM fine-tuning, **not** for NER or field extraction training. **No training performed on this dataset.**

---

## 4. Training Strategy

| Step | Action |
|---|---|
| 1 | Evaluate pretrained models → none cover all 6 document types with correct labels |
| 2 | No labeled data available |
| 3 | Bootstrap with **synthetic OCR-style samples** (template-based BIO annotations) |
| 4 | Fine-tune DistilBERT token classifier for 2 epochs (~720 samples) |
| 5 | `entity_decoder.py` applies regex fallbacks for dates, amounts, IBAN, email, phone |

When real labeled data becomes available, run `download_model.py` with `fine_tune_synthetic: false` and point to a JSONL NER dataset.

---

## 5. Measured Performance (Development CPU)

Benchmark run via `benchmark.py` on Windows development machine (35-word sample, 512 max sequence):

| Metric | Measured | Target |
|---|---|---|
| Model size (INT8) | **63.75 MB** | <100 MB ✓ |
| Latency mean | **38.9 ms** | <1000 ms ✓ |
| Latency p95 | **51.1 ms** | <1000 ms ✓ |
| Throughput | **25.7 docs/s** | Adequate ✓ |
| RAM inference overhead | ~163 MB | Acceptable on modern phones |

**Expected Android (mid-range):** 60–150 ms per document — still well within the 1 s budget.

---

## 6. Final Architecture

```
OCR raw_text + document_type
        │
        ▼
   WordTokenizer (DistilBERT)
        │
        ▼
   DistilBERT Encoder  ──►  logits [seq × 43 labels]
        │                        │
        │                        ▼
        │              BIO tag per subword token
        │                        │
        │                        ▼
        │              Entity Aggregation (entity_decoder)
        │                        │
        │              ┌─────────┴─────────┐
        │              │ Regex fallbacks   │ (date, total, IBAN…)
        │              └─────────┬─────────┘
        │                        ▼
        └────────────►  JSON per document_type
```

**ONNX inputs:** `input_ids` (int64), `attention_mask` (int64)  
**ONNX output:** `logits` (float32, shape `[batch, seq, num_labels]`)

---

## 7. Deliverables

| File | Purpose |
|---|---|
| `config.yaml` | Labels, paths, hyperparameters |
| `download_model.py` | Download base model + optional synthetic fine-tune |
| `export_onnx.py` | Dynamic-axis ONNX export + INT8 quantization |
| `inference.py` | End-to-end Python inference |
| `entity_decoder.py` | BIO → JSON + regex fallbacks |
| `benchmark.py` | Latency, RAM, size, throughput |
| `flutter_example.md` | Production Flutter integration guide |
| `models/onnx/model.onnx` | Deployable INT8 model |
