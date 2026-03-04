# Edge-AI Based Document Processing for Mobile Applications

## 1. Project Description

This project addresses the challenge of performing document classification and structured information extraction directly on mobile devices. While large-scale language models demonstrate strong performance in document understanding tasks, their computational demands typically prevent deployment on resource-constrained hardware such as smartphones.

The target use case is the **BMD Go** mobile application, where users can upload heterogeneous document types, including but not limited to:

* Medical visit confirmations
* Business travel expense receipts
* Delivery and logistics documents

The overarching product goal is to enable a **unified document upload interface**, independent of the document type. To achieve this, the system performs automated document recognition and extracts relevant key fields locally on the device.

The solution follows an Edge-AI approach, prioritizing:

* On-device inference
* Low latency
* Data privacy
* Modular extensibility

Since the input consists of camera images or scans, optical character recognition (OCR) is required. However, OCR is treated as an optional extension, whereas the main focus lies on local document classification and targeted field extraction.

---

## 2. System Objectives

* Design a pipeline for on-device document analysis
* Implement document type classification
* Implement structured data extraction based on document category
* Produce standardized machine-readable outputs
* Maintain a modular architecture to allow future expansion (e.g., OCR integration)
* Avoid dependency on cloud-based inference services

---

## 3. Processing Pipeline

The conceptual workflow is defined as:

```
Image Input  
   ↓  
Preprocessing  
   ↓  
Document Classification  
   ↓  
Document-Specific Field Extraction  
   ↓  
Structured Output (JSON)
```

Optional extension:

```
Image → OCR → Text → Classification → Extraction
```

---

## 4. Repository Structure

```
.
├── data/                 # Sample data and output schemas
│   ├── samples/
│   └── schemas/
│
├── edge-model/           # ML models and inference logic
│   ├── classification/
│   ├── extraction/
│   └── inference/
│
├── mobile-app/           # Mobile client integration layer
│   └── src/
│
├── ocr/                  # OCR module (optional extension)
│
├── api/                  # Interfaces between application and ML pipeline
│
├── scripts/              # Utility and automation scripts
│
├── tests/                # Unit and integration tests
│
├── docs/                 # Technical documentation
│   ├── architecture.md
│   └── model_pipeline.md
│
├── README.md
└── .gitignore
```

---

## 5. Core Components

### 5.1 Document Classification

Identifies the category of the uploaded document (e.g., receipt, medical confirmation, delivery note).

### 5.2 Information Extraction

Extracts predefined key fields based on the detected document type, such as:

* Date and time
* Monetary values
* Identifiers
* Issuing entity

### 5.3 Optical Character Recognition (Extension)

Transforms visual document content into textual representations for downstream processing.


---

## 6. Data Management

* No real or personally identifiable user data is included in this repository
* Only anonymized or synthetic samples are stored
* Raw datasets must be excluded from version control

---

## 7. Testing and Validation

The `tests/` directory contains:

* Classification validation tests
* Field extraction tests
* End-to-end inference tests

All outputs are validated against the schemas defined in `data/schemas/`.

---

## 8. Intended Users and Roles

| Role                       | Responsibility                     |
| -------------------------- | ---------------------------------- |
| Machine Learning Engineers | Model development and optimization |
| Mobile Developers          | Client-side integration            |
| OCR Developers             | Text recognition module            |
| Quality Assurance          | Testing and evaluation             |

---

## 9. Future Work

* Extension to additional document categories
* Performance optimization for mobile hardware
* Multilingual document support
* Full OCR integration
* Incremental model updates

---

## 10. License

This project is intended for research and educational purposes.
