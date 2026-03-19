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

## 8. Intended Roles

| Contibuter                 | Role                       | Responsibility                     |
| -------------------------- | -------------------------- | ---------------------------------- |
| Celina Binder              |                            |                                    |
| Eichsteininger Natalie     |                            |                                    |
| Hysenlli Klevi             |                            |                                    |
| Schlosser Lorenz Johannes  |                            |                                    |
| Suchomel Raphael           |                            |                                    |

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








# Setup & Contribution Guide

## Contributors

* **Binder Celina Anna**
* **Eichsteininger Natalie**
* **Hysenlli Klevi**
* **Schlosser Lorenz Johannes**
* **Suchomel Raphael**

This guide explains how to set up the project, manage dependencies, and contribute in a structured way.

---

## Clone the repository

```bash
git clone https://github.com/L-Schlosser/AIS_SPR_4.git
cd AIS_SPR_4
```

---

## Project Dependencies & Environment Setup

This project uses **uv + pyproject.toml** for dependency management. Dependency groups allow installing only the components you need:

* **Base dependencies**: core functionality and scripts
* **OCR module**: Optical Character Recognition for document processing
* **Visualization tools**: plotting and analysis tools

This keeps environments **clean, conflict-free, and fast**, and simplifies collaboration.

---

### Installing Dependencies

**Base dependencies (recommended for general users):**

```bash
uv sync
```

**OCR module:**

```bash
uv sync --group ocr
```

**Visualization tools:**

```bash
uv sync --group viz
```

**Multiple groups at once:**

```bash
uv sync --group ocr --group viz
```

**All groups (use only if necessary):**

```bash
uv sync --all-groups
```

---

### Adding New Dependencies

Dependencies are tracked in **pyproject.toml**.

* Add a package to **base dependencies**:

```bash
uv add PACKAGE
```

* Add a package to a **specific group**:

```bash
uv add PACKAGE --group ocr
uv add PACKAGE --group viz
```

---

### Checking Installed Packages

* List all installed packages:

```bash
uv pip list
```

* View available dependency groups:

```bash
uv tree
```

---

### Python File Guidelines

* All Python code **must use `.py` files**
* Filenames should be **lowercase and descriptive**, e.g.:

```text
document_classifier.py
ocr_pipeline.py
data_loader.py
```

---

### Summary Table

| Task                        | Command                                 |
| --------------------------- | --------------------------------------- |
| Install base deps           | `uv sync`                               |
| Install OCR module          | `uv sync --group ocr`                   |
| Install visualization tools | `uv sync --group viz`                   |
| Install multiple groups     | `uv sync --group GROUP1 --group GROUP2` |
| Add package to base         | `uv add PACKAGE`                        |
| Add package to group        | `uv add PACKAGE --group GROUPNAME`      |
| List installed packages     | `uv pip list`                           |
| Show available groups       | `uv tree`                               |

> Python 3.12.11 is recommended for this project to ensure compatibility with all dependencies.

---

# Branch Instructions

### Check current branch

```bash
git branch
```

If not on **main**, switch:

```bash
git checkout main
```

---

### Pull the latest changes from origin

```bash
git pull origin main
```

---

### Create a new feature branch

```bash
git checkout -b <feature-name>
```

**Branch naming convention:**
`[FirstLetterOfFirstName][FirstLetterOfLastName]-[feature_added]`

Example:

```text
CB-preprocessing
AS-visualizations
```

---

### First push of a new branch

```bash
git push -u origin <feature-name>
```

> Required only for the first push of a branch.

---

### Subsequent pushes

```bash
git add .
git commit -m "message_in_lowercase_with_underscores"
git push
```

> **Commit messages** must be lowercase with underscores, e.g.:
> `add_document_classifier`
> `update_ocr_pipeline`

---

### Merging

1. Go to GitHub → **Pull Requests** → **New Pull Request**
2. Set:

```text
base: main <- compare: <your-feature-branch>
```

3. Review and resolve conflicts if necessary.

---

### Switching back to main

```bash
git fetch
git pull
git checkout main
git pull origin main
```
