# Demo Website — PRD

## Overview
Multi-page Streamlit demo website for presenting the Edge-AI Document Processing project.
Audience: university professors. Language: English. Runs locally via `streamlit run demo/Home.py`.
Shows: live document processing demo, architecture, model metrics, team info.

## Technical Constraints
- Python 3.12, dependency management via uv + pyproject.toml
- Streamlit multi-page app (built-in pages feature)
- All UI text in English
- Charts via plotly or matplotlib (embedded in Streamlit)
- Must work with existing trained models in edge_model/
- Run `uv run ruff check .` after each task
- Run `uv run pytest tests/unit/ -x` after each task to ensure nothing breaks
- Demo dependency group in pyproject.toml: streamlit, plotly, transformers

## Existing Code to Reuse
- `edge_model/inference/pipeline.py` — DocumentPipeline.from_config("config.yaml") for processing
- `api/models.py` — ProcessingResult, DocumentType enum
- `api/service.py` — DocumentService for convenience methods
- `scripts/generate_samples.py` — _generate_one_arztbesuch(), _generate_one_reisekosten(), _generate_one_lieferschein()
- `edge_model/classification/validate.py` — classifier validation logic
- `edge_model/extraction/labels.py` — LABEL_SETS, BIO tag definitions
- `mobile_app/src/model_manager.py` — ModelManager for model info/sizes
- `data/schemas/` — JSON schemas per document type
- `docs/architecture.md` — architecture content to display
- `results/*.json` — example JSON outputs

## Commit Convention
- Lowercase with underscores: `add_demo_home_page`
- One commit per completed task

---

## Tasks

### Phase 1: Setup & Dependencies

- [x] **Task 1.1: Update pyproject.toml demo dependencies**
  Update the `demo` dependency group in pyproject.toml:
  ```toml
  demo = [
      "streamlit>=1.30.0",
      "plotly>=5.18.0",
      "transformers>=4.38.0",
  ]
  ```
  Run `uv sync --group demo --group ocr` to install.
  Delete the old `demo_app.py` file from the project root (it's broken/replaced).
  Verify: `uv run python -c "import streamlit; import plotly; print('OK')"` succeeds.

- [x] **Task 1.2: Create demo directory structure**
  Create the Streamlit multi-page app structure:
  ```
  demo/
  ├── Home.py
  ├── pages/
  │   ├── 1_Live_Demo.py
  │   ├── 2_Architecture.py
  │   ├── 3_Models_and_Metrics.py
  │   └── 4_About.py
  └── utils.py          # shared helpers (page config, styling)
  ```
  Each file should have a minimal placeholder (st.title with page name).
  `demo/utils.py` should contain a `setup_page(title, icon)` helper that calls `st.set_page_config()`.
  Verify: `uv run streamlit run demo/Home.py --server.headless true` starts without errors (kill after 5 seconds).

### Phase 2: Home Page

- [ ] **Task 2.1: Build Home page**
  In `demo/Home.py`, create:
  - Page title: "Edge-AI Document Processing for BMD Go"
  - Subtitle: "On-device document classification and field extraction pipeline"
  - Three st.metric columns: "Document Types: 3", "ONNX Models: 4", "Total Size: ~199 MB"
  - Architecture flow diagram using st.graphviz_chart (DOT format):
    `Document Image → Preprocessor → EfficientNet Classifier → RapidOCR → DistilBERT NER → Postprocessor → JSON Schema Validator → Structured JSON`
  - Supported document types table (st.dataframe or st.table):
    | Document Type | Description | Key Fields |
    | Arztbesuchsbestaetigung | Medical visit confirmation | Patient, doctor, facility, date, time |
    | Reisekostenbeleg | Travel expense receipt | Vendor, date, amount, currency, VAT |
    | Lieferschein | Delivery note | Delivery number, sender, recipient, items |
  - Sidebar with navigation info and "How to Run" section

### Phase 3: Live Demo Page

- [ ] **Task 3.1: Build Live Demo page**
  In `demo/pages/1_Live_Demo.py`, create:
  - Two-column layout: left = input, right = results
  - Left column:
    - st.file_uploader for JPG/PNG images
    - st.checkbox "Generate sample document" with st.selectbox for document type
    - Show uploaded/generated image with st.image
  - Right column:
    - Load pipeline with @st.cache_resource using DocumentPipeline.from_config("config.yaml")
    - On image submit: run pipeline.process(), show:
      - Document type as colored badge (st.success/st.info/st.warning based on type)
      - Confidence as st.progress bar + percentage
      - Processing time
      - Extracted fields as formatted key-value pairs
      - Expandable "Raw JSON" section with st.json
      - Expandable "OCR Text" section with st.text
  - Handle errors gracefully (st.error if pipeline fails)
  - Import generators: _generate_one_arztbesuch, _generate_one_reisekosten, _generate_one_lieferschein from scripts.generate_samples

### Phase 4: Architecture Page

- [ ] **Task 4.1: Build Architecture page**
  In `demo/pages/2_Architecture.py`, create:
  - Section "Processing Pipeline" with a detailed graphviz flowchart showing all pipeline steps
  - Section "Pipeline Steps" with expandable sections (st.expander) for each step:
    1. **Image Preprocessing**: 224x224 resize, ImageNet normalization, numpy array
    2. **Document Classification**: EfficientNet-Lite0, 3 classes, softmax confidence
    3. **OCR Text Extraction**: RapidOCR (PaddleOCR ONNX), region detection + text recognition
    4. **Named Entity Recognition**: DistilBERT German, BIO tagging, per-document-type model
    5. **Postprocessing**: BIO tags → structured fields, date/time parsing, number extraction
    6. **Schema Validation**: JSON Schema Draft-07 validation per document type
  - Section "Model Details" with a table:
    | Model | Architecture | Format | Size | Quantization |
    | Classifier | EfficientNet-Lite0 | ONNX | 6.5 MB | Float16 |
    | NER Arztbesuch | DistilBERT German | ONNX | 64 MB | INT8 |
    | NER Reisekosten | DistilBERT German | ONNX | 64 MB | INT8 |
    | NER Lieferschein | DistilBERT German | ONNX | 64 MB | INT8 |

### Phase 5: Metrics Page

- [ ] **Task 5.1: Create metrics generation script**
  Create `demo/generate_metrics.py` that:
  - Loads the trained classifier and runs validation on sample images (use data/samples/)
  - Computes: accuracy, per-class precision/recall/F1, confusion matrix
  - For each NER model: run inference on validation JSONL files, compute entity-level F1
  - Saves all results to `demo/metrics/` as JSON files:
    - `demo/metrics/classifier_metrics.json` (accuracy, per-class metrics, confusion matrix)
    - `demo/metrics/ner_metrics.json` (per-doc-type entity F1 scores)
  - Run with: `uv run python demo/generate_metrics.py`
  Use existing code: edge_model/classification/validate.py for classifier validation patterns.
  Use sklearn.metrics for classification_report, confusion_matrix.

- [ ] **Task 5.2: Build Models & Metrics page**
  In `demo/pages/3_Models_and_Metrics.py`, create:
  - Load metrics from `demo/metrics/*.json` (generate if missing by showing st.warning + instructions)
  - Section "Classification Performance":
    - Overall accuracy as big st.metric
    - Confusion matrix as plotly heatmap (3x3: arztbesuch, reisekosten, lieferschein)
    - Per-class precision/recall/F1 as bar chart
  - Section "NER Extraction Performance":
    - Per-document-type F1 scores as bar chart
    - Table with entity-level breakdown
  - Section "Model Sizes":
    - Use ModelManager to get actual model sizes
    - Horizontal bar chart comparing model sizes
    - Total size metric

### Phase 6: About Page

- [ ] **Task 6.1: Build About page**
  In `demo/pages/4_About.py`, create:
  - Section "Team":
    | Name | Role |
    | Binder Celina | |
    | Eichsteininger Natalie | |
    | Hysenlli Klevi | |
    | Schlosser Lorenz Johannes | |
    | Suchomel Raphael | |
  - Section "Technology Stack" as a nice table:
    | Component | Technology | Purpose |
    (Copy from README.md tech stack table)
  - Section "How to Run":
    ```bash
    # Install dependencies
    uv sync --group demo --group ocr
    # Start the demo
    uv run streamlit run demo/Home.py
    ```
  - Section "Project Structure": show directory tree from README
  - Section "Repository": link to GitHub if applicable

### Phase 7: Polish & Verify

- [ ] **Task 7.1: Add consistent styling and navigation**
  In `demo/utils.py`, add:
  - Consistent page config (page_title, page_icon, layout="wide")
  - Custom CSS via st.markdown (subtle styling: hide Streamlit hamburger menu, consistent fonts)
  - Footer with project name and university info
  Apply utils.setup_page() in every page file.
  Run `uv run ruff check .` and fix any lint issues.
  Run `uv run pytest tests/unit/ -x` to make sure existing tests still pass.

- [ ] **Task 7.2: Run full test suite and verify demo**
  Run: `uv run pytest tests/ -v --tb=short` — all 372 tests must pass.
  Run: `uv run ruff check .` — must be clean.
  Run: `uv run streamlit run demo/Home.py --server.headless true` — must start without errors.
  Verify each page loads without crashes by navigating to each page programmatically or manually.
  If any test fails, fix it before marking complete.
