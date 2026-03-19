# Demo Website Design

## Context

The Edge-AI Document Processing project is feature-complete (372 tests pass, all models trained). We need a professional demo website to present the project to university professors. The website runs locally via Streamlit and showcases the full pipeline interactively.

## Architecture

Multi-page Streamlit app using Streamlit's built-in pages feature.

```
demo/
├── Home.py                    # Entry point (streamlit run demo/Home.py)
└── pages/
    ├── 1_Live_Demo.py         # Interactive document processing
    ├── 2_Architecture.py      # System architecture & pipeline flow
    ├── 3_Models_&_Metrics.py  # Model cards, accuracy, charts
    └── 4_About.py             # Team, tech stack, deployment
```

## Pages

### Home (Home.py)
- Project title + one-line description
- Architecture flow diagram (using st.columns or st.graphviz_chart)
- Quick stats: 3 doc types, 4 ONNX models, ~199 MB, 372 tests
- Supported document types table with key fields

### Live Demo (1_Live_Demo.py)
- File uploader (JPG/PNG)
- "Generate Sample" checkbox with document type selector
- Two-column layout: image left, results right
- Results: document type badge, confidence meter, extracted fields, expandable JSON + OCR text

### Architecture (2_Architecture.py)
- Full pipeline flow with step-by-step explanation
- Each component: name, model, input/output format, purpose
- Model cards for EfficientNet-Lite0 and DistilBERT German
- Quantization details (Float16, INT8)

### Models & Metrics (3_Models_&_Metrics.py)
- Run classifier validation to get confusion matrix + accuracy
- Run NER validation per doc type to get F1 scores
- Visualize with matplotlib/plotly charts embedded in Streamlit
- Model size comparison table

### About (4_About.py)
- Team members table
- Technology stack
- How to run the project
- Repository structure overview

## Dependencies

Add to `demo` group in pyproject.toml:
- streamlit>=1.30.0
- plotly>=5.18.0 (for interactive charts)
- transformers>=4.38.0 (needed by ExtractorInference at import time)

## Key Technical Decisions

1. **Fix transformers import**: ExtractorInference imports `DistilBertTokenizerFast` at module level. Demo group must include transformers, or we lazy-load it. Simplest: add transformers to demo deps.
2. **Metrics generation**: Create a `demo/generate_metrics.py` script that runs model validation and saves results to `demo/metrics/`. Charts are built from these saved results.
3. **All English**: Every UI string in English for the presentation.
4. **Single command**: `uv run streamlit run demo/Home.py` starts everything.

## How to Run

```bash
uv sync --group demo --group ocr
uv run streamlit run demo/Home.py
```
