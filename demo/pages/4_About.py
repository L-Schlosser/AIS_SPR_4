"""About page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("About", "ℹ️")

st.title("ℹ️ About")
st.write("Team information, technology stack, and project details.")

# --- Section 1: Team ---
st.header("Team")

st.table(
    {
        "Name": [
            "Binder Celina",
            "Eichsteininger Natalie",
            "Hysenlli Klevi",
            "Schlosser Lorenz Johannes",
            "Suchomel Raphael",
        ],
        "Role": ["", "", "", "", ""],
    }
)

# --- Section 2: Technology Stack ---
st.header("Technology Stack")

st.table(
    {
        "Component": [
            "Classifier",
            "OCR",
            "NER Extractor",
            "Runtime",
            "Quantization",
            "Data Models",
            "Schema Validation",
            "Package Manager",
            "Linting",
            "Testing",
        ],
        "Technology": [
            "EfficientNet-Lite0 (timm)",
            "RapidOCR (PaddleOCR ONNX)",
            "DistilBERT German (transformers)",
            "ONNX Runtime",
            "Float16 (classifier), INT8 (NER)",
            "Pydantic v2",
            "jsonschema (Draft-07)",
            "uv",
            "Ruff",
            "pytest",
        ],
        "Purpose": [
            "Document type classification",
            "Text extraction from images",
            "Named entity recognition for field extraction",
            "Cross-platform model inference",
            "Model size reduction",
            "Input/output validation",
            "Output structure validation",
            "Fast Python dependency management",
            "Code quality enforcement",
            "Unit, integration, and e2e tests",
        ],
    }
)

# --- Section 3: How to Run ---
st.header("How to Run")

st.code(
    """# Install dependencies
uv sync --group demo --group ocr

# Start the demo
uv run streamlit run demo/Home.py""",
    language="bash",
)

# --- Section 4: Project Structure ---
st.header("Project Structure")

st.code(
    """.
├── api/                        # Service layer
│   ├── models.py               # Pydantic data models + DocumentType enum
│   └── service.py              # DocumentService orchestrator
├── edge_model/
│   ├── classification/         # Document classifier
│   │   ├── config.py           # ClassificationConfig
│   │   ├── dataset.py          # Image dataset + transforms
│   │   ├── train.py            # Two-phase transfer learning
│   │   ├── export_onnx.py      # ONNX export + float16 quantization
│   │   └── validate.py         # ONNX model validation
│   ├── extraction/             # NER field extraction
│   │   ├── labels.py           # BIO tag definitions per doc type
│   │   ├── config.py           # ExtractionConfig
│   │   ├── dataset.py          # NER dataset with subword alignment
│   │   ├── train.py            # HuggingFace Trainer-based NER training
│   │   ├── postprocess.py      # BIO tags → structured fields
│   │   └── export_onnx.py      # ONNX export + INT8 quantization
│   └── inference/              # Runtime inference
│       ├── preprocessor.py     # Image preprocessing (ImageNet normalization)
│       ├── classifier_inference.py  # ONNX classifier wrapper
│       ├── extractor_inference.py   # ONNX NER wrapper
│       ├── validator.py        # JSON schema validation
│       ├── config.py           # PipelineConfig + YAML loader
│       └── pipeline.py         # Full pipeline orchestrator
├── ocr/                        # OCR module
│   ├── engine.py               # RapidOCR wrapper
│   ├── preprocessing.py        # Grayscale, thresholding, deskew
│   └── postprocessing.py       # Region sorting + text merging
├── mobile_app/                 # Mobile client integration layer
│   └── src/
│       ├── model_manager.py    # ONNX model file management
│       └── app.py              # CLI: process, info, batch, demo
├── scripts/
│   ├── generate_samples.py     # Synthetic document image generator
│   └── generate_text_samples.py # BIO-tagged NER text generator
├── data/
│   ├── schemas/                # JSON schemas for output validation
│   └── samples/                # Generated training data (gitignored)
├── tests/
│   ├── unit/                   # 333+ unit tests
│   ├── integration/            # OCR + model integration tests
│   └── e2e/                    # Full pipeline end-to-end tests
├── docs/
│   ├── architecture.md         # System architecture + Mermaid diagrams
│   └── model_pipeline.md       # Training + export procedures
├── results/                    # Example JSON outputs per document type
├── config.yaml                 # Pipeline configuration (model paths)
└── pyproject.toml              # Dependencies managed via uv""",
    language="text",
)

# --- Section 5: Repository ---
st.header("Repository")

st.info(
    "This project is intended for research and educational purposes. "
    "Developed as part of a university student project."
)
