"""About page for the Edge-AI Document Processing demo."""

import streamlit as st
from utils import metric_card, section_header, setup_page

setup_page("About", "ℹ️")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">About This Project</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            University student project — Edge-AI document processing for BMD Go.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
section_header("Team")

team_members = [
    ("CB", "Celina Binder"),
    ("NE", "Natalie Eichsteininger"),
    ("KH", "Klevi Hysenlli"),
    ("LS", "Lorenz Schlosser"),
    ("RS", "Raphael Suchomel"),
]

team_cols = st.columns(5)
for col, (initials, name) in zip(team_cols, team_members):
    with col:
        st.markdown(
            f"""
            <div class="team-card">
                <div class="avatar">{initials}</div>
                <div class="member-name">{name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Project Stats
# ---------------------------------------------------------------------------
section_header("Project at a Glance")

stat_cols = st.columns(4)
stats = [
    ("\U0001f9ea", "372", "Tests Passing", "green"),
    ("\U0001f4c4", "3", "Document Types", "cyan"),
    ("\U0001f9e0", "4", "ONNX Models", "purple"),
    ("\U0001f4e6", "~199 MB", "Total Size", "amber"),
]
for col, (icon, value, label, color) in zip(stat_cols, stats):
    with col:
        st.markdown(metric_card(icon, value, label, color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How to Run
# ---------------------------------------------------------------------------
section_header("Getting Started")

run_col1, run_col2 = st.columns(2)

with run_col1:
    st.markdown("#### Install & Run")
    st.code(
        """# Install dependencies
uv sync --group demo --group ocr

# Generate training data
uv run python -m scripts.generate_samples
uv run python -m scripts.generate_text_samples

# Train models
uv run python -m edge_model.classification.train
uv run python -m edge_model.extraction.train arztbesuchsbestaetigung
uv run python -m edge_model.extraction.train reisekostenbeleg
uv run python -m edge_model.extraction.train lieferschein

# Export to ONNX
uv run python -m edge_model.classification.export_onnx
uv run python -m edge_model.extraction.export_onnx

# Start the demo
uv run streamlit run demo/Home.py""",
        language="bash",
    )

with run_col2:
    st.markdown("#### Run Tests")
    st.code(
        """# All tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# Integration tests (requires models)
uv run pytest tests/integration/ -v -m integration

# E2E tests
uv run pytest tests/e2e/ -v -m e2e

# Lint check
uv run ruff check .""",
        language="bash",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Project Structure
# ---------------------------------------------------------------------------
section_header("Project Structure")

st.code(
    """.
\u251c\u2500\u2500 api/                        # Service layer (Pydantic models, DocumentService)
\u251c\u2500\u2500 edge_model/
\u2502   \u251c\u2500\u2500 classification/         # EfficientNet-Lite0 classifier (train, export, config)
\u2502   \u251c\u2500\u2500 extraction/             # DistilBERT NER models (train, export, postprocess)
\u2502   \u2514\u2500\u2500 inference/              # ONNX Runtime (classify\u2192OCR\u2192extract\u2192validate)
\u251c\u2500\u2500 ocr/                        # RapidOCR wrapper (preprocessing, engine, postprocessing)
\u251c\u2500\u2500 mobile_app/src/             # CLI integration + ONNX Runtime Mobile guide
\u251c\u2500\u2500 scripts/                    # Synthetic data generators (images + BIO text)
\u251c\u2500\u2500 data/schemas/               # JSON Schema Draft-07 per document type
\u251c\u2500\u2500 tests/                      # 372 tests (unit + integration + e2e)
\u251c\u2500\u2500 demo/                       # This Streamlit website
\u251c\u2500\u2500 docs/                       # Architecture + model pipeline documentation
\u251c\u2500\u2500 config.yaml                 # Pipeline configuration (model paths, thresholds)
\u2514\u2500\u2500 pyproject.toml              # Dependencies managed via uv""",
    language="text",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="feature-card" style="text-align: center; border-color: #2D3748;">
        <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
            This project is intended for research and educational purposes.<br>
            Developed as part of a university student project for BMD Go.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
