"""Tech Stack & Methodology page for the Edge-AI Document Processing demo."""

import streamlit as st
from utils import metric_card, section_header, setup_page

setup_page("Tech Stack", "\U0001f527")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">Tech Stack & Methodology</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Technologies, training approach, and engineering practices.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Technology Cards
# ---------------------------------------------------------------------------
section_header("Core Technologies")

tech_items = [
    (
        "\U0001f9e0",
        "ONNX Runtime",
        "Inference Engine",
        "Cross-platform ML inference. Runs all 4 models on-device with CPU.",
    ),
    (
        "\U0001f4ca",
        "EfficientNet-Lite0",
        "Document Classifier",
        "Lightweight CNN architecture. Float16 quantized to 6.5 MB.",
    ),
    (
        "\U0001f3f7\ufe0f",
        "DistilBERT German",
        "NER Extractor",
        "German-language transformer for named entity recognition. INT8 quantized.",
    ),
    (
        "\U0001f441\ufe0f",
        "RapidOCR",
        "Text Extraction",
        "PaddleOCR-based ONNX bundles for text detection and recognition.",
    ),
    (
        "\U0001f525",
        "PyTorch + HuggingFace",
        "Training Framework",
        "Two-phase transfer learning for classifier. HF Trainer for NER.",
    ),
    (
        "\u2705",
        "Pydantic v2 + jsonschema",
        "Validation",
        "Dual-layer validation with Pydantic models and JSON Schema Draft-07.",
    ),
]

for row_start in range(0, len(tech_items), 3):
    cols = st.columns(3)
    for col, item in zip(cols, tech_items[row_start : row_start + 3]):
        icon, name, badge, desc = item
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3>{name}</h3>
                    <span class="tech-badge" style="font-size: 0.72rem;">{badge}</span>
                    <p style="margin-top: 0.5rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Training Methodology
# ---------------------------------------------------------------------------
section_header("Training Methodology")

method_col1, method_col2 = st.columns(2)

with method_col1:
    st.markdown("#### Classification Training")
    st.markdown(
        """
    <div class="pipeline-step" style="border-left-color: #7C3AED;">
        <span class="step-number">1</span>
        <span class="step-title">Phase 1: Frozen Backbone</span>
        <div class="step-desc">
            Train only the classifier head while keeping EfficientNet backbone frozen.
            Fast convergence on document-type features.
        </div>
    </div>
    <div class="pipeline-step" style="border-left-color: #00D4FF;">
        <span class="step-number">2</span>
        <span class="step-title">Phase 2: Full Fine-Tuning</span>
        <div class="step-desc">
            Unfreeze all layers. Lower learning rate. Fine-tune entire model on
            750 synthetic document images (250 per type).
        </div>
    </div>
    <div class="pipeline-step" style="border-left-color: #10B981;">
        <span class="step-number">3</span>
        <span class="step-title">ONNX Export + Float16 Quantization</span>
        <div class="step-desc">
            Export via torch.onnx.export (opset 17). Apply float16 quantization.
            Final size: 6.5 MB.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with method_col2:
    st.markdown("#### NER Training")
    st.markdown(
        """
    <div class="pipeline-step" style="border-left-color: #7C3AED;">
        <span class="step-number">1</span>
        <span class="step-title">Synthetic BIO Data Generation</span>
        <div class="step-desc">
            Faker (German locale) generates realistic text with BIO tags.
            4500 samples total (1500 per document type, 80/20 train/val split).
        </div>
    </div>
    <div class="pipeline-step" style="border-left-color: #00D4FF;">
        <span class="step-number">2</span>
        <span class="step-title">DistilBERT Fine-Tuning</span>
        <div class="step-desc">
            HuggingFace Trainer with entity-level metrics. Subword alignment
            via word_ids(). One model per document type.
        </div>
    </div>
    <div class="pipeline-step" style="border-left-color: #10B981;">
        <span class="step-number">3</span>
        <span class="step-title">ONNX Export + INT8 Quantization</span>
        <div class="step-desc">
            Export via Optimum ORTModelForTokenClassification. Dynamic INT8
            quantization. ~64 MB per model.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
section_header("Synthetic Data Pipeline")

st.markdown(
    """
<div class="feature-card" style="padding: 1.5rem;">
    <p>All training data is <strong style="color: #00D4FF;">100% synthetic</strong> —
    no real documents, no PII, no privacy concerns.</p>
    <br>
    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
        <div class="stat-item" style="flex:1;">
            <div class="stat-number" style="color: #00D4FF;">750</div>
            <div class="stat-label">Document Images</div>
        </div>
        <div class="stat-item" style="flex:1;">
            <div class="stat-number" style="color: #7C3AED;">4,500</div>
            <div class="stat-label">NER Text Samples</div>
        </div>
        <div class="stat-item" style="flex:1;">
            <div class="stat-number" style="color: #10B981;">3</div>
            <div class="stat-label">Document Types</div>
        </div>
        <div class="stat-item" style="flex:1;">
            <div class="stat-number" style="color: #F59E0B;">de_DE</div>
            <div class="stat-label">Faker Locale</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Engineering Practices
# ---------------------------------------------------------------------------
section_header("Engineering Practices")

practice_cols = st.columns(4)
practices = [
    ("\U0001f9ea", "372", "Tests Passing", "green"),
    ("\U0001f4c1", "28", "Test Files", "cyan"),
    ("\U0001f50d", "Ruff", "Linter", "purple"),
    ("\U0001f4e6", "uv", "Package Manager", "amber"),
]
for col, (icon, val, label, color) in zip(practice_cols, practices):
    with col:
        st.markdown(metric_card(icon, val, label, color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
<div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
    <span class="tech-badge">Unit Tests</span>
    <span class="tech-badge purple-badge">Integration Tests</span>
    <span class="tech-badge green-badge">E2E Tests</span>
    <span class="tech-badge">Type Hints</span>
    <span class="tech-badge purple-badge">Docstrings</span>
    <span class="tech-badge green-badge">Python 3.12</span>
    <span class="tech-badge">JSON Schema Validation</span>
    <span class="tech-badge purple-badge">Pydantic v2</span>
    <span class="tech-badge green-badge">ONNX Quantization</span>
</div>
""",
    unsafe_allow_html=True,
)
