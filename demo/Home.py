"""Home page for the Edge-AI Document Processing demo."""

import streamlit as st
from utils import metric_card, section_header, setup_page

setup_page("Home", "🧠")

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">Edge-AI &middot; On-Device &middot; Privacy-First</div>
        <div class="hero-title">Edge-AI Document Processing</div>
        <div class="hero-subtitle">
            On-device classification and field extraction for BMD Go<br>
            <strong style="color: #00D4FF;">98.4% accuracy</strong> &middot;
            <strong style="color: #7C3AED;">4 ONNX models</strong> &middot;
            <strong style="color: #10B981;">~199 MB total</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Key Metrics Row
# ---------------------------------------------------------------------------
cols = st.columns(4)
cards = [
    ("📄", "3", "Doc Types", "cyan"),
    ("🧠", "4", "ONNX Models", "purple"),
    ("🎯", "98.4%", "Accuracy", "green"),
    ("📦", "~199 MB", "Model Size", "amber"),
]
for col, (icon, value, label, color) in zip(cols, cards):
    with col:
        st.markdown(metric_card(icon, value, label, color), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How It Works — Pipeline Steps
# ---------------------------------------------------------------------------
section_header("How It Works")

steps = [
    ("Document Image", "Camera photo or scan uploaded as JPEG/PNG"),
    ("Image Preprocessing", "Resize to 224x224, ImageNet normalization"),
    ("Document Classification", "EfficientNet-Lite0 identifies document type with 98.4% accuracy"),
    ("OCR Text Extraction", "RapidOCR converts image to text for field extraction"),
    ("NER Field Extraction", "DistilBERT extracts key fields using BIO tagging per document type"),
    ("Postprocessing", "Date/time parsing, number formatting, field normalization"),
    ("Schema Validation", "JSON Schema Draft-07 validates output structure"),
    ("Structured JSON", "Clean, validated result ready for BMD Go integration"),
]
for i, (title, desc) in enumerate(steps, 1):
    st.markdown(
        f"""
        <div class="pipeline-step">
            <span class="step-number">{i}</span>
            <span class="step-title">{title}</span>
            <div class="step-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Why Edge-AI — Feature Highlights
# ---------------------------------------------------------------------------
section_header("Why Edge-AI?")

feat_cols = st.columns(3)
features = [
    (
        "🔒",
        "Privacy First",
        "All processing happens on-device. No document data leaves the phone. "
        "No cloud APIs, no internet required.",
    ),
    (
        "⚡",
        "Lightning Fast",
        "Sub-second inference with quantized ONNX models. "
        "INT8 NER + Float16 classifier for minimal latency.",
    ),
    (
        "🎯",
        "High Accuracy",
        "98.4% document classification accuracy. "
        "Up to 99.1% NER F1 score for field extraction.",
    ),
]
for col, (icon, title, desc) in zip(feat_cols, features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Supported Document Types
# ---------------------------------------------------------------------------
section_header("Supported Document Types")

doc_cols = st.columns(3)
doc_types = [
    (
        "🏥",
        "Arztbesuchsbestaetigung",
        "Medical Visit Confirmation",
        "Patient, Doctor, Facility, Address, Date, Time, Duration",
        "cyan",
    ),
    (
        "🧾",
        "Reisekostenbeleg",
        "Travel Expense Receipt",
        "Vendor, Date, Amount, Currency, VAT Rate, Category, Receipt No.",
        "purple",
    ),
    (
        "📦",
        "Lieferschein",
        "Delivery Note",
        "Delivery No., Date, Sender, Recipient, Items, Order No., Weight",
        "green",
    ),
]
for col, (icon, name, desc, fields, color) in zip(doc_cols, doc_types):
    with col:
        border_color = {"cyan": "#00D4FF", "purple": "#7C3AED", "green": "#10B981"}[color]
        st.markdown(
            f"""
            <div class="doc-card" style="border-top: 3px solid {border_color};">
                <div class="doc-icon">{icon}</div>
                <h3>{name}</h3>
                <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.8rem;">{desc}</p>
                <div class="doc-fields">
                    {"<br>".join(f"&bull; {f.strip()}" for f in fields.split(","))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Edge-AI Docs")
    st.markdown(
        """
Navigate the demo:
- **Live Demo** — Process documents in real time
- **Architecture** — Pipeline design and flow
- **Models & Metrics** — Performance benchmarks
- **Tech Stack** — Technologies and methodology
- **About** — Team and project info
"""
    )
    st.divider()
    st.markdown("##### Quick Start")
    st.code(
        "uv sync --group demo --group ocr\nuv run streamlit run demo/Home.py",
        language="bash",
    )
