# Demo Website Redesign — Dark AI Tech Theme

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the boring default-Streamlit demo website into a visually stunning dark-themed AI product dashboard that impresses university professors and BMD company stakeholders.

**Architecture:** Full Streamlit redesign with aggressive custom CSS/HTML injection via `st.markdown(unsafe_allow_html=True)`. Dark backgrounds, gradient accent cards, Plotly gauge charts, animated metrics, technology badges. No new frameworks — stay within Streamlit + Plotly + custom CSS. All existing pipeline functionality preserved.

**Tech Stack:** Python 3.12, Streamlit, Plotly, custom CSS/HTML, uv package manager

---

## Phase 1: Foundation — Dark Theme CSS + Shared Components

### Task 1.1: Rewrite `demo/utils.py` with dark theme CSS and reusable card components

**Files:**
- Modify: `demo/utils.py`

**Step 1: Replace `demo/utils.py` entirely with the following content:**

```python
"""Shared helpers for the Streamlit demo app."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so project modules (api, edge_model, etc.) are importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg_dark": "#0E1117",
    "bg_card": "#1A1F2E",
    "bg_card_hover": "#232938",
    "accent_cyan": "#00D4FF",
    "accent_purple": "#7C3AED",
    "accent_green": "#10B981",
    "accent_amber": "#F59E0B",
    "accent_red": "#EF4444",
    "text_primary": "#E2E8F0",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "border": "#2D3748",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_DARK_CSS = """
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Dark scrollbar */
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: #0E1117;}
    ::-webkit-scrollbar-thumb {background: #2D3748; border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #4A5568;}

    /* Font */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 50%, #1E1B4B 100%);
        border: 1px solid #2D3748;
        border-radius: 16px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(124, 58, 237, 0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #00D4FF 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(124, 58, 237, 0.15));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 999px;
        padding: 0.4rem 1.2rem;
        font-size: 0.85rem;
        color: #00D4FF;
        margin-bottom: 1.5rem;
        position: relative;
    }

    /* Metric cards */
    .metric-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: #00D4FF;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
        transform: translateY(-2px);
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .cyan { color: #00D4FF; }
    .purple { color: #7C3AED; }
    .green { color: #10B981; }
    .amber { color: #F59E0B; }

    /* Feature cards */
    .feature-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #7C3AED;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.1);
    }
    .feature-card h3 {
        font-size: 1.1rem;
        margin: 0.8rem 0 0.5rem 0;
        color: #E2E8F0;
    }
    .feature-card p {
        font-size: 0.9rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }
    .feature-icon {
        font-size: 2rem;
    }

    /* Pipeline step cards */
    .pipeline-step {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-left: 3px solid #00D4FF;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .pipeline-step:hover {
        border-left-color: #7C3AED;
        background: #232938;
    }
    .pipeline-step .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.8rem;
    }
    .pipeline-step .step-title {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 1rem;
    }
    .pipeline-step .step-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        margin-left: 2.8rem;
    }

    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #00D4FF;
        margin: 0.2rem;
    }
    .tech-badge.purple-badge {
        background: rgba(124, 58, 237, 0.1);
        border-color: rgba(124, 58, 237, 0.2);
        color: #A78BFA;
    }
    .tech-badge.green-badge {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
    }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2D3748;
    }
    .section-header span {
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Doc type cards */
    .doc-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    .doc-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .doc-card h3 {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin: 0.8rem 0 0.5rem 0;
    }
    .doc-card .doc-fields {
        color: #94A3B8;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .doc-card .doc-icon {
        font-size: 2.2rem;
    }

    /* Result field cards for live demo */
    .field-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .field-card .field-key {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .field-card .field-value {
        font-size: 1.05rem;
        color: #E2E8F0;
        font-weight: 600;
    }

    /* Team member cards */
    .team-card {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .team-card:hover {
        border-color: #7C3AED;
    }
    .team-card .avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00D4FF, #7C3AED);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .team-card .member-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .team-card .member-role {
        font-size: 0.8rem;
        color: #94A3B8;
    }

    /* Stat counter cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .stat-item {
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
    }
    .stat-item .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00D4FF;
    }
    .stat-item .stat-label {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    /* Custom footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(180deg, transparent, #0E1117);
        text-align: center;
        padding: 12px 0;
        font-size: 0.78rem;
        color: #64748B;
        z-index: 999;
    }

    /* Override Streamlit elements for dark consistency */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1F2E;
        border-radius: 8px;
        border: 1px solid #2D3748;
        color: #94A3B8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #232938;
        border-color: #00D4FF;
        color: #00D4FF;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1A1F2E;
        border-radius: 8px;
        border: 1px solid #2D3748;
    }

    /* Progress bar override */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #7C3AED);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #2D3748;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #E2E8F0;
    }
</style>
"""

_FOOTER_HTML = """
<div class="custom-footer">
    Edge-AI Document Processing for BMD Go — University Student Project — 2025
</div>
"""


def setup_page(title: str, icon: str) -> None:
    """Configure page settings, apply dark theme CSS, and inject footer."""
    st.set_page_config(
        page_title=f"{title} — Edge-AI Document Processing",
        page_icon=icon,
        layout="wide",
    )
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)


def metric_card(icon: str, value: str, label: str, color: str = "cyan") -> str:
    """Return HTML for a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value {color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def section_header(text: str) -> None:
    """Render a styled section header with gradient accent."""
    st.markdown(f'<div class="section-header"><span>{text}</span></div>', unsafe_allow_html=True)


def dark_plotly_layout(fig, height: int = 400) -> None:
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
        ),
    )
```

**Step 2: Run ruff to verify**

```bash
uv run ruff check demo/utils.py --fix
```

**Step 3: Commit**

```bash
git add demo/utils.py
git commit -m "feat: dark AI theme CSS system and reusable card components"
```

---

## Phase 2: Home Page — Hero Landing

### Task 2.1: Rewrite `demo/Home.py` with hero section, animated metric cards, pipeline steps, and document type showcase

**Files:**
- Modify: `demo/Home.py`

**Step 1: Replace `demo/Home.py` entirely with:**

```python
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
    ("📄", "3", "Document Types", "cyan"),
    ("🧠", "4", "ONNX Models", "purple"),
    ("🎯", "98.4%", "Classification Accuracy", "green"),
    ("📦", "~199 MB", "Total Model Size", "amber"),
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
```

**Step 2: Run ruff and commit**

```bash
uv run ruff check demo/Home.py --fix
git add demo/Home.py
git commit -m "feat: hero landing page with gradient cards, pipeline steps, feature highlights"
```

---

## Phase 3: Live Demo — Professional UX

### Task 3.1: Rewrite `demo/pages/1_Live_Demo.py` with dark-themed results display and styled field cards

**Files:**
- Modify: `demo/pages/1_Live_Demo.py`

**Step 1: Replace `demo/pages/1_Live_Demo.py` entirely with:**

```python
"""Live Demo page — upload or generate a document and process it through the pipeline."""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from PIL import Image
from utils import section_header, setup_page

setup_page("Live Demo", "🔍")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">Live Demo</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Upload a document image or generate a sample, then run it through the full Edge-AI pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline loader (cached)
# ---------------------------------------------------------------------------
PIPELINE_ERROR: str | None = None


@st.cache_resource
def _load_pipeline():
    """Load the document processing pipeline once, cached across reruns."""
    try:
        from edge_model.inference.pipeline import DocumentPipeline

        return DocumentPipeline.from_config("config.yaml"), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------
DOC_TYPE_OPTIONS = {
    "Arztbesuchsbestaetigung": "arztbesuch",
    "Reisekostenbeleg": "reisekosten",
    "Lieferschein": "lieferschein",
}


def _generate_sample(doc_type_key: str) -> Image.Image:
    """Generate a synthetic document image for the selected type."""
    from scripts.generate_samples import (
        _generate_one_arztbesuch,
        _generate_one_lieferschein,
        _generate_one_reisekosten,
    )

    generators = {
        "arztbesuch": _generate_one_arztbesuch,
        "reisekosten": _generate_one_reisekosten,
        "lieferschein": _generate_one_lieferschein,
    }
    img, _label = generators[doc_type_key]()
    return img


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

# ---- Left column: input ----
with left_col:
    section_header("Input Document")

    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPEG, PNG",
    )

    use_sample = st.checkbox("Generate sample document instead")
    sample_type = st.selectbox(
        "Document type",
        options=list(DOC_TYPE_OPTIONS.keys()),
        disabled=not use_sample,
    )

    # Determine which image to use
    input_image: Image.Image | None = None

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
    elif use_sample and sample_type is not None:
        with st.spinner("Generating sample..."):
            input_image = _generate_sample(DOC_TYPE_OPTIONS[sample_type])

    if input_image is not None:
        st.image(input_image, caption="Input document", use_container_width=True)

# ---- Right column: results ----
with right_col:
    section_header("Extraction Results")

    if input_image is None:
        st.markdown(
            """
            <div class="feature-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                <p>Upload an image or generate a sample to see results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pipeline, load_error = _load_pipeline()

        if load_error is not None:
            st.error(f"Could not load pipeline: {load_error}")
            st.code("uv sync --group demo --group ocr", language="bash")
        else:
            try:
                image_array = np.array(input_image)
                start_time = time.perf_counter()
                result = pipeline.process(image_array)
                elapsed = time.perf_counter() - start_time

                # Document type + confidence
                doc_type_str = result.document_type.value
                type_colors = {
                    "arztbesuchsbestaetigung": "#00D4FF",
                    "reisekostenbeleg": "#7C3AED",
                    "lieferschein": "#10B981",
                }
                type_color = type_colors.get(doc_type_str, "#00D4FF")

                st.markdown(
                    f"""
                    <div class="metric-card" style="border-top: 3px solid {type_color}; margin-bottom: 1rem;">
                        <div class="metric-label">Document Type</div>
                        <div class="metric-value" style="font-size: 1.3rem; color: {type_color};">
                            {doc_type_str}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <span class="tech-badge">Confidence: {result.confidence:.1%}</span>
                            <span class="tech-badge purple-badge">Time: {elapsed:.2f}s</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence bar
                st.progress(result.confidence)

                # Extracted fields as styled cards
                if result.fields:
                    st.markdown("#### Extracted Fields")
                    for key, value in result.fields.items():
                        if key.startswith("_"):
                            continue
                        st.markdown(
                            f"""
                            <div class="field-card">
                                <div class="field-key">{key}</div>
                                <div class="field-value">{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No fields extracted (confidence may be below threshold).")

                # Raw JSON
                with st.expander("Raw JSON Output"):
                    st.json(result.model_dump(mode="json"))

                # OCR Text
                with st.expander("OCR Text"):
                    if result.raw_text:
                        st.code(result.raw_text, language=None)
                    else:
                        st.caption("No OCR text available.")

            except Exception as exc:
                st.error(f"Pipeline processing failed: {exc}")
```

**Step 2: Run ruff and commit**

```bash
uv run ruff check demo/pages/1_Live_Demo.py --fix
git add demo/pages/1_Live_Demo.py
git commit -m "feat: dark-themed live demo with styled field cards and confidence badges"
```

---

## Phase 4: Architecture Page

### Task 4.1: Rewrite `demo/pages/2_Architecture.py` with styled pipeline cards and technology badges

**Files:**
- Modify: `demo/pages/2_Architecture.py`

**Step 1: Replace `demo/pages/2_Architecture.py` entirely with:**

```python
"""Architecture page for the Edge-AI Document Processing demo."""

import streamlit as st
from utils import section_header, setup_page

setup_page("Architecture", "🏗️")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">Architecture</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            End-to-end pipeline from document image to structured JSON output.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline Flow — Styled Cards
# ---------------------------------------------------------------------------
section_header("Processing Pipeline")

pipeline_steps = [
    ("📸", "Document Image", "JPEG/PNG", "Camera photo or scan input", "#64748B"),
    ("🔧", "Image Preprocessor", "224x224, ImageNet norm", "Resize, normalize, CHW layout, batch dim", "#00D4FF"),
    (
        "🧠",
        "EfficientNet-Lite0 Classifier",
        "3-class softmax",
        "Identifies document type with confidence score. Threshold: 0.7",
        "#7C3AED",
    ),
    (
        "👁️",
        "RapidOCR Engine",
        "Text extraction",
        "Grayscale, adaptive threshold, text detection + recognition, region sorting",
        "#F59E0B",
    ),
    (
        "🏷️",
        "DistilBERT NER",
        "BIO tagging",
        "One fine-tuned model per document type. Extracts named entities from OCR text",
        "#10B981",
    ),
    (
        "⚙️",
        "Postprocessor",
        "Field normalization",
        "Date/time parsing, number formatting, type-specific field mapping",
        "#00D4FF",
    ),
    (
        "✅",
        "JSON Schema Validator",
        "Draft-07",
        "Validates output against document-type-specific schemas",
        "#7C3AED",
    ),
    ("📋", "Structured JSON", "ProcessingResult", "Clean, validated output for BMD Go integration", "#10B981"),
]

for i, (icon, title, badge_text, desc, color) in enumerate(pipeline_steps, 1):
    st.markdown(
        f"""
        <div class="pipeline-step" style="border-left-color: {color};">
            <span class="step-number">{i}</span>
            <span class="step-title">{icon} {title}</span>
            <span class="tech-badge" style="margin-left: 0.5rem; font-size: 0.72rem;">{badge_text}</span>
            <div class="step-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline Detail Expanders
# ---------------------------------------------------------------------------
section_header("Component Details")

with st.expander("1. Image Preprocessing"):
    st.markdown(
        """
**Module:** `edge_model/inference/preprocessor.py`

- Resize input image to **224x224** pixels
- Normalize with ImageNet mean/std: `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- Convert HWC → CHW layout and add batch dimension
- Output: `(1, 3, 224, 224)` float32 numpy array
"""
    )

with st.expander("2. Document Classification"):
    st.markdown(
        """
**Model:** EfficientNet-Lite0 (pretrained on ImageNet, fine-tuned for 3 classes)

- **Input:** `(1, 3, 224, 224)` float32 image tensor
- **Output:** `(1, 3)` logits → softmax → class probabilities
- **Training:** Two-phase transfer learning (frozen backbone → full fine-tuning)
- **Quantization:** Float16 — model size reduced to **6.5 MB**
- **Confidence threshold:** 0.7 (below → rejected)
"""
    )

with st.expander("3. OCR Text Extraction"):
    st.markdown(
        """
**Library:** RapidOCR (PaddleOCR ONNX bundles)

- **Preprocessing:** Grayscale conversion, adaptive thresholding, sharpening
- **Engine:** Text region detection + recognition via ONNX models
- **Postprocessing:** Sorts regions top-to-bottom / left-to-right, merges into readable text
"""
    )

with st.expander("4. Named Entity Recognition (NER)"):
    st.markdown(
        """
**Model:** DistilBERT-base-german-cased (one fine-tuned model per document type)

- **Input:** Tokenized OCR text (max 256 tokens, `is_split_into_words=True`)
- **Output:** BIO tag sequence per token, collapsed to word-level via `word_ids()`
- **Quantization:** INT8 dynamic — each model **~64 MB**
- Separate models trained on type-specific entity labels
"""
    )

with st.expander("5. Postprocessing & Validation"):
    st.markdown(
        """
**Postprocessing** (`edge_model/extraction/postprocess.py`):
- BIO tag sequences → merged field values
- Date parsing (DD.MM.YYYY → ISO), time parsing, number extraction
- Type-specific postprocessors for each document type

**Schema Validation** (`edge_model/inference/validator.py`):
- JSON Schema Draft-07 validation per document type
- Pydantic v2 models as secondary validation layer
- Schemas in `data/schemas/`
"""
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model Specifications
# ---------------------------------------------------------------------------
section_header("Model Specifications")

st.markdown(
    """
<table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
    <thead>
        <tr style="border-bottom: 2px solid #2D3748;">
            <th style="padding: 12px; text-align: left; color: #94A3B8;">Model</th>
            <th style="padding: 12px; text-align: left; color: #94A3B8;">Architecture</th>
            <th style="padding: 12px; text-align: left; color: #94A3B8;">Format</th>
            <th style="padding: 12px; text-align: left; color: #94A3B8;">Size</th>
            <th style="padding: 12px; text-align: left; color: #94A3B8;">Quantization</th>
        </tr>
    </thead>
    <tbody>
        <tr style="border-bottom: 1px solid #1A1F2E;">
            <td style="padding: 10px; color: #E2E8F0;">Classifier</td>
            <td style="padding: 10px; color: #94A3B8;">EfficientNet-Lite0</td>
            <td style="padding: 10px;"><span class="tech-badge">ONNX</span></td>
            <td style="padding: 10px; color: #10B981; font-weight: 600;">6.5 MB</td>
            <td style="padding: 10px;"><span class="tech-badge green-badge">Float16</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #1A1F2E;">
            <td style="padding: 10px; color: #E2E8F0;">NER Arztbesuch</td>
            <td style="padding: 10px; color: #94A3B8;">DistilBERT German</td>
            <td style="padding: 10px;"><span class="tech-badge">ONNX</span></td>
            <td style="padding: 10px; color: #F59E0B; font-weight: 600;">64 MB</td>
            <td style="padding: 10px;"><span class="tech-badge purple-badge">INT8</span></td>
        </tr>
        <tr style="border-bottom: 1px solid #1A1F2E;">
            <td style="padding: 10px; color: #E2E8F0;">NER Reisekosten</td>
            <td style="padding: 10px; color: #94A3B8;">DistilBERT German</td>
            <td style="padding: 10px;"><span class="tech-badge">ONNX</span></td>
            <td style="padding: 10px; color: #F59E0B; font-weight: 600;">64 MB</td>
            <td style="padding: 10px;"><span class="tech-badge purple-badge">INT8</span></td>
        </tr>
        <tr>
            <td style="padding: 10px; color: #E2E8F0;">NER Lieferschein</td>
            <td style="padding: 10px; color: #94A3B8;">DistilBERT German</td>
            <td style="padding: 10px;"><span class="tech-badge">ONNX</span></td>
            <td style="padding: 10px; color: #F59E0B; font-weight: 600;">64 MB</td>
            <td style="padding: 10px;"><span class="tech-badge purple-badge">INT8</span></td>
        </tr>
    </tbody>
</table>
<div style="margin-top: 1rem;">
    <span class="tech-badge" style="font-size: 0.85rem;">Total on-device footprint: ~199 MB</span>
</div>
""",
    unsafe_allow_html=True,
)
```

**Step 2: Run ruff and commit**

```bash
uv run ruff check demo/pages/2_Architecture.py --fix
git add demo/pages/2_Architecture.py
git commit -m "feat: dark-themed architecture page with pipeline cards and model specs table"
```

---

## Phase 5: Models & Metrics — The Star Page

### Task 5.1: Rewrite `demo/pages/3_Models_and_Metrics.py` with gauge charts, dark Plotly theme, and styled tables

**Files:**
- Modify: `demo/pages/3_Models_and_Metrics.py`

**Step 1: Replace `demo/pages/3_Models_and_Metrics.py` entirely with:**

```python
"""Models & Metrics page for the Edge-AI Document Processing demo."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from utils import dark_plotly_layout, metric_card, section_header, setup_page

setup_page("Models & Metrics", "📊")

METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


classifier_metrics = _load_json(METRICS_DIR / "classifier_metrics.json")
ner_metrics = _load_json(METRICS_DIR / "ner_metrics.json")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container" style="padding: 2rem;">
        <div class="hero-title" style="font-size: 2rem;">Models & Metrics</div>
        <div class="hero-subtitle" style="font-size: 1rem;">
            Performance benchmarks across classification and field extraction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if classifier_metrics is None or ner_metrics is None:
    st.warning("Metrics files not found. Run `uv run python -m demo.generate_metrics` to generate them.")

# ---------------------------------------------------------------------------
# Classification Performance
# ---------------------------------------------------------------------------
section_header("Classification Performance")

if classifier_metrics:
    # Hero gauge + metric cards
    gauge_col, cards_col = st.columns([1, 1])

    with gauge_col:
        accuracy = classifier_metrics["accuracy"]
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=accuracy * 100,
                number={"suffix": "%", "font": {"size": 48, "color": "#10B981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#2D3748", "dtick": 10},
                    "bar": {"color": "#10B981", "thickness": 0.3},
                    "bgcolor": "#1A1F2E",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [50, 80], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [80, 100], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#00D4FF", "width": 3},
                        "thickness": 0.8,
                        "value": accuracy * 100,
                    },
                },
                title={"text": "Overall Accuracy", "font": {"size": 16, "color": "#94A3B8"}},
            )
        )
        dark_plotly_layout(fig_gauge, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with cards_col:
        st.markdown("<br>", unsafe_allow_html=True)
        per_class = classifier_metrics["per_class"]
        pcols = st.columns(3)
        class_icons = {"arztbesuchsbestaetigung": "🏥", "lieferschein": "📦", "reisekostenbeleg": "🧾"}
        class_colors = {"arztbesuchsbestaetigung": "cyan", "lieferschein": "green", "reisekostenbeleg": "purple"}
        for col, c_name in zip(pcols, per_class):
            with col:
                f1 = per_class[c_name]["f1"]
                icon = class_icons.get(c_name, "📄")
                color = class_colors.get(c_name, "cyan")
                st.markdown(
                    metric_card(icon, f"{f1:.1%}", f"F1 {c_name[:10]}...", color),
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"""<div style="text-align: center; margin-top: 1rem;">
                <span class="tech-badge">Evaluated on {classifier_metrics['total_images']} images</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Confusion matrix + Per-class bar chart
    cm_col, bar_col = st.columns(2)

    with cm_col:
        st.markdown("#### Confusion Matrix")
        labels = ["arztbesuchsbest.", "lieferschein", "reisekostenbeleg"]
        cm = classifier_metrics["confusion_matrix"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                texttemplate="%{z}",
                colorscale=[[0, "#0E1117"], [0.5, "#1E3A5F"], [1, "#00D4FF"]],
                showscale=False,
                textfont={"color": "#E2E8F0", "size": 14},
            )
        )
        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        dark_plotly_layout(fig_cm, height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with bar_col:
        st.markdown("#### Per-Class Metrics")
        class_names = list(per_class.keys())
        short_names = [n[:12] + "." if len(n) > 12 else n for n in class_names]
        precisions = [per_class[c]["precision"] for c in class_names]
        recalls = [per_class[c]["recall"] for c in class_names]
        f1s = [per_class[c]["f1"] for c in class_names]

        fig_cls = go.Figure()
        fig_cls.add_trace(go.Bar(name="Precision", x=short_names, y=precisions, marker_color="#00D4FF"))
        fig_cls.add_trace(go.Bar(name="Recall", x=short_names, y=recalls, marker_color="#7C3AED"))
        fig_cls.add_trace(go.Bar(name="F1", x=short_names, y=f1s, marker_color="#10B981"))
        fig_cls.update_layout(
            barmode="group",
            yaxis=dict(range=[0.9, 1.005], title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        dark_plotly_layout(fig_cls, height=380)
        st.plotly_chart(fig_cls, use_container_width=True)

# ---------------------------------------------------------------------------
# NER Extraction Performance
# ---------------------------------------------------------------------------
section_header("NER Extraction Performance")

if ner_metrics:
    # Gauge charts for each document type
    ner_cols = st.columns(3)
    ner_colors = ["#00D4FF", "#10B981", "#7C3AED"]
    doc_types = list(ner_metrics.keys())

    for col, dt, color in zip(ner_cols, doc_types, ner_colors):
        with col:
            f1_val = ner_metrics[dt]["micro_f1"]
            fig_ner_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=f1_val * 100,
                    number={"suffix": "%", "font": {"size": 36, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#2D3748", "dtick": 20},
                        "bar": {"color": color, "thickness": 0.3},
                        "bgcolor": "#1A1F2E",
                        "borderwidth": 0,
                        "steps": [{"range": [90, 100], "color": "rgba(16, 185, 129, 0.1)"}],
                    },
                    title={"text": dt[:15], "font": {"size": 12, "color": "#94A3B8"}},
                )
            )
            dark_plotly_layout(fig_ner_gauge, height=250)
            st.plotly_chart(fig_ner_gauge, use_container_width=True)

    # Entity-level breakdown per doc type
    st.markdown("#### Entity-Level Breakdown")
    for dt in doc_types:
        with st.expander(f"{dt} — Micro F1: {ner_metrics[dt]['micro_f1']:.1%}"):
            entities = ner_metrics[dt]["per_entity"]
            rows_html = ""
            for entity_name, metrics in entities.items():
                f1 = metrics["f1"]
                f1_color = "#10B981" if f1 >= 0.95 else "#F59E0B" if f1 >= 0.8 else "#EF4444"
                rows_html += f"""
                <tr style="border-bottom: 1px solid #1A1F2E;">
                    <td style="padding: 8px; color: #E2E8F0;">{entity_name}</td>
                    <td style="padding: 8px; color: #94A3B8;">{metrics['precision']:.4f}</td>
                    <td style="padding: 8px; color: #94A3B8;">{metrics['recall']:.4f}</td>
                    <td style="padding: 8px; color: {f1_color}; font-weight: 600;">{f1:.4f}</td>
                </tr>
                """
            st.markdown(
                f"""
                <table style="width:100%; border-collapse: collapse; font-size: 0.85rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #2D3748;">
                            <th style="padding: 8px; text-align: left; color: #64748B;">Entity</th>
                            <th style="padding: 8px; text-align: left; color: #64748B;">Precision</th>
                            <th style="padding: 8px; text-align: left; color: #64748B;">Recall</th>
                            <th style="padding: 8px; text-align: left; color: #64748B;">F1</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                <div style="margin-top: 0.5rem;">
                    <span class="tech-badge" style="font-size: 0.75rem;">
                        {ner_metrics[dt]['num_samples']} samples
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Model Sizes
# ---------------------------------------------------------------------------
section_header("Model Size Breakdown")

try:
    from mobile_app.src.model_manager import ModelManager

    manager = ModelManager("edge_model")
    model_info = manager.get_model_info()
    total_size = manager.get_total_size_mb()

    names = []
    sizes = []
    for name, info in model_info.items():
        names.append(name)
        sizes.append(info.size_mb if info.exists else 0.0)

    st.markdown(
        metric_card("📦", f"{total_size:.1f} MB", "Total Model Size", "amber"),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    fig_sizes = go.Figure(
        go.Bar(
            x=sizes,
            y=names,
            orientation="h",
            marker_color=["#00D4FF", "#7C3AED", "#10B981", "#F59E0B"][: len(names)],
            text=[f"{s:.1f} MB" for s in sizes],
            textposition="outside",
            textfont={"color": "#E2E8F0"},
        )
    )
    dark_plotly_layout(fig_sizes, height=280)
    st.plotly_chart(fig_sizes, use_container_width=True)
except Exception:
    size_cols = st.columns(4)
    fallback = [
        ("🧠", "6.5 MB", "Classifier", "cyan"),
        ("🏥", "64 MB", "NER Arztbesuch", "purple"),
        ("🧾", "64 MB", "NER Reisekosten", "green"),
        ("📦", "64 MB", "NER Lieferschein", "amber"),
    ]
    for col, (icon, size, name, color) in zip(size_cols, fallback):
        with col:
            st.markdown(metric_card(icon, size, name, color), unsafe_allow_html=True)
```

**Step 2: Run ruff and commit**

```bash
uv run ruff check demo/pages/3_Models_and_Metrics.py --fix
git add demo/pages/3_Models_and_Metrics.py
git commit -m "feat: dark metrics dashboard with gauge charts, color-coded tables, model sizes"
```

---

## Phase 6: New Tech Stack Page

### Task 6.1: Create `demo/pages/5_Tech_Stack.py`

**Files:**
- Create: `demo/pages/5_Tech_Stack.py`

**Step 1: Create the file with:**

```python
"""Tech Stack & Methodology page for the Edge-AI Document Processing demo."""

import streamlit as st
from utils import metric_card, section_header, setup_page

setup_page("Tech Stack", "🔧")

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
    ("🧠", "ONNX Runtime", "Inference Engine", "Cross-platform ML inference. Runs all 4 models on-device with CPU."),
    (
        "📊",
        "EfficientNet-Lite0",
        "Document Classifier",
        "Lightweight CNN architecture. Float16 quantized to 6.5 MB.",
    ),
    (
        "🏷️",
        "DistilBERT German",
        "NER Extractor",
        "German-language transformer for named entity recognition. INT8 quantized.",
    ),
    ("👁️", "RapidOCR", "Text Extraction", "PaddleOCR-based ONNX bundles for text detection and recognition."),
    (
        "🔥",
        "PyTorch + HuggingFace",
        "Training Framework",
        "Two-phase transfer learning for classifier. HF Trainer for NER.",
    ),
    (
        "✅",
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
    ("🧪", "372", "Tests Passing", "green"),
    ("📁", "28", "Test Files", "cyan"),
    ("🔍", "Ruff", "Linter", "purple"),
    ("📦", "uv", "Package Manager", "amber"),
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
```

**Step 2: Run ruff and commit**

```bash
uv run ruff check demo/pages/5_Tech_Stack.py --fix
git add demo/pages/5_Tech_Stack.py
git commit -m "feat: new tech stack page with technology cards, methodology, data pipeline stats"
```

---

## Phase 7: About Page Redesign

### Task 7.1: Rewrite About page with team cards and stats — rename to `6_About.py`

**Files:**
- Delete: `demo/pages/4_About.py`
- Create: `demo/pages/6_About.py`

**Step 1: Delete the old file and create new one**

```bash
rm demo/pages/4_About.py
```

**Step 2: Create `demo/pages/6_About.py` with:**

```python
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
    ("🧪", "372", "Tests Passing", "green"),
    ("📄", "3", "Document Types", "cyan"),
    ("🧠", "4", "ONNX Models", "purple"),
    ("📦", "~199 MB", "Total Size", "amber"),
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
├── api/                        # Service layer (Pydantic models, DocumentService)
├── edge_model/
│   ├── classification/         # EfficientNet-Lite0 classifier (train, export, config)
│   ├── extraction/             # DistilBERT NER models (train, export, postprocess)
│   └── inference/              # ONNX Runtime pipeline (classify → OCR → extract → validate)
├── ocr/                        # RapidOCR wrapper (preprocessing, engine, postprocessing)
├── mobile_app/src/             # CLI integration + ONNX Runtime Mobile guide
├── scripts/                    # Synthetic data generators (images + BIO text)
├── data/schemas/               # JSON Schema Draft-07 per document type
├── tests/                      # 372 tests (unit + integration + e2e)
├── demo/                       # This Streamlit website
├── docs/                       # Architecture + model pipeline documentation
├── config.yaml                 # Pipeline configuration (model paths, thresholds)
└── pyproject.toml              # Dependencies managed via uv""",
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
```

**Step 3: Run ruff and commit**

```bash
uv run ruff check demo/pages/6_About.py --fix
git add demo/pages/6_About.py
git rm demo/pages/4_About.py
git commit -m "feat: redesigned About page with team cards, stats, getting started guide"
```

---

## Phase 8: Cleanup & Gitignore Fix

### Task 8.1: Fix `.gitignore` to exclude `_label.json` files

**Files:**
- Modify: `.gitignore`

**Step 1: Add `data/samples/**/*_label.json` after the existing `data/samples/**/*.jsonl` line in `.gitignore`:**

The line to add (after line 25 `data/samples/**/*.jsonl`):
```
data/samples/**/*_label.json
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "fix: gitignore label JSON training artifacts"
```

---

## Phase 9: Verification

### Task 9.1: Run full test suite and lint check

**Step 1: Run ruff on all demo files**

```bash
uv run ruff check demo/ --fix
```

Expected: no errors (or auto-fixed).

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: 372 tests pass.

**Step 3: Start the demo and verify visually**

```bash
uv run streamlit run demo/Home.py
```

Expected: Dark-themed website at http://localhost:8501 with:
- Home: gradient hero, metric cards, pipeline steps, feature highlights, document type cards
- Live Demo: dark upload area, styled field cards, confidence badges
- Architecture: styled pipeline steps, component details, dark model specs table
- Models & Metrics: accuracy gauge, dark confusion matrix, NER gauges, color-coded entity tables
- Tech Stack: technology cards, training methodology, data pipeline stats, engineering badges
- About: team avatar cards, project stats, getting started guide, project structure

**Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: complete dark AI tech theme redesign — all pages verified"
```
