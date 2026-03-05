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
