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
