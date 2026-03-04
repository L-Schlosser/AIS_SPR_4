"""Live Demo page — upload or generate a document and process it through the pipeline."""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from PIL import Image

from demo.utils import setup_page

setup_page("Live Demo", "🔍")

st.title("Live Demo")
st.markdown("Upload a document image or generate a sample, then run it through the full pipeline.")

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

left_col, right_col = st.columns(2)

# ---- Left column: input ----
with left_col:
    st.subheader("Input")

    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPEG, PNG",
    )

    use_sample = st.checkbox("Generate sample document")
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
    st.subheader("Results")

    if input_image is None:
        st.info("Upload an image or generate a sample to see results.")
    else:
        pipeline, load_error = _load_pipeline()

        if load_error is not None:
            st.error(f"Could not load pipeline: {load_error}")
            st.warning(
                "Make sure the ONNX models are available and dependencies are installed.\n\n"
                "```bash\nuv sync --group demo --group ocr\n```"
            )
        else:
            # Run processing
            try:
                image_array = np.array(input_image)
                start_time = time.perf_counter()
                result = pipeline.process(image_array)
                elapsed = time.perf_counter() - start_time

                # Document type badge
                doc_type_str = result.document_type.value
                badge_funcs = {
                    "arztbesuchsbestaetigung": st.success,
                    "reisekostenbeleg": st.info,
                    "lieferschein": st.warning,
                }
                badge_func = badge_funcs.get(doc_type_str, st.write)
                badge_func(f"Document Type: **{doc_type_str}**")

                # Confidence bar
                st.markdown(f"**Confidence:** {result.confidence:.1%}")
                st.progress(result.confidence)

                # Processing time
                st.caption(f"Processing time: {elapsed:.2f}s")

                # Extracted fields
                st.markdown("---")
                if result.fields:
                    st.markdown("**Extracted Fields**")
                    for key, value in result.fields.items():
                        if key.startswith("_"):
                            continue
                        st.markdown(f"- **{key}:** {value}")
                else:
                    st.warning("No fields extracted (confidence may be below threshold).")

                # Raw JSON
                with st.expander("Raw JSON"):
                    st.json(result.model_dump(mode="json"))

                # OCR Text
                with st.expander("OCR Text"):
                    if result.raw_text:
                        st.text(result.raw_text)
                    else:
                        st.caption("No OCR text available.")

            except Exception as exc:
                st.error(f"Pipeline processing failed: {exc}")
