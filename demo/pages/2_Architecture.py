"""Architecture page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("Architecture", "🏗️")

st.title("🏗️ Architecture")
st.write("Overview of the Edge-AI document processing pipeline.")

# --- Section 1: Processing Pipeline ---
st.header("Processing Pipeline")

st.graphviz_chart(
    """
    digraph pipeline {
        rankdir=TD
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11]
        edge [fontname="Helvetica", fontsize=9]

        A [label="Document Image\\n(JPEG/PNG)", fillcolor="#e3f2fd"]
        B [label="Image Preprocessor\\n224×224, ImageNet norm", fillcolor="#e8f5e9"]
        C [label="EfficientNet-Lite0\\nClassifier", fillcolor="#fff3e0"]
        D [label="RapidOCR Engine\\nText Extraction", fillcolor="#fce4ec"]
        E [label="Low Confidence\\nResult", fillcolor="#ffebee", style="rounded,filled,dashed"]
        F [label="Document Type\\nRouter", shape=diamond, fillcolor="#f3e5f5"]
        G1 [label="DistilBERT NER\\nArztbesuch", fillcolor="#e0f7fa"]
        G2 [label="DistilBERT NER\\nReisekosten", fillcolor="#e0f7fa"]
        G3 [label="DistilBERT NER\\nLieferschein", fillcolor="#e0f7fa"]
        H [label="BIO Tag\\nPostprocessor", fillcolor="#f1f8e9"]
        I [label="JSON Schema\\nValidator", fillcolor="#fff8e1"]
        J [label="Structured JSON\\nResult", fillcolor="#e8eaf6"]

        A -> B
        B -> C
        C -> D [label="confidence ≥ 0.7"]
        C -> E [label="confidence < 0.7", style=dashed]
        D -> F
        F -> G1 [label="arztbesuch"]
        F -> G2 [label="reisekosten"]
        F -> G3 [label="lieferschein"]
        G1 -> H
        G2 -> H
        G3 -> H
        H -> I
        I -> J
    }
    """,
    use_container_width=True,
)

# --- Section 2: Pipeline Steps ---
st.header("Pipeline Steps")

with st.expander("1. Image Preprocessing", expanded=False):
    st.markdown(
        """
**Module:** `edge_model/inference/preprocessor.py`

- Resize input image to **224×224** pixels
- Normalize with ImageNet mean/std: `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- Convert HWC → CHW layout and add batch dimension
- Output: `(1, 3, 224, 224)` float32 numpy array
"""
    )

with st.expander("2. Document Classification", expanded=False):
    st.markdown(
        """
**Model:** EfficientNet-Lite0 (pretrained on ImageNet, fine-tuned)

- **Input:** `(1, 3, 224, 224)` float32 image tensor
- **Output:** `(1, 3)` logits → softmax → class probabilities
- **Classes:** Arztbesuchsbestätigung, Reisekostenbeleg, Lieferschein
- **Training:** Two-phase transfer learning (frozen backbone → full fine-tuning)
- **Confidence threshold:** 0.7 (below → rejected as low confidence)
"""
    )

with st.expander("3. OCR Text Extraction", expanded=False):
    st.markdown(
        """
**Library:** RapidOCR (PaddleOCR ONNX bundles)

- **Preprocessing:** Grayscale conversion, adaptive thresholding, sharpening
- **Engine:** Wraps RapidOCR for text region detection + text recognition
- **Postprocessing:** Sorts regions top-to-bottom / left-to-right, merges into readable text
- Accepts any input image size
"""
    )

with st.expander("4. Named Entity Recognition (NER)", expanded=False):
    st.markdown(
        """
**Model:** DistilBERT-base-german-cased (one fine-tuned model per document type)

- **Input:** Tokenized OCR text (max 256 tokens)
- **Output:** BIO tag sequence per token
- Separate model for each document type, trained on type-specific entities
- Uses Hugging Face tokenizer for subword tokenization
"""
    )

with st.expander("5. Postprocessing", expanded=False):
    st.markdown(
        """
**Module:** `edge_model/extraction/postprocess.py`

- Converts BIO tag sequences → merged field values
- Handles multi-token entities (B-tag starts, I-tag continues)
- Type-specific postprocessors for date/time parsing, number extraction
- Output: structured dictionary of extracted fields
"""
    )

with st.expander("6. Schema Validation", expanded=False):
    st.markdown(
        """
**Module:** `edge_model/inference/validator.py`

- Validates extracted fields against JSON Schema (Draft-07)
- One schema per document type in `data/schemas/`
- Ensures output correctness before returning results
- Returns `ProcessingResult` with validation status
"""
    )

# --- Section 3: Model Details ---
st.header("Model Details")

st.table(
    {
        "Model": [
            "Classifier",
            "NER Arztbesuch",
            "NER Reisekosten",
            "NER Lieferschein",
        ],
        "Architecture": [
            "EfficientNet-Lite0",
            "DistilBERT German",
            "DistilBERT German",
            "DistilBERT German",
        ],
        "Format": ["ONNX", "ONNX", "ONNX", "ONNX"],
        "Size": ["6.5 MB", "64 MB", "64 MB", "64 MB"],
        "Quantization": ["Float16", "INT8", "INT8", "INT8"],
    }
)

st.info("**Total on-device footprint:** ~199 MB (classifier + 3 NER extractors)")
