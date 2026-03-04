"""Home page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("Home", "🏠")

st.title("Edge-AI Document Processing for BMD Go")
st.markdown("#### On-device document classification and field extraction pipeline")

# --- Key Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Document Types", "3")
col2.metric("ONNX Models", "4")
col3.metric("Total Size", "~199 MB")

# --- Architecture Flow Diagram ---
st.subheader("Processing Pipeline")
st.graphviz_chart(
    """
    digraph pipeline {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#f0f2f6",
              fontname="Helvetica", fontsize=11];
        edge [color="#666666"];

        A [label="Document\\nImage"];
        B [label="Preprocessor"];
        C [label="EfficientNet\\nClassifier"];
        D [label="RapidOCR"];
        E [label="DistilBERT\\nNER"];
        F [label="Postprocessor"];
        G [label="JSON Schema\\nValidator"];
        H [label="Structured\\nJSON"];

        A -> B -> C -> D -> E -> F -> G -> H;
    }
    """
)

# --- Supported Document Types ---
st.subheader("Supported Document Types")
st.table(
    {
        "Document Type": [
            "Arztbesuchsbestaetigung",
            "Reisekostenbeleg",
            "Lieferschein",
        ],
        "Description": [
            "Medical visit confirmation",
            "Travel expense receipt",
            "Delivery note",
        ],
        "Key Fields": [
            "Patient, doctor, facility, date, time",
            "Vendor, date, amount, currency, VAT",
            "Delivery number, sender, recipient, items",
        ],
    }
)

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    st.markdown(
        """
Use the sidebar pages to explore:
- **Live Demo** — process documents in real time
- **Architecture** — pipeline design and model details
- **Models & Metrics** — performance benchmarks
- **About** — team and technology stack
"""
    )

    st.divider()
    st.header("How to Run")
    st.code(
        """\
# Install dependencies
uv sync --group demo --group ocr

# Start the demo
uv run streamlit run demo/Home.py""",
        language="bash",
    )
