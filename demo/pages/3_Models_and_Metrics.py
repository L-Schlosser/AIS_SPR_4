"""Models & Metrics page for the Edge-AI Document Processing demo."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demo.utils import setup_page

setup_page("Models & Metrics", "📊")

METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


classifier_metrics = _load_json(METRICS_DIR / "classifier_metrics.json")
ner_metrics = _load_json(METRICS_DIR / "ner_metrics.json")

if classifier_metrics is None or ner_metrics is None:
    st.warning(
        "Metrics files not found. Run `uv run python -m demo.generate_metrics` to generate them."
    )

# ---------------------------------------------------------------------------
# Classification Performance
# ---------------------------------------------------------------------------
st.title("Models & Metrics")

st.header("Classification Performance")

if classifier_metrics:
    # Overall accuracy
    st.metric("Overall Accuracy", f"{classifier_metrics['accuracy']:.1%}")

    col1, col2 = st.columns(2)

    # Confusion matrix heatmap
    with col1:
        st.subheader("Confusion Matrix")
        labels = ["arztbesuchsbest.", "lieferschein", "reisekostenbeleg"]
        cm = classifier_metrics["confusion_matrix"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                texttemplate="%{z}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class bar chart
    with col2:
        st.subheader("Per-Class Metrics")
        per_class = classifier_metrics["per_class"]
        class_names = list(per_class.keys())
        precisions = [per_class[c]["precision"] for c in class_names]
        recalls = [per_class[c]["recall"] for c in class_names]
        f1s = [per_class[c]["f1"] for c in class_names]

        fig_cls = go.Figure()
        fig_cls.add_trace(go.Bar(name="Precision", x=class_names, y=precisions))
        fig_cls.add_trace(go.Bar(name="Recall", x=class_names, y=recalls))
        fig_cls.add_trace(go.Bar(name="F1", x=class_names, y=f1s))
        fig_cls.update_layout(
            barmode="group",
            yaxis=dict(range=[0.9, 1.005], title="Score"),
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    st.caption(f"Evaluated on {classifier_metrics['total_images']} sample images.")

# ---------------------------------------------------------------------------
# NER Extraction Performance
# ---------------------------------------------------------------------------
st.header("NER Extraction Performance")

if ner_metrics:
    # Per-document-type F1 bar chart
    doc_types = list(ner_metrics.keys())
    micro_f1s = [ner_metrics[dt]["micro_f1"] for dt in doc_types]

    fig_ner = px.bar(
        x=doc_types,
        y=micro_f1s,
        labels={"x": "Document Type", "y": "Micro F1"},
        text=[f"{v:.3f}" for v in micro_f1s],
    )
    fig_ner.update_traces(textposition="outside")
    fig_ner.update_layout(
        yaxis=dict(range=[0.9, 1.02]),
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_ner, use_container_width=True)

    # Entity-level breakdown per doc type
    st.subheader("Entity-Level Breakdown")
    for dt in doc_types:
        with st.expander(f"{dt} (F1: {ner_metrics[dt]['micro_f1']:.3f})"):
            entities = ner_metrics[dt]["per_entity"]
            rows = {
                "Entity": list(entities.keys()),
                "Precision": [f"{entities[e]['precision']:.4f}" for e in entities],
                "Recall": [f"{entities[e]['recall']:.4f}" for e in entities],
                "F1": [f"{entities[e]['f1']:.4f}" for e in entities],
            }
            st.table(rows)
            st.caption(f"Evaluated on {ner_metrics[dt]['num_samples']} samples.")

# ---------------------------------------------------------------------------
# Model Sizes
# ---------------------------------------------------------------------------
st.header("Model Sizes")

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

    st.metric("Total Model Size", f"{total_size:.1f} MB")

    fig_sizes = px.bar(
        x=sizes,
        y=names,
        orientation="h",
        labels={"x": "Size (MB)", "y": "Model"},
        text=[f"{s:.1f} MB" for s in sizes],
    )
    fig_sizes.update_traces(textposition="outside")
    fig_sizes.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_sizes, use_container_width=True)
except Exception as exc:
    st.warning(f"Could not load model sizes: {exc}")
    # Fallback static table
    st.table(
        {
            "Model": ["Classifier", "NER Arztbesuch", "NER Reisekosten", "NER Lieferschein"],
            "Size": ["~6.5 MB", "~64 MB", "~64 MB", "~64 MB"],
        }
    )
