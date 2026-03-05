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
    accuracy = classifier_metrics["accuracy"]
    per_class = classifier_metrics["per_class"]

    # Accuracy gauge — centered, moderate width
    g1, g2, g3 = st.columns([1, 2, 1])
    with g2:
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
        dark_plotly_layout(fig_gauge, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Per-class F1 cards — full width row
    class_icons = {"arztbesuchsbestaetigung": "🏥", "lieferschein": "📦", "reisekostenbeleg": "🧾"}
    class_labels = {"arztbesuchsbestaetigung": "Arztbesuch", "lieferschein": "Lieferschein", "reisekostenbeleg": "Reisekosten"}
    class_colors = {"arztbesuchsbestaetigung": "cyan", "lieferschein": "green", "reisekostenbeleg": "purple"}
    pcols = st.columns(3)
    for col, c_name in zip(pcols, per_class):
        with col:
            f1 = per_class[c_name]["f1"]
            icon = class_icons.get(c_name, "📄")
            color = class_colors.get(c_name, "cyan")
            label = class_labels.get(c_name, c_name[:12])
            st.markdown(
                metric_card(icon, f"{f1:.1%}", f"F1 {label}", color),
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
        fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
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
