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

    /* Font + global text color */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        color: #E2E8F0;
    }

    /* Force all Streamlit text elements to be light */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6,
    .stText, label, .stSelectbox label, .stCheckbox label,
    [data-testid="stWidgetLabel"] {
        color: #E2E8F0 !important;
    }
    .stMarkdown a { color: #00D4FF !important; }

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
        padding: 1rem 0.6rem;
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
        font-size: clamp(1rem, 2vw, 1.6rem);
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .metric-card .metric-label {
        font-size: clamp(0.55rem, 1vw, 0.78rem);
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.03em;
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
