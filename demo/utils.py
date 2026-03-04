"""Shared helpers for the Streamlit demo app."""

import streamlit as st

_CUSTOM_CSS = """
<style>
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Consistent font stack */
    html, body, [class*="css"] {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    /* Subtle header underline */
    h1 {
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.3em;
    }
</style>
"""

_FOOTER_HTML = """
<style>
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f2f6;
        text-align: center;
        padding: 8px 0;
        font-size: 0.8em;
        color: #666;
        z-index: 999;
        border-top: 1px solid #ddd;
    }
</style>
<div class="custom-footer">
    Edge-AI Document Processing for BMD Go — University Student Project
</div>
"""


def setup_page(title: str, icon: str) -> None:
    """Configure page settings, apply custom CSS, and inject footer."""
    st.set_page_config(
        page_title=f"{title} — Edge-AI Document Processing",
        page_icon=icon,
        layout="wide",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
