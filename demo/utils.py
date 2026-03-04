"""Shared helpers for the Streamlit demo app."""

import streamlit as st


def setup_page(title: str, icon: str) -> None:
    """Configure page settings for a Streamlit page."""
    st.set_page_config(
        page_title=f"{title} — Edge-AI Document Processing",
        page_icon=icon,
        layout="wide",
    )
