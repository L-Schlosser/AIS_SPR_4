"""Live Demo page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("Live Demo", "🔍")

st.title("Live Demo")
st.write("Upload a document image to process it through the pipeline.")
