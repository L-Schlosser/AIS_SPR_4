"""About page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("About", "ℹ️")

st.title("About")
st.write("Team information and project details.")
