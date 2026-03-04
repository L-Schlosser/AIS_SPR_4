"""Home page for the Edge-AI Document Processing demo."""

import streamlit as st

from demo.utils import setup_page

setup_page("Home", "🏠")

st.title("Edge-AI Document Processing")
st.write("Welcome to the Edge-AI Document Processing demo.")
