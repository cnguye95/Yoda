"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Phase 0: just a placeholder page that proves the install works. Mode toggle,
progress display, and report download all come in Phase 8.
"""

import streamlit as st

# Set the browser tab title and the page layout. Streamlit must run this
# before any other st.* call.
st.set_page_config(page_title="Yoda", layout="centered")

# Page heading and a one-line description so the user knows what they opened.
st.title("Yoda — pre-earnings research assistant")
st.caption("Yoda is starting up...")

# Placeholder ticker input. Phase 8 will wire this up to the report generator;
# for now it just renders so we can confirm Streamlit boots end-to-end.
st.text_input("Ticker", placeholder="e.g. NFLX")
