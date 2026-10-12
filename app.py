"""Stockflow dashboard entry point: ``streamlit run app.py``.

Pages live in ``src/dashboard.py``. The app only reads pipeline outputs (run ``make run``
first); the one thing it writes is the planners' approve / reject decision log.
"""

import streamlit as st

st.set_page_config(page_title="Stockflow", layout="wide", initial_sidebar_state="auto")

from src.dashboard import main  # noqa: E402

main()
