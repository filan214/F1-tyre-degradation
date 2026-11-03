"""Entry point of the read-only app over saved 2025 held-out results."""

import streamlit as st

from app_common import APP_DATA_DIR, MANIFEST_FILE

st.set_page_config(page_title="F1 Race Pace Forecasting", layout="wide")

if not (APP_DATA_DIR / MANIFEST_FILE).is_file():
    st.error("The app_data folder is missing. Run scripts/08_export_app_data.py to create it.")
    st.stop()

st.navigation([
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/race_explorer.py", title="Race explorer"),
    st.Page("views/where_model_helps.py", title="Where the model helps"),
    st.Page("views/method_and_limits.py", title="Method and limits"),
]).run()
