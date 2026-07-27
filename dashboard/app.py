import sys
import os
import streamlit as st
import pandas as pd

# Add project root to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# Define pages
main_page = st.Page("pages/home.py", title="Thyroid Cancer Recurrence")
page_2 = st.Page("pages/eda.py", title="EDA", icon="📊")
page_3 = st.Page("pages/model_comparison.py", title="Model Comparison", icon="✨")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()