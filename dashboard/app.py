import sys
import os

# Add project root to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import streamlit as st
import pandas as pd
from src.load_data import load_thyroid_data

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Dashboard", layout="wide")

# Helper functions

...

# Load data
X, y, metadata = load_thyroid_data()

# Input widgets/sidebar
with st.sidebar:
    st.title("Thyroid Cancer Recurrence Dashboard")
    st.header("Settings")

    model_selection = st.selectbox(
        "Select a model:",("KNN", "RF", "SVM"))
    ...

# Display
st.subheader("Data Overview")
st.dataframe(pd.concat([X, y], axis=1))