import streamlit as st
import pandas as pd
from dashboard.utils import get_data

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Dashboard", layout="wide")

st.markdown("# *Thyroid Cancer Recurrence*")

# Load data
X, y = get_data()

# Display
st.subheader("Data Overview")
st.dataframe(pd.concat([X, y], axis=1))