import streamlit as st
from dashboard.utils import get_data

X, y = get_data()

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Dashboard", layout="wide")

st.markdown("# Exploratory Data Analysis (EDA)")
