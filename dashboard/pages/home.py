import streamlit as st
import pandas as pd
from dashboard.utils import get_data

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Dashboard", layout="wide")

st.header("Thyroid Cancer Recurrence")

# Load data
X, y = get_data()

# Display
st.markdown("This dataset is aimed at predicting recurrance in well differentiated thyroid cancer from clinical and pathological features.")
st.markdown("Borzooei, S., Briganti, G., Golparian, M. et al. Machine learning for risk stratification of thyroid cancer patients: a 15-year cohort study. Eur Arch Otorhinolaryngol 281, 2095–2104 (2024). https://doi.org/10.1007/s00405-023-08299-w")
st.subheader("Data Overview")
st.dataframe(pd.concat([X, y], axis=1))