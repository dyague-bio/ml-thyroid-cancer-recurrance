import streamlit as st
import pandas as pd
from dashboard.utils import get_data
from dashboard.utils import create_plot # FIX

X, y = get_data()
df = pd.concat([X, y], axis=1)

# Set page config
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.markdown("Select the variables to analyze")
    Var_x = st.selectbox("Select variable x", df.columns[:-1])
    Var_y = st.selectbox("Select variable y", df.columns[:-1])

# Display
st.markdown("# Exploratory Data Analysis (EDA)")
st.markdown(f"Number of samples: {df.shape[0]}")
st.markdown(f"Number of features: {df.shape[1] - 1}")
st.markdown(f"Target variable name: {df.columns[-1]}")
st.markdown(f"NA values: {df.isna().sum().sum()}")

st.subheader(f"{Var_x} vs {Var_y}")
create_plot(df, Var_x, Var_y)