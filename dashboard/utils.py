import streamlit as st
from src.load_data import load_thyroid_data

# Load data
@st.cache_data
def get_data():
    X, y, metadata = load_thyroid_data()
    return X, y

# 