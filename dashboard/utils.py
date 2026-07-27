import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.load_data import load_thyroid_data


# Load data
@st.cache_data
def get_data():
    X, y, metadata = load_thyroid_data()
    return X, y

# Function to create EDA plots
def create_plot(df, x, y):
    if df[x].dtype in ["int64", "float64"] and df[y].dtype in ["int64", "float64"]:
        st.write(f"Scatter plot of {x} vs {y}")
        st.scatter_chart(df[[x, y]])
    elif df[x].dtype in ["int64", "float64"] and df[y].dtype == "object":
        st.write(f"Distribution of {x} by {y}")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=x, kde=True, hue=y, bins=30, palette="colorblind")

        ax.set_title(f'{x} distribution by {y}', fontsize=11)

        ax.legend(
            title=y,
            labels = (df[y].unique()[::-1]),
            loc="center right"
        )
        st.pyplot(fig)
        