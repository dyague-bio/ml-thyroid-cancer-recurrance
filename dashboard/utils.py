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

    elif df[x].dtype == "object" and df[y].dtype in ["int64", "float64"]:
        st.write(f"Boxplot of {y} by {x}")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x, y=y, hue=x, ax=ax, palette="colorblind")

        ax.set_title(f'{y} by {x}', fontsize=11)
        st.pyplot(fig)

    elif df[x].dtype == "object" and df[y].dtype == "object":
        st.write(f"Proportion stacked bar plot of {x} vs {y}")
        fig, ax = plt.subplots()

        prop = (
            df.groupby(x)[y]
            .value_counts(normalize=True)
            .rename("prop")
            .reset_index()
        )

        pivot = prop.pivot(index=x, columns=y, values="prop").fillna(0)

        groups = pivot.index
        categories = pivot.columns

        palette = sns.color_palette("colorblind")
        bottom = [0] * len(groups)

        for cat, color in zip(categories, palette):
            values = pivot[cat].values
            ax.bar(groups, values, bottom=bottom, color=color, edgecolor="white", label=cat)
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_title(f'{x} vs {y}', fontsize=11)

        ax.legend(
            title=y,
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        st.pyplot(fig)
