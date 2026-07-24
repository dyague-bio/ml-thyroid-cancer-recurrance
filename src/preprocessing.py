import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_col_types(X):
    """
    Identifies the types of columns in the dataset.
    Returns a tuple of pd.Index objects: (categorical_cols, numeric_cols).
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=np.number).columns
    
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    """
    Builds a preprocessor for the dataset based on the types of columns.
    Categorical columns are one-hot encoded, and numeric columns are standardized.
    The remainder of the columns are passed through without transformation.
    Returns a ColumnTransformer object (prep).
    """
    prep = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )

    return prep