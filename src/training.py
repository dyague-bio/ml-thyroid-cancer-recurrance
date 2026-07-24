import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.preprocessing import get_col_types, build_preprocessor

def build_pipelines(prep):
    """
    Build pipelines for each model with preprocessor and classifier.
    Parameters:
        - prep: ColumnTransformer object for preprocessing the data.
    Returns three Pipeline objects: pipe_knn, pipe_rf, pipe_svm.

    """
    pipe_knn = Pipeline([
        ("preprocess", prep),
        ("knn", KNeighborsClassifier())
    ])

    pipe_rf = Pipeline([
        ("preprocess", prep),
        ("rf", RandomForestClassifier())
    ])

    pipe_svm = Pipeline([
        ("preprocess", prep),
        ("svm", SVC())
    ])

    return pipe_knn, pipe_rf, pipe_svm

def define_param_grids():
    """
    Define hyperparameter grids for each model.
    Returns three dictionaries: param_knn, param_rf, param_svm.
    """
    param_knn = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }

    param_rf = {
        "rf__n_estimators": [100, 200, 300],
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5, 10]
    }

    param_svm = {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"]
    }

    return param_knn, param_rf, param_svm

def train_models(X, y, cv):
    """
    Train KNN, Random Forest, and SVM models using GridSearchCV.
    Parameters:
        - X: Features of the dataset.
        - y: Target labels of the dataset.
        - cv: Number of cross-validation folds.
    Returns a dictionary of best models with their best parameters and cross-validation scores, along with the test set (X_test, y_test).
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat, num = get_col_types(X_train)
    prep = build_preprocessor(cat, num)
    pipe_knn, pipe_rf, pipe_svm = build_pipelines(prep)
    param_knn, param_rf, param_svm = define_param_grids()

    models = {
        "KNN": (pipe_knn, param_knn),
        "RF":  (pipe_rf,  param_rf),
        "SVM": (pipe_svm, param_svm)
    }

    grids = {}

    for model, (pipe, param) in models.items():
        grid = GridSearchCV(pipe, param, cv = cv, n_jobs=-1)
        print(f"Training {model}...")
        grid.fit(X_train, y_train)
        grids[model] = grid

    best_models = {
        "KNN": {
            "best_model": grids["KNN"].best_estimator_,
            "cv_score": grids["KNN"].best_score_,
            "best_params": grids["KNN"].best_params_
        },
        "RandomForest": {
            "best_model": grids["RF"].best_estimator_,
            "cv_score": grids["RF"].best_score_,
            "best_params": grids["RF"].best_params_
        },
        "SVM": {
            "best_model": grids["SVM"].best_estimator_,
            "cv_score": grids["SVM"].best_score_,
            "best_params": grids["SVM"].best_params_
        }
    }

    return best_models, X_test, y_test