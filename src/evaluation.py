import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def get_model_metrics(model, X_test, y_test):
    """
    Calculate metrics for a given model.
    Parameters:
        - model: Trained model.
        - X_test: Test features.
        - y_test: Test labels.
    Returns predicted labels, accuracy, and AUC.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    auc = roc_auc_score(y_test, y_score)

    return y_pred, acc, auc


def compare_models(models: dict, X_test, y_test):
    """
    Compare the performance of different models.
    Parameters:
        - models: Dictionary of trained models.
        - X_test: Test features.
        - y_test: Test labels.
    Returns a DataFrame with results, confusion matrices, and classification reports.
    """
    results = []
    confusion_matrices = {}
    reports = {}

    for name, model in models.items():
        y_pred, acc, auc = get_model_metrics(model["best_model"], X_test, y_test)
    
        results.append({
            "Model": name,
            "Accuracy": acc,
            "AUC": auc
        })

        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        reports[name] = classification_report(y_test, y_pred)
    
    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    
    return results_df, confusion_matrices, reports