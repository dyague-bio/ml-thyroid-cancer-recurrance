from ucimlrepo import fetch_ucirepo

def load_thyroid_data():
    """
    Imports the 'Differentiated Thyroid Cancer Recurrence' dataset from the UCI Machine Learning Repository.
    Returns the features (X), target labels (y), and metadata of the dataset.
    """
    dataset = fetch_ucirepo(id=915)
    X = dataset.data.features
    y = dataset.data.targets
    return X, y, dataset.metadata
