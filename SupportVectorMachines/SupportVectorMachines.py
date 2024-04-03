import numpy as np
from sklearn.svm import SVC


def train_svm(X_train, y_train, num_top_features):
    """
    Train a support vector machine (SVM) model and return the trained model and top features.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data target.
    - num_top_features: Number of top features to select.

    Returns:
    - SVM model
    - Top features
    """
    # Implement your SVM training logic here
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Calculate feature importance based on the magnitude of support vector coefficients
    coef_magnitudes = abs(svm_model.dual_coef_)
    feature_importances = coef_magnitudes.sum(axis=0)

    # Select top features based on importance scores
    top_features = np.argsort(feature_importances)[-num_top_features:]

    return svm_model, top_features  # Return the SVM model along with top features

def predict_svm(model, X_test):
    """
    Make predictions using a trained SVM model.

    Parameters:
        model (SVC): Trained SVM model.
        X_test (array-like): Test features.

    Returns:
        prediction (array): Predicted labels.
    """
    # Implement SVM prediction logic using the provided model
    prediction = model.predict(X_test)
    return prediction
