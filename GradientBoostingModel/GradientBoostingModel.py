import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def train_gradient_boosting(X_train, y_train, num_top_features):
    """
    Train a Gradient Boosting classifier and return the trained model and top features.

    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        num_top_features (int): Number of top features to select.

    Returns:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
        top_features (array-like): Indices of the top features.
    """
    # Create and train a Gradient Boosting model
    gradient_boosting = GradientBoostingClassifier()
    gradient_boosting.fit(X_train, y_train)

    # Extract feature importances
    feature_importances = gradient_boosting.feature_importances_

    # Select top features based on importance scores
    top_features = np.argsort(feature_importances)[-num_top_features:]

    # Return the trained Gradient Boosting model and top features
    return gradient_boosting, top_features

def predict_gradient_boosting(model, X_test):
    """
    Make predictions using a trained Gradient Boosting model.

    Parameters:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
        X_test (array-like): Test features.

    Returns:
        prediction (array): Predicted labels.
    """
    # Implement GBM prediction logic using the provided model
    prediction = model.predict(X_test)
    return prediction

