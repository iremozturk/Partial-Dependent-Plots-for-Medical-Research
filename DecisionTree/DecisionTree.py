from sklearn.tree import DecisionTreeClassifier
import numpy as np
def train_decision_tree(X, y, num_top_features):
    """
    Train a decision tree classifier.

    Parameters:
        X (array-like): Training features.
        y (array-like): Training labels.
        num_top_features (int): Number of top features to select.

    Returns:
        model (DecisionTreeClassifier): Trained decision tree model.
        top_features (array): Indices of top features.
    """
    # Create and train a decision tree model
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)

    # Extract feature importances
    feature_importances = decision_tree.feature_importances_

    # Rank features based on importance scores
    feature_indices = np.argsort(feature_importances)[::-1]  # Sort in descending order
    top_features = feature_indices[:num_top_features]  # Select top features

    # Return the trained decision tree model and top features
    return decision_tree, top_features


def predict_decision_tree(model, X_test):
    """
    Make predictions using a trained decision tree model.

    Parameters:
        model (DecisionTreeClassifier): Trained decision tree model.
        X_test (array-like): Test features.

    Returns:
        prediction (array): Predicted labels.
    """
    # Implement Decision Tree prediction logic using the provided model
    prediction = model.predict(X_test)
    return prediction
