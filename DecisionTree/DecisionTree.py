from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X, y):
    """
    Train a decision tree classifier.

    Parameters:
        X (array-like): Training features.
        y (array-like): Training labels.

    Returns:
        model (DecisionTreeClassifier): Trained decision tree model.
        feature_importances (array): Feature importances.
    """
    # Create and train a decision tree model
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)

    # Extract feature importances
    feature_importances = decision_tree.feature_importances_

    # Return the trained decision tree model
    return decision_tree, feature_importances


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
