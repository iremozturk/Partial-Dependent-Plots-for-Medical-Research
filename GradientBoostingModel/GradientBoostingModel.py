from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X, y):
    """
    Train a Gradient Boosting classifier.

    Parameters:
        X (array-like): Training features.
        y (array-like): Training labels.

    Returns:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
    """
    # Create and train a Gradient Boosting model
    gradient_boosting = GradientBoostingClassifier()
    gradient_boosting.fit(X, y)
    feature_importances = gradient_boosting.feature_importances_
    # Return the trained Gradient Boosting model
    return gradient_boosting, feature_importances

def predict_gradient_boosting(model, X_test):
    """
    Make predictions using a trained Gradient Boosting model.

    Parameters:
        model (GradientBoostingClassifier): Trained Gradient Boosting model.
        X_test (array-like): Test features.

    Returns:
        prediction (array): Predicted labels.
    """
    # Implement Gradient Boosting prediction logic using the provided model
    prediction = model.predict(X_test)
    return prediction
