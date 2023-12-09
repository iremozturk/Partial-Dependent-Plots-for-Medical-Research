# GradientBoostingModel.py

from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=None):
    """
    Train a gradient boosting model and return the trained model and feature importances.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data target.
    - n_estimators: Number of boosting stages to be run.
    - learning_rate: Step size to shrink the contribution of each tree.
    - random_state: Seed for random number generator.

    Returns:
    - GradientBoosting model
    - Feature importances
    """
    # Create a gradient boosting classifier
    gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    # Train the model
    gradient_boosting.fit(X_train, y_train)

    # Get feature importances
    feature_importances = gradient_boosting.feature_importances_

    return gradient_boosting, feature_importances
