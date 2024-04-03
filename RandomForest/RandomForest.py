import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train, num_top_features, n_estimators=100, random_state=None):
    """
    Train a random forest model and return the trained model and top features.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data target.
    - num_top_features: Number of top features to select.
    - n_estimators: Number of trees in the forest.
    - random_state: Seed for random number generator.

    Returns:
    - RandomForest model
    - Top features
    """
    # Create a random forest classifier
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model
    random_forest.fit(X_train, y_train)

    # Get feature importances
    feature_importances = random_forest.feature_importances_

    # Select top features based on importance scores
    top_features = np.argsort(feature_importances)[-num_top_features:]

    return random_forest, top_features

def predict_random_forest(model, X_test):
    """
    Make predictions using a trained Random Forest model.

    Parameters:
        model (RandomForestClassifier): Trained Random Forest model.
        X_test (array-like): Test features.

    Returns:
        prediction (array): Predicted labels.
    """
    # Extracting only the model from the tuple
    rf_model = model[0]

    # Implement Random Forest prediction logic using the provided model
    prediction = rf_model.predict(X_test)
    return prediction
