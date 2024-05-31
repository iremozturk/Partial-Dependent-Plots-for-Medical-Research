import numpy as np
from pygam import LinearGAM
from sklearn.metrics import mean_squared_error
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_gam(X_train, y_train, num_top_features):
    """
    Train a Generalized Additive Model (GAM) and return the trained model and top features.

    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        num_top_features (int): Number of top features to select.

    Returns:
        model (LinearGAM): Trained Generalized Additive Model.
        top_features (array-like): Indices of the top features.
    """
    # Train a GAM model
    gam = LinearGAM()
    gam.fit(X_train, y_train)

    # Calculate feature importances
    feature_importances = gam.coef_  # Adjust this according to GAM's feature importance

    # Select top features based on importance scores
    top_features = np.argsort(np.abs(feature_importances))[-num_top_features:]

    return gam, top_features


def predict_gam(model, X_test):

    # Make predictions using the trained GAM model
    prediction = model.predict(X_test)

    return prediction


def evaluate_gam(model, X_test, y_test):

    # Make predictions using the trained GAM model
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    return mse
