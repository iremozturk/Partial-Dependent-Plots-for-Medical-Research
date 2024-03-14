from pygam import LinearGAM
from sklearn.metrics import mean_squared_error


def train_gam(X, y):
    # Train a GAM model
    gam = LinearGAM()
    gam.fit(X, y)
    feature_importances = gam.coef_  # Adjust this according to GAM's feature importance
    return gam, feature_importances
    return gam


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
