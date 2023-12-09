# SupportVectorMachines.py

from sklearn.svm import SVC

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', random_state=None):
    """
    Train a Support Vector Machine (SVM) classifier and return the trained model.

    Parameters:
    - X_train: Training data features.
    - y_train: Training data target.
    - kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.).
    - C: Regularization parameter.
    - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    - random_state: Seed for random number generator.

    Returns:
    - SVM model
    """
    # Create a SVM classifier
    svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

    # Train the model
    svm_classifier.fit(X_train, y_train)

    return svm_classifier
