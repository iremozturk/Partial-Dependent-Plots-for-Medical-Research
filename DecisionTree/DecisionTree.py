from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X, y):
    # Create and train a decision tree model
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X, y)

    # Extract feature importances
    feature_importances = decision_tree.feature_importances_

    # Return the trained decision tree model and feature importances
    return decision_tree, feature_importances

