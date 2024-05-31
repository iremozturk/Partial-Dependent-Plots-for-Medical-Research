import numpy as np
import pandas as pd
from DecisionTree.DecisionTree import train_decision_tree, predict_decision_tree
from RandomForest.RandomForest import train_random_forest, predict_random_forest
from SupportVectorMachines.SupportVectorMachines import train_svm, predict_svm
from GeneralizedAdditiveModels.GeneralizedAdditiveModels import train_gam, predict_gam
from GradientBoostingModel.GradientBoostingModel import train_gradient_boosting, predict_gradient_boosting
from ConvolutionalNeuralNetworks.ConvolutionalNeuralNetworks import evaluate_models_with_cnn
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split

def dynamic_import(dataset_number):
    module_name = f"dataset_{dataset_number}"
    module = __import__(module_name)
    func_name = f"clean_dataset{dataset_number}"
    cleaning_function = getattr(module, func_name)
    return cleaning_function

def select_top_features(results, num_top_features):
    feature_importance_scores = defaultdict(float)
    for model_name, model in results.items():
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            sorted_indices = np.argsort(feature_importances)[::-1]
            top_indices = sorted_indices[:num_top_features]
            for index in top_indices:
                feature_importance_scores[index] += feature_importances[index]
    sorted_features = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [index for index, score in sorted_features[:num_top_features]]
    return top_features

def train_and_plot(dataset_number, num_top_features):
    clean_dataset_function = dynamic_import(dataset_number)
    output_csv_path = f'cleaned_dataset_{dataset_number}.csv'
    X_train, X_test, y_train, y_test, X, column_names = clean_dataset_function(output_csv_path)

    model_functions = {
        'DecisionTree': train_decision_tree,
        'RandomForest': train_random_forest,
        'SVM': train_svm,
        'GAM': train_gam,
        'GradientBoosting': train_gradient_boosting
    }
    results = {}
    for model_name, train_function in model_functions.items():
        model, _ = train_function(X_train, y_train, num_top_features)
        results[model_name] = model

    top_features = select_top_features(results, num_top_features)
    X_df = pd.DataFrame(X, columns=column_names)
    selected_features = [column_names[i] for i in top_features]

    num_classes = len(model_functions)
    evaluate_models_with_cnn(results, X_df, selected_features, num_classes)

#if __name__ == "__main__":
#    dataset_number = 5
#   num_top_features = 1
#train_and_plot(dataset_number, num_top_features)