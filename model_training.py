import numpy as np
import pandas as pd
from DecisionTree.DecisionTree import train_decision_tree, predict_decision_tree
from RandomForest.RandomForest import train_random_forest, predict_random_forest
from SupportVectorMachines.SupportVectorMachines import train_svm, predict_svm
from GeneralizedAdditiveModels.GeneralizedAdditiveModels import train_gam, predict_gam
from GradientBoostingModel.GradientBoostingModel import train_gradient_boosting, predict_gradient_boosting
import matplotlib.pyplot as plt
from collections import defaultdict
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split
from PIL import Image

def dynamic_import(dataset_number):
    module_name = f"dataset_{dataset_number}"
    module = __import__(module_name)
    return getattr(module, f"clean_dataset{dataset_number}")


def select_top_features(results, num_top_features):
    feature_importance_scores = defaultdict(float)

    for model_name, (_, feature_importances) in results.items():
        if isinstance(feature_importances, np.ndarray):
            sorted_indices = np.argsort(feature_importances)[::-1]
            top_indices = sorted_indices[:num_top_features]
            for index in top_indices:
                feature_importance_scores[index] += feature_importances[index]
        else:
            for feature, importance in feature_importances.items():
                feature_importance_scores[feature] += importance

    sorted_features = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [index for index, _ in sorted_features[:num_top_features]]

    return top_features


def calculate_pdp(model, X, feature_of_interest, feature_values):
    pdp_values = []
    for value in feature_values:
        X_temp = X.copy()
        X_temp[:, feature_of_interest] = value
        predictions = model.predict_proba(X_temp)[:, 1]  # Assuming binary classification
        pdp_values.append(np.mean(predictions))
    return pdp_values

def save_images(images, model_name, column_name):
    for idx, img in enumerate(images):
        file_name = f"{model_name}_{column_name}_{idx}.png"
        image = Image.fromarray(img)
        image.save(file_name)

def train_and_plot(dataset_number, num_top_features):
    # Dynamically import the dataset module and clean the dataset
    clean_dataset_function = dynamic_import(dataset_number)
    output_csv_path = f'cleaned_dataset_{dataset_number}.csv'
    X_train, X_test, y_train, y_test, X, column_names = clean_dataset_function(output_csv_path)
    # Exclude the 'binaryClass' column from column_names
    column_names.remove('binaryClass')

    # Train the model
    model_functions = {
        'DecisionTree': (train_decision_tree, predict_decision_tree),
        'RandomForest': (train_random_forest, predict_random_forest),
        'SVM': (train_svm, predict_svm),
        'GAM': (train_gam, predict_gam),
        'GradientBoosting': (train_gradient_boosting, predict_gradient_boosting)
    }
    results = {}
    for model_name, (train_function, predict_function) in model_functions.items():
        model, feature_importances = train_function(X_train, y_train)
        results[model_name] = (model, feature_importances)

    # Select top features based on importance scores
    top_features = select_top_features(results, num_top_features)
    print("Feature indices:", list(range(len(column_names))))  # Print feature indices
    print("Column names:", column_names)  # Print column names

    # Convert X to DataFrame using column names
    X_df = pd.DataFrame(X, columns=column_names)

    print("Number of features:", len(column_names))  # Print the number of features
    print("Top features:", top_features)  # Print the list of top features
    print("Length of column names:", len(column_names))

    # Inside your loop for plotting PDPs
    for model_name, (model, _) in results.items():
        for feature_index in top_features:
            print("Model:", model_name)
            print("Plotting PDP for feature index:", feature_index)  # Print feature index
            print("Max feature index:", max(top_features))
            print("Shape of X_df:", X_df.shape)
            if feature_index >= X_df.shape[1]:
                print(f"Feature index {feature_index} is out of bounds.")
            else:
                feature_values = np.linspace(X_df.iloc[:, feature_index].min(), X_df.iloc[:, feature_index].max(), 600)
                pdp_feature = pdp.pdp_isolate(model=model, dataset=X_df, model_features=column_names,
                                              feature=column_names[feature_index])
                if pdp_feature is not None and not np.isnan(pdp_feature.pdp).all():
                    pdp.pdp_plot(pdp_feature, feature_name=column_names[feature_index])
                    plt.title(f'Partial Dependence Plot for {column_names[feature_index]} with {model_name}')
                    plt.show()
                    # Save the image
                    save_images([plt.gcf()], model_name, column_names[feature_index])
                else:
                    print(f"Skipping PDP plot for feature {column_names[feature_index]} as no data is available.")

