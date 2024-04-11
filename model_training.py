import numpy as np
import pandas as pd
from DecisionTree.DecisionTree import train_decision_tree, predict_decision_tree
from RandomForest.RandomForest import train_random_forest, predict_random_forest
from SupportVectorMachines.SupportVectorMachines import train_svm, predict_svm
from GeneralizedAdditiveModels.GeneralizedAdditiveModels import train_gam, predict_gam
from GradientBoostingModel.GradientBoostingModel import train_gradient_boosting, predict_gradient_boosting
from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
import matplotlib.pyplot as plt
from collections import defaultdict
from pdpbox import pdp, get_dataset, info_plots
from sklearn.model_selection import train_test_split
from PIL import Image
import os


def dynamic_import(dataset_number):
    module_name = f"dataset_{dataset_number}"
    print("Importing module:", module_name)
    module = __import__(module_name)
    print("Module:", module)
    func_name = f"clean_dataset{dataset_number}"
    print("Function name:", func_name)
    cleaning_function = getattr(module, func_name)
    print("Cleaning function:", cleaning_function)
    return cleaning_function


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




def save_images(fig, model_name, feature_name, dataset_number, save_dir="images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Replace problematic characters in the feature name
    feature_name = feature_name.replace(">", "_gt_").replace("<", "_lt_").replace("/", "_")

    file_name = f"{model_name}_dataset_{dataset_number}_{feature_name}.svg"
    file_path = os.path.join(save_dir, file_name)
    fig.savefig(file_path)


def train_and_plot(dataset_number, num_top_features):
    # Dynamically import the dataset module and clean the dataset
    clean_dataset_function = dynamic_import(dataset_number)
    output_csv_path = f'cleaned_dataset_{dataset_number}.csv'
    X_train, X_test, y_train, y_test, X, column_names = clean_dataset_function(output_csv_path)
    # Exclude the 'binaryClass' column from column_names
    #column_names.remove('binaryClass')

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
        model, top_features = train_function(X_train, y_train, num_top_features=3)
        results[model_name] = (model, top_features)

    # Select top features based on importance scores
    top_features = select_top_features(results, num_top_features)

    # Before creating the DataFrame
    print("Shape of X:", X.shape)
    print("Length of column names:", len(column_names))

    # Convert X to DataFrame using column names
    X_df = pd.DataFrame(X, columns=column_names)

    # Inside your loop for plotting PDPs
    for model_name, (model, _) in results.items():
        for feature_index in top_features:
            print("Model:", model_name)
            print("Plotting PDP for feature index:", feature_index)  # Print feature index

            # Add debugging print statements
            print("Top features:", top_features)
            print("Length of column names:", len(column_names))

            # Check if feature_index is within the bounds of column_names
            if feature_index >= len(column_names):
                print("Index out of range:", feature_index)
                continue

            # Call pdp_isolate and inspect the return value
            pdp_feature = pdp.pdp_isolate(model=model, dataset=X_df, model_features=column_names,
                                          feature=column_names[feature_index])
            print("pdp_feature:", pdp_feature)  # Add this line to inspect the return value

            # Further logic for plotting PDPs
            if pdp_feature is not None:
                if isinstance(pdp_feature, list):  # Check if pdp_feature is a list
                    for isolate in pdp_feature:
                        if not np.isnan(isolate.pdp).all():
                            pdp.pdp_plot(isolate, feature_name=column_names[feature_index])
                            plt.title(f'Partial Dependence Plot for {column_names[feature_index]} with {model_name}')
                            plt.show()
                            # Save the current figure
                            fig = plt.gcf()

                            # Save the image
                            save_images(plt.gcf(), model_name, column_names[feature_index], dataset_number,
                                        r"C:\Users\iremo\PycharmProjects\pythonProject1\images")
                        else:
                            print(
                                f"Skipping PDP plot for feature {column_names[feature_index]} as no data is available.")
                else:  # Handle case when pdp_feature is not a list
                    if not np.isnan(pdp_feature.pdp).all():
                        pdp.pdp_plot(pdp_feature, feature_name=column_names[feature_index])
                        plt.title(f'Partial Dependence Plot for {column_names[feature_index]} with {model_name}')
                        plt.show()
                        # Save the current figure
                        fig = plt.gcf()

                        # Save the image
                        save_images(plt.gcf(), model_name, column_names[feature_index], dataset_number,
                                    r"C:\Users\iremo\PycharmProjects\pythonProject1\images")
                    else:
                        print(f"Skipping PDP plot for feature {column_names[feature_index]} as no data is available.")
    # Specify the path to your images directory
    # images_dir = "C:/Users/iremo/PycharmProjects/pythonProject1/images"

    # Create and train the CNN model using the images in the specified directory
    # model = ConvolutionalNeuralNetworks.create_cnn_model(images_dir)
