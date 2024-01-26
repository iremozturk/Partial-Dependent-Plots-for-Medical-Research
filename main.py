import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pdpbox import pdp
from ConvolutionalNeuralNetworks.ConvolutionalNeuralNetworks import create_cnn_model, predict_image, predict_cnn
from DecisionTree.DecisionTree import train_decision_tree, predict_decision_tree
from RandomForest.RandomForest import train_random_forest, predict_random_forest
from SupportVectorMachines.SupportVectorMachines import train_svm, predict_svm
import cv2
from sklearn.inspection import permutation_importance
from dataset_1 import clean_dataset1
from dataset_2 import clean_dataset2

# Load dataset 1
X_train_1, X_test_1, y_train_1, y_test_1, X_1 = clean_dataset1(r'C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv')

# Train and predict using Decision Tree for dataset 1
decision_tree_1, feature_importances_dt_1 = train_decision_tree(X_train_1, y_train_1)
prediction_dt_1 = predict_decision_tree(decision_tree_1, X_test_1)

# Train and predict using Random Forest for dataset 1
random_forest_1, feature_importances_rf_1 = train_random_forest(X_train_1, y_train_1)
prediction_rf_1 = predict_random_forest(random_forest_1, X_test_1)

# Train and predict using SVM for dataset 1
svm_model_1 = train_svm(X_train_1, y_train_1)
prediction_svm_1 = predict_svm(svm_model_1, X_test_1)

result_svm_1 = permutation_importance(svm_model_1, X_test_1, y_test_1, n_repeats=10, random_state=42)
feature_importances_svm_1 = result_svm_1.importances_mean

# Load dataset 2
output_csv_path = 'cleaned_dataset.csv'
file_path = r'C:\Users\iremo\PycharmProjects\pythonProject1\cancer patient data sets.csv'
X_train_2, X_test_2, y_train_2, y_test_2, X_2 = clean_dataset2(file_path, output_csv_path)

# Train and predict using Decision Tree for dataset 2
decision_tree_2, feature_importances_dt_2 = train_decision_tree(X_train_2, y_train_2)
prediction_dt_2 = predict_decision_tree(decision_tree_2, X_test_2)

# Train and predict using Random Forest for dataset 2
random_forest_2, feature_importances_rf_2 = train_random_forest(X_train_2, y_train_2)
prediction_rf_2 = predict_random_forest(random_forest_2, X_test_2)

# Train and predict using SVM for dataset 2
svm_model_2 = train_svm(X_train_2, y_train_2)
prediction_svm_2 = predict_svm(svm_model_2, X_test_2)

result_svm_2 = permutation_importance(svm_model_2, X_test_2, y_test_2, n_repeats=10, random_state=42)
feature_importances_svm_2 = result_svm_2.importances_mean

# Identify the top 2 features based on importance for each model
top_n = 2  # Change this to 2 to select the top 2 features
important_features_dt_1 = np.argsort(feature_importances_dt_1)[::-1][:top_n]
important_features_rf_1 = np.argsort(feature_importances_rf_1)[::-1][:top_n]
important_features_svm_1 = np.argsort(feature_importances_svm_1)[::-1][:top_n]

important_features_dt_2 = np.argsort(feature_importances_dt_2)[::-1][:top_n]
important_features_rf_2 = np.argsort(feature_importances_rf_2)[::-1][:top_n]
important_features_svm_2 = np.argsort(feature_importances_svm_2)[::-1][:top_n]

# Convert X_train back to a DataFrame and extract the names of the top 2 features for each model
X_train_df_dt_1 = pd.DataFrame(X_train_1, columns=important_features_dt_1)
feature_names_dt_1 = X_train_df_dt_1.columns

X_train_df_rf_1 = pd.DataFrame(X_train_1, columns=important_features_rf_1)
feature_names_rf_1 = X_train_df_rf_1.columns

X_train_df_svm_1 = pd.DataFrame(X_train_1, columns=important_features_svm_1)
feature_names_svm_1 = X_train_df_svm_1.columns
#X_train_cnn=pd.DataFrame(X_train_1,)
# Convert X_train back to a DataFrame and extract the names of the top N features for Decision Tree (dataset 2)
X_train_df_dt_2 = pd.DataFrame(X_train_2, columns=important_features_dt_2)
feature_names_dt_2 = X_train_df_dt_2.columns

X_train_df_rf_2 = pd.DataFrame(X_train_2, columns=important_features_rf_2)
feature_names_rf_2 = X_train_df_rf_2.columns

X_train_df_svm_2 = pd.DataFrame(X_train_2, columns=important_features_svm_2)
feature_names_svm_2 = X_train_df_svm_2.columns

pdp_images_dt_2 = []


# Display PDP plots for Decision Tree (dataset 1)
for feature in feature_names_dt_1:
    pdp_feature_dt_1 = pdp.pdp_isolate(model=decision_tree_1, dataset=X_train_df_dt_1, model_features=feature_names_dt_1, feature=feature)
    if pdp_feature_dt_1 is not None and not np.isnan(pdp_feature_dt_1.pdp).all():
        pdp.pdp_plot(pdp_feature_dt_1, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Decision Tree)')
        pdp_image_path = f'pdp_dt_1_{feature}.png'
        plt.savefig(pdp_image_path)
        plt.close()
        pdp_images_dt_2.append(pdp_image_path)
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Display PDP plots for Random Forest (dataset 1)
for feature in feature_names_rf_1:
    pdp_feature_rf_1 = pdp.pdp_isolate(model=random_forest_1, dataset=X_train_df_rf_1, model_features=feature_names_rf_1, feature=feature)
    if pdp_feature_rf_1 is not None and not np.isnan(pdp_feature_rf_1.pdp).all():
        pdp.pdp_plot(pdp_feature_rf_1, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Random Forest)')
        pdp_image_path = f'pdp_rf_1_{feature}.png'
        plt.savefig(pdp_image_path)
        plt.close()
        pdp_images_dt_2.append(pdp_image_path)
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Display PDP plots for SVM (dataset 1)
for feature in feature_names_svm_1:
    pdp_feature_svm_1 = pdp.pdp_isolate(model=svm_model_1, dataset=X_train_df_svm_1, model_features=feature_names_svm_1, feature=feature)
    if pdp_feature_svm_1 is not None and not np.isnan(pdp_feature_svm_1.pdp).all():
        pdp.pdp_plot(pdp_feature_svm_1, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Support Vector Machines)')
        pdp_image_path = f'pdp_svm_1_{feature}.png'
        plt.savefig(pdp_image_path)
        plt.close()
        pdp_images_dt_2.append(pdp_image_path)
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Display PDP plots for Decision Tree (dataset 2)
for feature in feature_names_dt_2:
    if feature in [13, 17] and np.isnan(X_train_df_dt_2[feature]).all():
        print(f"Skipping PDP plot for feature {feature} due to NaN values.")
        continue

    print(f"Creating PDP plot for feature: {feature}")

    pdp_feature_dt_2 = pdp.pdp_isolate(model=decision_tree_2, dataset=X_train_df_dt_2,
                                       model_features=feature_names_dt_2, feature=feature)

    if pdp_feature_dt_2 is not None and not np.isnan(pdp_feature_dt_2.pdp).all():
        pdp.pdp_plot(pdp_feature_dt_2, feature_name=feature, cluster=True, n_cluster_centers=30)
        pdp_image_path = f'pdp_dt_2_{feature}.png'
        plt.savefig(pdp_image_path)
        plt.close()
        pdp_images_dt_2.append(pdp_image_path)
        print(f"Saved PDP plot for feature {feature}")
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")


# Display PDP plots for Random Forest (dataset 2)
for feature in feature_names_rf_2:
    if feature == 13 and np.isnan(X_train_df_rf_2[feature]).all():
        print(f"Skipping PDP plot for feature {feature} due to NaN values.")
        continue

    # Check if there are any non-NaN values in the feature
    if not np.isnan(X_train_df_rf_2[feature]).any():
        pdp_feature_rf_2 = pdp.pdp_isolate(model=random_forest_2, dataset=X_train_df_rf_2, model_features=feature_names_rf_2, feature=feature)

        if pdp_feature_rf_2 is not None:
            pdp.pdp_plot(pdp_feature_rf_2, feature_name=feature)
            plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Random Forest)')
            pdp_image_path = f'pdp_rf_2_{feature}.png'
            plt.savefig(pdp_image_path)
            plt.close()
            pdp_images_dt_2.append(pdp_image_path)
        else:
            print(f"Skipping PDP plot for feature {feature} as no data is available.")
    else:
        print(f"Skipping PDP plot for feature {feature} due to NaN values.")

# Display PDP plots for SVM (dataset 2)
for feature in feature_names_svm_2:
    if np.isnan(X_train_df_svm_2[feature]).all():
        print(f"Skipping PDP plot for feature {feature} due to NaN values.")
        continue

    pdp_feature_svm_2 = pdp.pdp_isolate(model=svm_model_2, dataset=X_train_df_svm_2, model_features=feature_names_svm_2, feature=feature)

    if pdp_feature_svm_2 is not None:
        pdp.pdp_plot(pdp_feature_svm_2, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with SVM (Dataset 2)')
        pdp_image_path = f'pdp_svm_2_{feature}.png'
        plt.savefig(pdp_image_path)
        plt.close()
        pdp_images_dt_2.append(pdp_image_path)
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Train and predict using CNN for dataset 2
cnn_model_1, X_train_cnn_1 = create_cnn_model(X_train_1, y_train_1)
prediction_cnn = predict_cnn(cnn_model_1, X_train_cnn_1)

# Train and predict using CNN for dataset 2
cnn_model_2, X_train_cnn_2 = create_cnn_model(X_train_2, y_train_2)
prediction_cnn = predict_cnn(cnn_model_2, X_train_cnn_2)

# Example of using the model to predict multiple images
image_paths = [f'pdp_rf_1_{feature_names_rf_1[0]}.png',  # Using attribute names instead of numerical indices
               f'pdp_rf_1_{feature_names_rf_1[1]}.png',
               f'pdp_dt_1_{feature_names_dt_1[0]}.png',
               f'pdp_dt_1_{feature_names_dt_1[1]}.png',
               f'pdp_svm_1_{feature_names_svm_1[0]}.png',
               f'pdp_svm_1_{feature_names_svm_1[1]}.png']

# Display the prediction result for each attribute
for i, feature in enumerate(feature_names_dt_1):
    # Get the corresponding image path
    image_path = image_paths[i]

    # Extract the feature name from the image path
    feature_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[-1]

    print(f"Combined Prediction Result (CNN) for {feature}: {prediction_cnn[i]}")

    # Save the prediction result as a PNG image with the feature name
    result_image_path = f'prediction_result_{feature_name}.png'
    plt.imshow(np.squeeze(prediction_cnn[i], axis=0))
    plt.title(f"Prediction Result (CNN) for {feature_name}")
    plt.savefig(result_image_path)
    plt.close()

    print(f"Saved prediction result image at {result_image_path}")
