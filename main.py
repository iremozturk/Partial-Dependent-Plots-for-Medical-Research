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
from sklearn.inspection import permutation_importance
from dataset_1 import clean_dataset1
from dataset_2 import clean_dataset2
from SupportVectorMachines.SupportVectorMachines import calculate_svm_feature_importance

# Load dataset 1
X_train_1, X_test_1, y_train_1, y_test_1, X_1 = clean_dataset1(r'C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv')

# Train and predict using Decision Tree for dataset 1
decision_tree_1, feature_importances_dt_1 = train_decision_tree(X_train_1, y_train_1)
prediction_dt_1 = predict_decision_tree(decision_tree_1, X_test_1)

# Train and predict using Random Forest for dataset 1
random_forest_1, feature_importances_rf_1 = train_random_forest(X_train_1, y_train_1)
prediction_rf_1 = predict_random_forest(random_forest_1, X_test_1)

# Train and predict using SVM for dataset 1
svm_model_1, feature_importances_svm_1 = train_svm(X_train_1, y_train_1)
prediction_svm_1 = predict_svm(svm_model_1, X_test_1)

result_svm_1 = permutation_importance(svm_model_1, X_test_1, y_test_1, n_repeats=10, random_state=42)
feature_importances_svm_1 = result_svm_1.importances_mean

# Load dataset 2
output_csv_path_2 = 'cleaned_dataset_2.csv'
file_path_2 = r'C:\Users\iremo\PycharmProjects\pythonProject1\cancer patient data sets.csv'
X_train_2, X_test_2, y_train_2, y_test_2, X_2 = clean_dataset2(file_path_2, output_csv_path_2)

# Train and predict using Decision Tree for dataset 2
decision_tree_2, feature_importances_dt_2 = train_decision_tree(X_train_2, y_train_2)
prediction_dt_2 = predict_decision_tree(decision_tree_2, X_test_2)

# Train and predict using Random Forest for dataset 2
random_forest_2, feature_importances_rf_2 = train_random_forest(X_train_2, y_train_2)
prediction_rf_2 = predict_random_forest(random_forest_2, X_test_2)

# Train and predict using SVM for dataset 2
svm_model_2, feature_importances_svm_2 = train_svm(X_train_2, y_train_2)
prediction_svm_2 = predict_svm(svm_model_2, X_test_2)

result_svm_2 = permutation_importance(svm_model_2, X_test_2, y_test_2, n_repeats=10, random_state=42)
# Calculate feature importance for SVM (dataset 2)
feature_importances_svm_2 = calculate_svm_feature_importance(svm_model_2)

# Identify the top 2 features based on importance for SVM in dataset 2
top_features_svm_2 = calculate_svm_feature_importance(X_train_2, y_train_2)

# Create DataFrames containing only the top 2 features for SVM and dataset 2
X_train_df_svm_2 = pd.DataFrame(X_train_2[:, top_features_svm_2], columns=top_features_svm_2)

# Display PDP plots for SVM (dataset 2) with the top 2 features
for feature in X_train_df_svm_2.columns:
    pdp_feature_svm_2 = pdp.pdp_isolate(model=svm_model_2, dataset=X_train_df_svm_2, model_features=X_train_df_svm_2.columns, feature=feature)
    if pdp_feature_svm_2 is not None and not np.isnan(pdp_feature_svm_2.pdp).all():
        pdp.pdp_plot(pdp_feature_svm_2, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with SVM (Dataset 2)')
        plt.show()
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Train and predict using CNN for dataset 2
cnn_model_1, X_train_cnn_1 = create_cnn_model(X_train_1, y_train_1)
prediction_cnn = predict_cnn(cnn_model_1, X_train_cnn_1)

# Train and predict using CNN for dataset 2
cnn_model_2, X_train_cnn_2 = create_cnn_model(X_train_2, y_train_2)
prediction_cnn_2 = predict_cnn(cnn_model_2, X_train_cnn_2)

# Example of using the model to predict multiple images
image_paths_2 = [f'pdp_rf_2_{feature_names_rf_2[0]}.png',  # Using attribute names instead of numerical indices
               f'pdp_rf_2_{feature_names_rf_2[1]}.png',
               f'pdp_dt_2_{feature_names_dt_2[0]}.png',
               f'pdp_dt_2_{feature_names_dt_2[1]}.png',
               f'pdp_svm_2_{feature_names_svm_2[0]}.png',
               f'pdp_svm_2_{feature_names_svm_2[1]}.png']

# Display the prediction result for each attribute
for i, feature in enumerate(feature_names_dt_2):
    # Get the corresponding image path
    image_path = image_paths_2[i]

    # Extract the feature name from the image path
    feature_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[-1]

    print(f"Combined Prediction Result (CNN) for {feature}: {prediction_cnn_2[i]}")

    # Save the prediction result as a PNG image with the feature name
    result_image_path = f'prediction_result_{feature_name}_2.png'
    plt.imshow(np.squeeze(prediction_cnn_2[i], axis=0))
    plt.title(f"Prediction Result (CNN) for {feature_name}")
    plt.savefig(result_image_path)
    plt.close()

    print(f"Saved prediction result image at {result_image_path}")

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
