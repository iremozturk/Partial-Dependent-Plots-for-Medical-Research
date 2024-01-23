import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from pdpbox import pdp
from ConvolutionalNeuralNetworks.ConvolutionalNeuralNetworks import create_cnn_model, predict_image
from DecisionTree.DecisionTree import train_decision_tree, predict_decision_tree
from RandomForest.RandomForest import train_random_forest, predict_random_forest
from SupportVectorMachines.SupportVectorMachines import train_svm, predict_svm
import cv2
from sklearn.inspection import permutation_importance
from dataset_1 import clean_dataset1

X_train, X_test, y_train, y_test, X = clean_dataset1(r'C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv')

# Train and predict using Decision Tree
decision_tree, feature_importances_dt = train_decision_tree(X_train, y_train)
prediction_dt = predict_decision_tree(decision_tree, X_test)

# Train and predict using Random Forest
random_forest, feature_importances_rf = train_random_forest(X_train, y_train)
prediction_rf = predict_random_forest(random_forest, X_test)

# Train and predict using SVM
svm_model = train_svm(X_train, y_train)
prediction_svm = predict_svm(svm_model, X_test)

result = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42)
feature_importances_svm = result.importances_mean


# Identify the top 2 features based on importance for each model
top_n = 2  # Change this to 2 to select the top 2 features
important_features_dt = np.argsort(feature_importances_dt)[::-1][:top_n]
important_features_rf = np.argsort(feature_importances_rf)[::-1][:top_n]
important_features_svm = np.argsort(feature_importances_svm)[::-1][:top_n]

# Convert X_train back to a DataFrame and extract the names of the top 2 features for each model
X_train_df_dt = pd.DataFrame(X_train, columns=important_features_dt)
feature_names_dt = X_train_df_dt.columns

X_train_df_rf = pd.DataFrame(X_train, columns=important_features_rf)
feature_names_rf = X_train_df_rf.columns

X_train_df_svm = pd.DataFrame(X_train, columns=important_features_svm)
feature_names_svm = X_train_df_svm.columns

# Display PDP plots for Decision Tree
for feature in feature_names_dt:
    pdp_feature_dt = pdp.pdp_isolate(model=decision_tree, dataset=X_train_df_dt, model_features=feature_names_dt, feature=feature)
    if not np.isnan(pdp_feature_dt.pdp).all():
        pdp.pdp_plot(pdp_feature_dt, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Decision Tree)')
        plt.savefig(f'pdp_dt_{feature}.png')
        plt.close()
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# ...

# Display PDP plots for Random Forest
for feature in feature_names_rf:
    pdp_feature_rf = pdp.pdp_isolate(model=random_forest, dataset=X_train_df_rf, model_features=feature_names_rf, feature=feature)
    if not np.isnan(pdp_feature_rf.pdp).all():
        pdp.pdp_plot(pdp_feature_rf, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Random Forest)')
        plt.savefig(f'pdp_rf_{feature}.png')
        plt.close()
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# ...


# Display PDP plots for Random Forest
for feature in feature_names_svm:
    pdp_feature_svm = pdp.pdp_isolate(model=svm_model, dataset=X_train_df_svm, model_features=feature_names_svm, feature=feature)
    if not np.isnan(pdp_feature_svm.pdp).all():
        pdp.pdp_plot(pdp_feature_svm, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass (Support Vector Machines)')
        plt.savefig(f'pdp_svm_{feature}.png')
        plt.close()
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")

# Display PDP plots for SVM


# Train and predict using CNN
cnn_model = create_cnn_model()
# Load  trained weights if available
# cnn_model.load_weights('weights.h5')

# Example of using the model to predict multiple images
image_paths = [r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_rf_0.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_rf_1.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_dt_0.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_dt_1.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_svm_0.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_svm_1.png']
for image_path in image_paths:
    prediction_cnn = predict_image(cnn_model, image_path)

    # Display the input image
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.show()

    # Display the prediction result
    print(f"Prediction Result (CNN) for {image_path}: {prediction_cnn}")

    # Save the prediction result as a PNG image
    result_image_path = f'prediction_result_{os.path.basename(image_path)}.png'
    plt.imshow(np.squeeze(prediction_cnn, axis=0))  # Assuming prediction_cnn is an image
    plt.title(f"Prediction Result (CNN) for {image_path}")
    plt.savefig(result_image_path)
    plt.close()

    print(f"Saved prediction result image at {result_image_path}")
