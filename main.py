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
import tensorflow as tf


# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv(r'C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv')

# Convert columns with non-numeric values to numeric (0 and 1)
columns_to_convert = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                      'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
                      'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 'TT4 measured',
                      'T4U measured', 'FTI measured', 'TBG measured']

for column in columns_to_convert:
    df[column] = df[column].map({'f': 0, 't': 1})

df['binaryClass'] = df['binaryClass'].map({'N': 0, 'P': 1})

# Convert non-numeric values to NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns to float
df = df.astype(float)

# Separate features and target
X = df.drop(columns=['binaryClass'])
y = df['binaryClass']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict using Decision Tree
decision_tree, feature_importances_dt = train_decision_tree(X_train, y_train)
prediction_dt = predict_decision_tree(decision_tree, X_test)

# Train and predict using Random Forest
random_forest, feature_importances_rf = train_random_forest(X_train, y_train)
prediction_rf = predict_random_forest(random_forest, X_test)

# Train and predict using SVM
svm_model = train_svm(X_train, y_train)
prediction_svm = predict_svm(svm_model, X_test)

# Identify the top 2 features based on importance for each model
top_n = 2  # Change this to 2 to select the top 2 features
important_features_dt = np.argsort(feature_importances_dt)[::-1][:top_n]
important_features_rf = np.argsort(feature_importances_rf)[::-1][:top_n]

# Convert X_train back to a DataFrame and extract the names of the top 2 features for each model
X_train_df_dt = pd.DataFrame(X_train, columns=df.columns[important_features_dt])
feature_names_dt = X_train_df_dt.columns

X_train_df_rf = pd.DataFrame(X_train, columns=df.columns[important_features_rf])
feature_names_rf = X_train_df_rf.columns

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

# Display PDP plots for SVM (you can modify this based on your SVM model's interpretation)
# ...

# Train and predict using CNN
cnn_model = create_cnn_model()
# Load your trained weights if available
# cnn_model.load_weights('your_weights.h5')

# Example of using the model to predict multiple images
image_paths = [r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_rf_age.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_rf_sex.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_dt_age.png',
               r'C:\Users\iremo\PycharmProjects\pythonProject1\pdp_dt_sex.png']
for image_path in image_paths:
    prediction_cnn = predict_image(cnn_model, image_path)

    # Display the input image
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.show()

    # Display the prediction result
    print(f"Prediction Result (CNN) for {image_path}: {prediction_cnn}")

    # Save the prediction result to a file
    result_file_path = f'prediction_result_{os.path.basename(image_path)}.txt'
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Prediction Result (CNN) for {image_path}: {prediction_cnn}")

    print(f"Saved prediction result at {result_file_path}")
