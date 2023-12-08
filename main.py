import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DecisionTree.DecisionTree import train_decision_tree
from sklearn.impute import SimpleImputer
from pdpbox import pdp
from RandomForest.RandomForest import train_random_forest
from itertools import combinations  # Import combinations from itertools

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

# Train a decision tree model
decision_tree, feature_importances = train_decision_tree(X_train, y_train)
random_forest, feature_importances_rf = train_random_forest(X_train, y_train)

# Identify the top 2 features based on importance
top_n = 2  # Change this to 2 to select the top 2 features
important_features = np.argsort(feature_importances)[::-1][:top_n]
important_features_rf = np.argsort(feature_importances_rf)[::-1][:top_n]

# Convert X_train back to a DataFrame and extract the names of the top 2 features
X_train_df_rf = pd.DataFrame(X_train, columns=df.columns[important_features_rf])
feature_names_rf = X_train_df_rf.columns

print("Top 2 Features By (Random Forest):", feature_names_rf)

#print("Top 2 Features:", feature_names)
#print("X_train shape before DataFrame conversion:", X_train.shape)
#print("X_train shape:", X_train_df.shape)
#print("X_train.columns:", X_train_df.columns)

# Create partial dependence plots (PDP) for top features
for feature in feature_names_rf:
    pdp_feature = pdp.pdp_isolate(model=random_forest, dataset=X_train_df_rf, model_features=feature_names_rf, feature=feature)

    # Check if there are NaN values in pdp_feature before attempting to plot
    if not np.isnan(pdp_feature.pdp).all():
        pdp.pdp_plot(pdp_feature, feature_name=feature)
        plt.title(f'Partial Dependence Plot for {feature} with binaryClass')
        plt.show()
    else:
        print(f"Skipping PDP plot for feature {feature} as no data is available.")
