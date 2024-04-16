import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_dataset18(file_path):

    cleaned_file_path = "cleaned_dataset_18.csv"

    if not os.path.exists(cleaned_file_path):
        # Perform data cleaning steps if the cleaned dataset doesn't exist
        df = pd.read_csv(r"C:\Users\iremo\PycharmProjects\pythonProject1\diabetes-18.csv")


        # Replace non-numeric values with numeric equivalents
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            categories = df[col].unique()
            mapping = {category: i + 1 for i, category in enumerate(categories)}
            df[col] = df[col].map(mapping)
        # Save the cleaned DataFrame to CSV
        df.to_csv(file_path, index=False)
    else:
        # Now load the cleaned dataset
        df = pd.read_csv(file_path)

    # Convert non-numeric values to NaN
    df.replace('?', np.nan, inplace=True)

    # Separate features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Get column names before converting to numpy array
    column_names = X.columns.tolist()
    print("Shape of X before imputation:", X.shape)
    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(X)
    print("Shape of X after imputation:", X.shape)

    # Check if the number of columns has changed
    print("Length of column names:", len(column_names))

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X, column_names
