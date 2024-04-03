import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_dataset4(file_path):

    cleaned_file_path = "cleaned_dataset_4.csv"

    if not os.path.exists(cleaned_file_path):
        # Perform data cleaning steps if the cleaned dataset doesn't exist
        df = pd.read_csv(
            "C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\heart_disease_uci.csv")
        # Drop the "id" column
        df.drop(columns=['id'], inplace=True)

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
    X = df.drop(columns=['num'])
    y = df['num']

    # Get column names before converting to numpy array
    column_names = X.columns.tolist()

    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(X)

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X, column_names
