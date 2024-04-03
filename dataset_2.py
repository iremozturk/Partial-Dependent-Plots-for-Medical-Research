import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os

def clean_dataset2(file_path):
    cleaned_file_path = "cleaned_dataset_2.csv"
    if not os.path.exists(cleaned_file_path):
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Patient Id', 'index'])
        # Replace non-numeric values with numeric equivalents
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            categories = df[col].unique()
            mapping = {category: i + 1 for i, category in enumerate(categories)}
            df[col] = df[col].map(mapping)

        # Convert all columns to float
        df = df.astype(float)
        df.to_csv(cleaned_file_path, index=False)
    else:
        # Now load the cleaned dataset
        df = pd.read_csv(cleaned_file_path)

    # Separate features and target
    X = df.drop(columns=['Level'])
    y = df['Level']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Convert X to DataFrame
    X_df = pd.DataFrame(X, columns=df.columns[:-1])

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    column_names = X_df.columns.tolist()

    return X_train, X_test, y_train, y_test, X_df, column_names
