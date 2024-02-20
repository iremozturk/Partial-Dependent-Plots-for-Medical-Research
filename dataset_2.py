import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_dataset2(file_path, output_csv_path):
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

    # Separate features and target
    X = df.drop(columns=['Level'])
    y = df['Level']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the cleaned dataset to a CSV file
    df.to_csv(output_csv_path, index=False)
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test, X
