import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import csv

def clean_dataset3(file_path):
    cleaned_file_path = "C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\cleaned_datasettt_3.csv"

    if not os.path.exists(cleaned_file_path):
        # Perform data cleaning steps if the cleaned dataset doesn't exist
        df = pd.read_csv("C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\Data_matrix_30_patients_additional.csv")

        # Replace blanks with commas
        df.replace(' ', ',', inplace=True)

        # Save the cleaned DataFrame to CSV
        df.to_csv(file_path, index=False)
    else:
        # Now load the cleaned dataset
        df = pd.read_csv(file_path)

    # Print the top 10 lines of the DataFrame
    print(df.head(10))

    # Convert non-numeric values to NaN
    df.replace('?', np.nan, inplace=True)

    # Drop columns that contain strings
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    df.drop(columns=non_numeric_cols, inplace=True)

    # Calculate the percentage of NA values in each column
    na_percentage = (df.isna().sum() / len(df)) * 100

    # Specify the threshold for dropping columns (70% in this case)
    threshold = 70

    # Get the names of columns exceeding the threshold
    columns_to_drop = na_percentage[na_percentage > threshold].index

    # Drop columns exceeding the threshold
    df = df.drop(columns=columns_to_drop)

    # Separate features and target
    X = df.drop(columns=['GVHD_factor']) if 'GVHD_factor' in df.columns else df
    y = df['GVHD_factor'] if 'GVHD_factor' in df.columns else None
    if y is not None:
        imputer = SimpleImputer(strategy='most_frequent')
        X = imputer.fit_transform(X)

        column_names = df.columns.tolist()

        # Split your data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, X, column_names

    else:
        return None, None, None, None, None, None


