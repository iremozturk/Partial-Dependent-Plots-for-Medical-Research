import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_dataset1(file_path):
    cleaned_file_path = "cleaned_dataset_1.csv"

    if not os.path.exists(cleaned_file_path):
        df = pd.read_csv("C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv")

        # Handle non-numeric data and missing values
        df.replace('?', np.nan, inplace=True)

        # Convert 'binaryClass' from 'N'/'P' to 0/1
        if 'binaryClass' in df.columns:
            df['binaryClass'] = df['binaryClass'].map({'N': 0, 'P': 1})

        # Convert other categorical columns using LabelEncoder or mapping
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            if col != 'binaryClass':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Convert all columns to float
        df = df.astype(float)
        df.to_csv(cleaned_file_path, index=False)
    else:
        df = pd.read_csv(cleaned_file_path)

    print(df.columns)

    y = df['binaryClass']
    X = df.drop(columns=['binaryClass'])
    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(X)

    column_names = df.columns.tolist()

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X, column_names
