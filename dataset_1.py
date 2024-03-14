import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_dataset1(file_path):

    cleaned_file_path = "cleaned_dataset_1.csv"

    if not os.path.exists(cleaned_file_path):
        # Perform data cleaning steps if the cleaned dataset doesn't exist
        df = pd.read_csv(
            "C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\outputsu.csv")
        # Perform data cleaning steps...
        # Save the cleaned DataFrame to CSV
        df.to_csv(file_path, index=False)
    else:
        # Now load the cleaned dataset
        df = pd.read_csv(file_path)



    df['binaryClass'] = df['binaryClass'].map({'N': 0, 'P': 1})

    # Convert non-numeric values to NaN
    df.replace('?', np.nan, inplace=True)

    # Convert all columns to float
    #df = df.astype(float)

    # Separate features and target
    X = df.drop(columns=['binaryClass'])
    y = df['binaryClass']
    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(X)


    column_names = df.columns.tolist()

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test, X, column_names
