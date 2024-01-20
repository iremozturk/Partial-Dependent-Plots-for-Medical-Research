# data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_dataset1(dataset_path):
    # Load your dataset (replace 'your_dataset.csv' with the actual file path)
    # Load your dataset (replace 'your_dataset.csv' with the actual file path)
    df = pd.read_csv(dataset_path)

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

    return X_train, X_test, y_train, y_test, X

# Add similar functions for other datasets if needed
