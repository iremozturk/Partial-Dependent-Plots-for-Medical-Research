import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import csv

# Open the text file for reading
with open('Data_matrix_30_patients_additional.txt', 'r') as txtfile:
    # Read each line from the text file
    lines = txtfile.readlines()

# Open a CSV file for writing
with open('Data_matrix_30_patients_additional.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Iterate over each line in the text file
    for line in lines:
        # Parse the line to extract fields (example assumes comma-separated values)
        fields = line.strip().split(',')  # Adjust this line based on your text file's structure

        # Write the fields to the CSV file
        writer.writerow(fields)

def clean_dataset3(file_path):

    cleaned_file_path = "cleaned_dataset_3.csv"

    if not os.path.exists(cleaned_file_path):
        # Perform data cleaning steps if the cleaned dataset doesn't exist
        df = pd.read_csv(
            "C:\\Users\\iremo\\PycharmProjects\\pythonProject1\\Data_matrix_30_patients_additional.csv")
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
