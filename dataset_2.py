import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def clean_dataset2(file_path, output_csv_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns (if any)
    df = df.drop(['index', 'Patient Id'], axis=1)

    # Print unique values in the 'Level' column before mapping
    print("Unique values in 'Level' column before mapping:", df['Level'].unique())

    # Convert categorical columns to numerical using Label Encoding
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    # Map 'Level' to numerical values
    level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Level'] = df['Level'].map(level_mapping)

    # Print unique values in the 'Level' column after mapping
    print("Unique values in 'Level' column after mapping:", df['Level'].unique())

    # Split the data into features and target
    X = df.drop('Level', axis=1)
    y = df['Level']

    # Handle missing values in features using imputation
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Drop rows with missing target values
    df = pd.concat([X, y], axis=1)  # Concatenate X and y before dropping rows
    df = df.dropna(subset=['Level'])

    # Check if there are remaining samples
    if df.empty:
        raise ValueError("No samples remaining after dropping rows with missing target values.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the cleaned dataset to a CSV file
    df.to_csv(output_csv_path, index=False)

    return X_train, X_test, y_train, y_test, X

# Example usage:
output_csv_path = 'cleaned_dataset.csv'
file_path = r'C:\Users\iremo\PycharmProjects\pythonProject1\cancer patient data sets.csv'
X_train, X_test, y_train, y_test, X = clean_dataset2(file_path, output_csv_path)
