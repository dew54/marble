import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def append_csv_files(folder_path):
    dfs = []
    first_file = True  # To track if this is the first file
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, header=None)  # Read without headers
            
            if first_file:
                appended_data = df
                first_file = False
            else:
                appended_data = appended_data.append(df, ignore_index=True)  # Append the data
    
    if first_file:
        raise ValueError("No CSV files found in the folder.")
    
    return appended_data

def preprocess_and_balance_data(data):
    # Adding the header to the concatenated dataframe
    header = ['label'] + [f'feature{i}' for i in range(1, len(data.columns))]
    data.columns = header
    
    # Removing duplicate rows
    initial_rows = len(data)
    data.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - len(data)
    
    # Balancing class distribution
    class_counts = data['label'].value_counts()
    min_class_count = class_counts.min()
    balanced_data = pd.DataFrame(columns=data.columns)
    
    for label, group in data.groupby('label'):
        sampled_group = group.sample(min_class_count, random_state=42)
        balanced_data = pd.concat([balanced_data, sampled_group])
    
    # Calculating class percentages
    total_samples = len(balanced_data)
    class_percentages = balanced_data['label'].value_counts() / total_samples * 100
    
    return balanced_data, removed_duplicates, class_percentages

def remove_duplicate_features(data):
    # Identify and remove rows with duplicate features
    initial_rows = len(data)
    data.drop_duplicates(subset=data.columns[1:], inplace=True)
    removed_duplicates = initial_rows - len(data)
    return data, removed_duplicates

def normalize_data(data):
    # Extract the features for normalization
    features = data.iloc[:, 1:]
    
    # Initialize the standard scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the features and transform them
    normalized_features = scaler.fit_transform(features)
    
    # Replace the original feature values with normalized values
    data.iloc[:, 1:] = normalized_features
    
    return data

if __name__ == "__main__":
    folder_path = "csvdata/val"  # Replace with your folder path containing CSV files
    
    appended_data = append_csv_files(folder_path)
    
    
    preprocessed_data, removed_duplicates, class_percentages = preprocess_and_balance_data(appended_data)

    
    normalized_data = normalize_data(preprocessed_data)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print("Balanced class percentages:")
    print(class_percentages)
    
   
    output_filename = "csvdata/balanced_normalized_dataset_val.csv" # Change with the name that you want
    normalized_data.to_csv(output_filename, index=False)
    print(f"Balanced and normalized dataset saved as '{output_filename}'.")