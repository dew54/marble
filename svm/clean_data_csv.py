import csv
import ast
import os
import pandas as pd
from collections import defaultdict


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def write_csv(file_path, data, header=None):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        if header is not None:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(data)

def balance_dataset(data, target_column_index, classes_to_balance):
    class_counts = defaultdict(int)

    # Count the occurrences of each class
    for row in data:
        class_label = row[target_column_index]
        class_counts[class_label] += 1

    balanced_data = []

    # Determine the target count for balancing
    target_count = min(class_counts[cls] for cls in classes_to_balance)

    # Keep track of instances added for each class
    added_counts = defaultdict(int)

    # Balance the dataset by randomly removing instances from the other class
    for row in data:
        class_label = row[target_column_index]

        if class_label in classes_to_balance:
            if added_counts[class_label] < target_count:
                balanced_data.append(row)
                added_counts[class_label] += 1
        else:
            balanced_data.append(row)

    return balanced_data


def remove_duplicates_and_balance(data, target_column_index, target_class_to_reduce):
    unique_data = []
    seen_rows = set()
    class_counts = defaultdict(int)

    for row in data:
        row_tuple = tuple(row)
        class_label = row[target_column_index]

        if class_label == target_class_to_reduce:
            if class_counts[class_label] <= class_counts[max(class_counts, key=class_counts.get)]:
                unique_data.append(row)
                seen_rows.add(row_tuple)
                class_counts[class_label] += 1
        else:
            unique_data.append(row)
            seen_rows.add(row_tuple)
            class_counts[class_label] += 1

    return unique_data


def calculate_class_percentages(data, target_column_index):
    class_counts = defaultdict(int)
    total_samples = len(data)

    for row in data:
        class_label = row[target_column_index]
        class_counts[class_label] += 1

    class_percentages = {}
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        class_percentages[class_label] = percentage

    return class_percentages

input_file = 'dataset_orb_val.csv'
output_file = 'data_clean_balanced_orb_val.csv'
target_column_name = 'label'  # Update with the column name containing class labels
classes_to_balance = ['larvae', 'plankton']  # Update with the classes to balance

# Read the original CSV data
original_data = read_csv(input_file)
original_size = len(original_data)

# Generate automatic column names for the features
header = original_data[0]
column_names = [target_column_name] + [f'feature{i}' for i in range(1, len(header))]

# Determine the target column index
target_column_index = 0

# Remove duplicates and balance data for specific class
target_class_to_reduce = 'larvae'  # Update with the class label to reduce
unique_data = remove_duplicates_and_balance(original_data, target_column_index, target_class_to_reduce)
unique_size = len(unique_data)

# Balance the dataset for specified classes
balanced_data = balance_dataset(unique_data, target_column_index, classes_to_balance)
balanced_size = len(balanced_data)

# Calculate and print class percentages in the balanced dataset
class_percentages = calculate_class_percentages(balanced_data, target_column_index)
print("\nClass Percentages in the Balanced Dataset:")
for class_label, percentage in class_percentages.items():
    print(f"Class {class_label}: {percentage:.2f}%")

# Write the balanced data to a new CSV file with the header
write_csv(output_file, balanced_data, column_names)
# Print sizes before and after balancing
print(f"Original size: {original_size} rows")
print(f"Balanced size: {balanced_size} rows")