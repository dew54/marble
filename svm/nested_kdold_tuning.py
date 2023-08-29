import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib





def display_class_percentages(csv_file):
    data = pd.read_csv(csv_file)
    class_counts = data['label'].value_counts()
    total_samples = data.shape[0]
    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{class_label}: {percentage:.2f}%")

def perform_nested_kfold_with_svm(csv_file, num_folds=5):
    data = pd.read_csv(csv_file)
    X = data.drop('label', axis=1).values
    y = data['label'].values

    kernels = ['poly', 'rbf', 'sigmoid']
    best_models = []
    best_scores = []

    skf_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for train_index, test_index in skf_outer.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model = None
        best_score = -1
        best_kernel = None

        for kernel in kernels:
            svc = SVC(kernel=kernel)
            param_grid = {
                'C': [0.1, 1, 4, 10],
                'gamma': ['scale', 'auto'] if kernel == 'rbf' else [1.0]
            }
            grid_search = GridSearchCV(svc, param_grid, cv=num_folds)
            
            skf_inner = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
            scores = cross_val_score(grid_search, X_train, y_train, cv=skf_inner)
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_model = grid_search.fit(X_train, y_train)
                best_kernel = kernel

        print(f"Best Kernel: {best_kernel}, Best Parameters: {best_model.best_params_}, Best Score: {best_score}")
        best_models.append(best_model)
        best_scores.append(best_score)

         
     # Save the best model for each kernel
    for idx, kernel in enumerate(kernels):
        joblib.dump(best_models[idx], f'best_model_{kernel}.pkl')
    
    return best_models

def evaluate_model(best_model, csv_file, test_size=0.2):
    data = pd.read_csv(csv_file)
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Train the best model on the full training set
    best_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print("Test Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))  # Set zero_division to 1

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_on_validation(best_model, validation_csv_file):
    validation_data = pd.read_csv(validation_csv_file)
    X_val = validation_data.drop('label', axis=1).values
    y_val = validation_data['label'].values

    # Make predictions on the validation set
    y_pred_val = best_model.predict(X_val)

    # Evaluate the model on the validation set
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print("\nValidation Set Evaluation:")
    print("Validation Accuracy:", val_accuracy)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=1))
    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_val))
if __name__ == "__main__":
    #csv_file_path = "csvdata/balanced_normalized_dataset.csv"
    csv_file_path = "filtered_data.csv"

    validation_csv_file = "csvdata/balanced_normalized_dataset_val.csv"

    # Task 1: Display the amount in percentage of the data for each class
    display_class_percentages(csv_file_path)

    # Task 2: Perform nested k-fold cross-validation with SVM using different kernels and tune hyperparameters
    best_models = perform_nested_kfold_with_svm(csv_file_path)
    
    # Load and evaluate the best models for each kernel
    for kernel in ['poly', 'rbf', 'sigmoid']:
        saved_best_model = joblib.load(f'best_model_{kernel}.pkl')
        print(f"Evaluating best model for kernel: {kernel}")
        evaluate_model(saved_best_model, csv_file_path)
        evaluate_on_validation(saved_best_model, validation_csv_file)    
    
    