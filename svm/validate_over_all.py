import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  

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

validation_csv_file = "csvdata/balanced_normalized_dataset_val.csv"

# Load the best model
saved_best_model = joblib.load('svm.pkl')
#  Evaluate the best model on the validation set
evaluate_on_validation(saved_best_model, validation_csv_file)
