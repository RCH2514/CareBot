import statistics #Provides a function to calculate the mode.
from scipy import stats #Provides another method to calculate the mode.
# PANDAS AND NUMPY For data manipulation and numerical computations.
import pandas as pd
import numpy as np
#sklearn: For machine learning models and preprocessing.
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder #LabelEncoder: Encodes categorical labels as integers.
from sklearn.model_selection import train_test_split



# Training the models on the whole dataset
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

# Loading and preparing the training data
training_data = pd.read_csv("data.csv").dropna(axis=1) # training_data : Loaded from Training.csv and cleaned by dropping columns with missing values (dropna(axis=1)).
print(f"Training data shape: {training_data.shape}")
encoder = LabelEncoder()
# Count and display the number of duplicated rows in the training data
duplicated_rows = training_data.duplicated().sum()
print(f"Number of duplicated rows in training data: {duplicated_rows}")
# Remove duplicate rows based on all columns
training_data_cleaned = training_data.drop_duplicates()
print(f"Cleaned training data shape: {training_data_cleaned.shape}")
target_column = 'diseases'
disease_counts = training_data_cleaned[target_column].value_counts()


# Create an empty DataFrame to store the downsampled data
downsampled_data = pd.DataFrame()

# Iterate through each disease and apply the rules
for disease, count in disease_counts.items():
    disease_data = training_data_cleaned[training_data_cleaned[target_column] == disease]
    
    if count < 50:
        downsampled_disease_data = disease_data  # Keep all rows if count is less than 200
    elif 50 <= count <= 100:
        downsampled_disease_data = disease_data.sample(n=50, random_state=42, replace=False)  # Take 200 rows randomly
    else:
        downsampled_disease_data = disease_data.sample(n=100, random_state=42, replace=False)  # Take 500 rows randomly
    
    # Append the downsampled data
    downsampled_data = pd.concat([downsampled_data, downsampled_disease_data])

# Check the new shape of the dataset
print(f"Downsampled data shape: {downsampled_data.shape}")

downsampled_data.to_csv('downsampled_data.csv', index=False)

