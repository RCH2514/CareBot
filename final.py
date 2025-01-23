import statistics
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # Importing scipy for mode calculation

# Training the models on the whole dataset
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

# Loading and preparing the training data
training_data = pd.read_csv("Training.csv").dropna(axis=1)
encoder = LabelEncoder()

X = training_data.iloc[:, :-1]  # Features
y = encoder.fit_transform(training_data.iloc[:, -1])  # Target variable

# Fit the models
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading and preparing the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])  # Use the same encoder for consistency

# Making predictions by taking the mode of predictions from all classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

# Calculate the final predictions using mode
final_preds = np.array([stats.mode([i, j, k], keepdims=True).mode[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)])
# Calculate accuracy
accuracy = accuracy_score(test_Y, final_preds) * 100
print(f"Accuracy on Test dataset by the combined model: {accuracy:.2f}%")

# Get the list of symptoms from the columns of the dataset
symptoms = X.columns.values

# Create a symptom index dictionary to encode input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

# Store symptom index and prediction classes in a dictionary
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_,
}

# Define the prediction function
def predictDisease(symptoms):
    """
    Predict disease based on input symptoms.

    Args:
        symptoms (str): A string containing symptoms separated by commas.

    Returns:
        dict: A dictionary containing predictions from each model and the final prediction.
    """
    symptoms = symptoms.split(",")
    
    # Create input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            raise ValueError(f"Symptom '{symptom}' not recognized. Please check the input.")

    # Reshape the input data into a format suitable for model predictions
    input_data = np.array(input_data).reshape(1, -1)
    
    # Generate individual model predictions
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # Make the final prediction by taking the mode of all predictions
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
    }
    return predictions

# Testing the function
try:
    result = predictDisease("Skin Rash,Joint Pain,Vclearomiting")
    print(result)
except ValueError as e:
    print(e)