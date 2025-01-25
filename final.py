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
data = pd.read_csv("downsampled_data.csv").dropna(axis=1) # training_data : Loaded from Training.csv and cleaned by dropping columns with missing values (dropna(axis=1)).
encoder = LabelEncoder()
#A feature (input data, such as symptoms in this case).
#A target variable (output data, such as disease in this case).
x = data.iloc[:, 1:]  # Features: All columns except the first one.  iloc: A Pandas function used for positional indexing. It selects rows and columns based on their numerical indices.
y = encoder.fit_transform(data.iloc[:, 0])  # Target variable : All columns except the first one.
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# Fit the models
final_svm_model.fit(X_train, y_train)
final_nb_model.fit(X_train, y_train)
final_rf_model.fit(X_train, y_train)
# Making predictions by taking the mode of predictions from all classifiers
svm_preds = final_svm_model.predict(X_test)
nb_preds = final_nb_model.predict(X_test)
rf_preds = final_rf_model.predict(X_test)

# Calculate the final predictions using mode
final_preds = np.array([stats.mode([i, j, k], keepdims=True).mode[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)])
# Calculate accuracy
accuracy = accuracy_score(y_test, final_preds) * 100
print(f"Accuracy on Test dataset by the combined model: {accuracy:.2f}%")

# Get the list of symptoms from the columns of the dataset
symptoms = x.columns.values

# Create a symptom index dictionary to encode input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom_index[value] = index

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
    result = predictDisease("depression,sharp chest pain")
    print(result)
except ValueError as e:
    print(e)