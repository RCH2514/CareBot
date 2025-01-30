import statistics
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Charger les données
data = pd.read_csv("downsampled_data.csv").dropna(axis=1)

# Encoder les labels
encoder = LabelEncoder()
x = data.iloc[:, 1:]  # Features
y = encoder.fit_transform(data.iloc[:, 0])  # Target variable

# Diviser les données en batchs
def split_into_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

batch_size = 1000  # Vous pouvez ajuster la taille du batch selon vos besoins
x_batches = split_into_batches(x, batch_size)
y_batches = split_into_batches(y, batch_size)

# Initialiser les modèles
svm_models = []
nb_models = []
rf_models = []

# Entraîner les modèles sur chaque batch
for x_batch, y_batch in zip(x_batches, y_batches):
    X_train, X_test, y_train, y_test = train_test_split(x_batch, y_batch, test_size=0.2, random_state=42)
    
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_models.append(svm_model)
    
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_models.append(nb_model)
    
    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)
    rf_models.append(rf_model)

# Fonction pour faire des prédictions sur un batch
def predict_on_batch(models, X_test):
    svm_preds = models[0].predict(X_test)
    nb_preds = models[1].predict(X_test)
    rf_preds = models[2].predict(X_test)
    final_preds = np.array([stats.mode([i, j, k], keepdims=True).mode[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)])
    return final_preds

# Faire des prédictions sur chaque batch et combiner les résultats
all_preds = []
all_y_test = []

for x_batch, y_batch, svm_model, nb_model, rf_model in zip(x_batches, y_batches, svm_models, nb_models, rf_models):
    X_train, X_test, y_train, y_test = train_test_split(x_batch, y_batch, test_size=0.2, random_state=42)
    final_preds = predict_on_batch([svm_model, nb_model, rf_model], X_test)
    all_preds.extend(final_preds)
    all_y_test.extend(y_test)

# Calculer l'accuracy globale
accuracy = accuracy_score(all_y_test, all_preds) * 100
print(f"Accuracy on Test dataset by the combined model: {accuracy:.2f}%")

#initialisation de data_dic
#Get the list of symptoms from the columns of the dataset
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

# Fonction pour prédire la maladie
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            raise ValueError(f"Symptom '{symptom}' not recognized. Please check the input.")

    input_data = np.array(input_data).reshape(1, -1)
    
    rf_prediction = data_dict["predictions_classes"][rf_models[0].predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_models[0].predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_models[0].predict(input_data)[0]]
    
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction,
    }
    return predictions

# Tester la fonction
try:
    result = predictDisease("depression,sharp chest pain")
    print(result)
except ValueError as e:
    print(e)