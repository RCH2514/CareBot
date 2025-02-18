
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
import statistics
data = pd.read_csv("downsampled_data.csv").dropna(axis=1)

encoder = LabelEncoder()
x = data.iloc[:, 1:] 
y = encoder.fit_transform(data.iloc[:, 0]) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_models = []
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    

rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f"Random Forest Accuracy: {accuracy_rf:.2f}%")

symptoms = x.columns.values
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom_index[value] = index
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_,
}

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
    
    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]

    return  rf_prediction

'''try:
    result = predictDisease("depression,sharp chest pain, skin growth, hot flashes")
    print(result)
except ValueError as e:
    print(e)'''

# Interface Streamlit
st.title('Chatbot de Prédiction de Maladies')
st.write("Entrez vos symptômes séparés par des virgules pour obtenir des prédictions.")
# Créer un champ de texte pour entrer les symptômes
symptoms_input = st.text_input('Symptômes (ex : fever, cough, headache)')
if symptoms_input:
    try:
        # Obtenir la prédiction
        result = predictDisease(symptoms_input)
        
        # Afficher les prédictions des modèles
        st.subheader("Prédiction du Modèle :")
        st.write(f"{result}")  # result est déjà la maladie prédite

    except ValueError as e:
        st.error(str(e))

        