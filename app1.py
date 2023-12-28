# Import necessary libraries
import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Create a function to make predictions
def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit App
def main():
    st.title("Application du prédiction de la maladie de diabète")

    # Create input fields for user to enter data
    pregnancies = st.number_input("Grossesses", value=0)
    glucose = st.number_input("Glucose", value=0)
    blood_pressure = st.number_input(" PressionArtérielle", value=0)
    skin_thickness = st.number_input("ÉpaisseurPeau", value=0)
    insulin = st.number_input("Insuline", value=0)
    bmi = st.number_input("IMC(Indice de Masse Corporelle)", value=0.0)
    diabetes_pedigree_function = st.number_input("FonctionPédigréeDiabète", value=0.0)
    age = st.number_input("Age", value=0)

    # Display the user inputs
    user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    st.write("caracteristiques saisies par l'utilisateur:")
    st.write(user_input)

    # Make a prediction
    result = predict_diabetes(user_input)

    # Display the result
    st.subheader("Prédiction:")
    if result == 1:
        st.write("La personne est diabétique.")
    else:
        st.write("La personne est non diabétique.")

if __name__ == "__main__":
    main()
