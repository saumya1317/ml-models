# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:06:43 2024

@author: Saumya
"""

import numpy as np
import pickle
import streamlit as st

# Load the model
loaded_model = pickle.load(open(r'C:\Users\Saumya\Desktop\mlmodel\heart_disease_model.sav', 'rb'))

def heart(input):
    input_data = np.asarray(input)
    reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(reshaped)
    if prediction == 1:
        return 'Defective heart'
    else:
        return 'Healthy heart'

def main():
    # Collect input values
    age = st.text_input('Your age', key='age')
    sex = st.text_input('Your gender', key='sex')
    cp = st.text_input('Chest pain level (cp)', key='cp')
    trestbps = st.text_input('Resting blood pressure (trestbps)', key='trestbps')
    chol = st.text_input('Cholesterol level (chol)', key='chol')
    fbs = st.text_input('Fasting blood sugar (fbs)', key='fbs')
    restecg = st.text_input('Resting electrocardiographic results (restecg)', key='restecg')
    thalach = st.text_input('Maximum heart rate achieved (thalach)', key='thalach')
    exang = st.text_input('Exercise induced angina (exang)', key='exang')
    oldpeak = st.text_input('ST depression induced by exercise (oldpeak)', key='oldpeak')
    slope = st.text_input('Slope of peak exercise ST segment (slope)', key='slope')
    ca = st.text_input('Number of major vessels (ca)', key='ca')
    thal = st.text_input('Thalassemia (thal)', key='thal')
    
    diagnosis = ''
    
    # Convert inputs to numeric values before passing to the model
    try:
        # Ensure that each input is converted to a float
        inputs = [
            float(age), float(sex), float(cp), float(trestbps),
            float(chol), float(fbs), float(restecg), float(thalach),
            float(exang), float(oldpeak), float(slope), float(ca), float(thal)
        ]
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
        return
    
    # Perform prediction only if the button is clicked
    if st.button('Get Heart Test Result'):
        diagnosis = heart(inputs)
    
    # Display the result
    if diagnosis:
        st.success(diagnosis)

if __name__ == '__main__':
    main()
