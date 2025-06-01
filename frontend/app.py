# streamlit_app.py

import streamlit as st
import requests

st.title("Breast Cancer Prediction App")

features = []
for i in range(30): 
    features.append(st.number_input(f"Feature {i+1}", step=0.1))

if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict", json={"features": features})
    st.write("Prediction:", response.json()["prediction"])
