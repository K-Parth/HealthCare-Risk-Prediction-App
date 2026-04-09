import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("xgb_model.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction")

st.title("❤️ Heart Disease Prediction System")

st.markdown("Enter patient details below:")

# -----------------------------
# Input Fields
# -----------------------------

age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression")
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0–3)", [0, 1, 2, 3])

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope,
                            ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")