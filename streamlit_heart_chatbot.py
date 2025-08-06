import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# File to store prediction history
CSV_FILE = "prediction_history.csv"

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🫀 Heart Disease Risk Chatbot")
st.markdown("Answer the questions below to assess your heart disease risk.")

# Create input fields for all 13 parameters
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol level", min_value=100, max_value=600, value=240)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", [0, 1])
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-induced angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–4) colored by fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])

# Define function to save prediction
def save_prediction(data, prediction):
    data["prediction (%)"] = round(prediction, 2)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)

# Prediction logic
if st.button("Check Risk"):
    try:
        # List for model prediction
        input_list = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        input_array = scaler.transform([input_list])
        prediction = model.predict_proba(input_array)[0][1] * 100

        # Dictionary for logging
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        # Save prediction to CSV
        save_prediction(input_data, prediction)

        # Show result
        st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
        if prediction > 70:
            st.warning("⚠️ This is a high risk. Please consult a medical professional.")
        elif prediction > 40:
            st.info("🔍 This is a moderate risk. A check-up is recommended.")
        else:
            st.info("✅ This appears to be a low risk. Keep up the healthy lifestyle!")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "rb") as f:
        st.download_button("📥 Download All Predictions", f, file_name=CSV_FILE)

