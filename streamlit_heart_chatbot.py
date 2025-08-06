import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# File to store prediction history
CSV_FILE = "prediction_history.csv"

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🫀 Heart Disease Risk Chatbot")
st.markdown("Answer the questions below to assess your heart disease risk.")

# Input fields
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

# Save prediction function
def save_prediction(data, prediction):
    data["prediction (%)"] = round(prediction, 2)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)

# Generate PDF report function
def generate_pdf(input_data, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    textobject = c.beginText(40, 750)
    textobject.setFont("Helvetica", 12)

    textobject.textLine("Heart Disease Risk Assessment Report")
    textobject.textLine("--------------------------------------")
    for key, value in input_data.items():
        textobject.textLine(f"{key}: {value}")
    textobject.textLine(f"\nPredicted Risk: {round(prediction, 2)}%")
    textobject.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Main button
if st.button("Check Risk"):
    try:
        input_list = [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        input_array = scaler.transform([input_list])
        prediction = model.predict_proba(input_array)[0][1] * 100

        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        save_prediction(input_data, prediction)

        st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
        if prediction > 70:
            st.warning("⚠️ This is a high risk. Please consult a medical professional.")
        elif prediction > 40:
            st.info("🔍 This is a moderate risk. A check-up is recommended.")
        else:
            st.info("✅ This appears to be a low risk. Keep up the healthy lifestyle!")

        # Pie Chart
        labels = ['At Risk', 'No Risk']
        sizes = [round(prediction, 2), 100 - round(prediction, 2)]
        colors = ['red', 'green']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.markdown("### 🧩 Risk Distribution")
        st.pyplot(fig)

        # PDF Download
        pdf = generate_pdf(input_data, prediction)
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name="heart_risk_report.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")

# CSV download
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "rb") as f:
        st.download_button("📥 Download All Predictions", f, file_name=CSV_FILE)
