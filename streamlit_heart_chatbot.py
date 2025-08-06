import streamlit as st
from streamlit_chat import message
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
CSV_FILE = "prediction_history.csv"

# Generate PDF report
def generate_pdf(input_data, prediction):
    field_names = {
        "age": "Age",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol Level (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
        "restecg": "Resting ECG Result (0–2)",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise-Induced Angina (0=No, 1=Yes)",
        "oldpeak": "ST Depression by Exercise",
        "slope": "Slope of Peak Exercise ST",
        "ca": "Major Vessels Colored (0–4)",
        "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
    }

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)

    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for key, value in input_data.items():
        label = field_names.get(key, key)
        text.textLine(f"{label}: {value}")
    text.textLine(f"\nPredicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Save prediction
def save_prediction(data, prediction):
    data["prediction (%)"] = round(prediction, 2)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)

# Define questions
questions = [
    {"key": "age", "text": "What is your age?", "type": int},
    {"key": "sex", "text": "What is your biological sex? (0 = Female, 1 = Male)", "type": int},
    {"key": "cp", "text": "What is your chest pain type? (0–3)", "type": int},
    {"key": "trestbps", "text": "Resting blood pressure (mm Hg)?", "type": int},
    {"key": "chol", "text": "Cholesterol level (mg/dl)?", "type": int},
    {"key": "fbs", "text": "Fasting blood sugar > 120 mg/dl? (0 or 1)", "type": int},
    {"key": "restecg", "text": "Resting ECG result? (0–2)", "type": int},
    {"key": "thalach", "text": "Maximum heart rate achieved?", "type": int},
    {"key": "exang", "text": "Exercise-induced angina? (0 or 1)", "type": int},
    {"key": "oldpeak", "text": "ST depression induced by exercise?", "type": float},
    {"key": "slope", "text": "Slope of peak ST segment? (0–2)", "type": int},
    {"key": "ca", "text": "Number of major vessels coloured? (0–4)", "type": int},
    {"key": "thal", "text": "Thalassemia? (0=Normal, 1=Fixed, 2=Reversible)", "type": int}
]

# Initial state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# Title
st.title("💬 Heart Disease Risk Chatbot (Chat Mode)")

# Chat history with unique keys
for i, msg in enumerate(st.session_state.chat_history):
    message(msg["text"], is_user=msg["is_user"], key=f"{'user' if msg['is_user'] else 'bot'}-{i}")

# Question flow
if st.session_state.current_q < len(questions):
    q = questions[st.session_state.current_q]
    user_input = st.chat_input(q["text"])

    if user_input:
        try:
            typed_input = q["type"](user_input)
            st.session_state.answers[q["key"]] = typed_input
            st.session_state.chat_history.append({"text": q["text"], "is_user": False})
            st.session_state.chat_history.append({"text": user_input, "is_user": True})
            st.session_state.current_q += 1
            st.rerun()
        except ValueError:
            st.warning("Please enter a valid value.")

# All answers collected
else:
    inputs = [st.session_state.answers[q["key"]] for q in questions]
    input_array = scaler.transform([inputs])
    prediction = model.predict_proba(input_array)[0][1] * 100

    save_prediction(st.session_state.answers, prediction)

    st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
    if prediction > 70:
        st.warning("⚠️ High risk! Please consult a doctor.")
    elif prediction > 40:
        st.info("🔍 Moderate risk. A check-up is recommended.")
    else:
        st.info("✅ Low risk. Great job keeping healthy!")

    # Pie chart
    st.markdown("### 📊 Risk Distribution")
    fig, ax = plt.subplots()
    ax.pie([prediction, 100 - prediction], labels=["At Risk", "No Risk"],
           colors=["red", "green"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # PDF
    pdf = generate_pdf(st.session_state.answers, prediction)
    st.download_button("📄 Download PDF Report", data=pdf,
                       file_name="heart_risk_report.pdf", mime="application/pdf")

    # CSV download
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("📥 Download All Predictions", f, file_name=CSV_FILE)
