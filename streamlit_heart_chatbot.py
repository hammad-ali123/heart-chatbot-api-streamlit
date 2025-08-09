import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Chatbot settings
ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

LABELS = {
    "age": "Age",
    "sex": "Sex (0=Female, 1=Male)",
    "cp": "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
    "restecg": "Resting ECG (0–2)",
    "thalach": "Max Heart Rate Achieved (bpm)",
    "exang": "Exercise-Induced Angina (0=No, 1=Yes)",
    "oldpeak": "ST Depression by Exercise",
    "slope": "Slope of Peak Exercise ST (0–2)",
    "ca": "Major Vessels Coloured (0–4)",
    "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
}

SCHEMA = {
    "age":       {"type": int, "min": 1, "max": 120},
    "sex":       {"type": int, "allowed": {0, 1}},
    "cp":        {"type": int, "allowed": {0, 1, 2, 3}},
    "trestbps":  {"type": int, "min": 80, "max": 250},
    "chol":      {"type": int, "min": 100, "max": 700},
    "fbs":       {"type": int, "allowed": {0, 1}},
    "restecg":   {"type": int, "allowed": {0, 1, 2}},
    "thalach":   {"type": int, "min": 60, "max": 250},
    "exang":     {"type": int, "allowed": {0, 1}},
    "oldpeak":   {"type": float, "min": 0.0, "max": 6.0},
    "slope":     {"type": int, "allowed": {0, 1, 2}},
    "ca":        {"type": int, "allowed": {0, 1, 2, 3, 4}},
    "thal":      {"type": int, "allowed": {0, 1, 2}}
}

# Functions
def parse_and_validate(key, text):
    spec = SCHEMA[key]
    try:
        val = int(float(text)) if spec["type"] == int else float(text)
    except ValueError:
        return None, "❌ Please enter a valid number."

    if "allowed" in spec and val not in spec["allowed"]:
        return None, f"❌ Value must be one of: {', '.join(map(str, spec['allowed']))}."
    if "min" in spec and val < spec["min"]:
        return None, f"❌ Value too low. Min is {spec['min']}."
    if "max" in spec and val > spec["max"]:
        return None, f"❌ Value too high. Max is {spec['max']}."
    return val, None

def generate_pdf(input_data, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for k in ORDER:
        text.textLine(f"{LABELS[k]}: {input_data[k]}")
    text.textLine("")
    text.textLine(f"Predicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = {}
if "await_confirm" not in st.session_state:
    st.session_state.await_confirm = False

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Determine prompt
if not st.session_state.await_confirm:
    current_key = ORDER[st.session_state.step]
    bot_prompt = f"{LABELS[current_key]}?"
else:
    bot_prompt = "Type YES to confirm or RESTART to start over."

with st.chat_message("assistant"):
    st.markdown(bot_prompt)

# Get user input
if user_input := st.chat_input("Your answer..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not st.session_state.await_confirm:
        # Validate answer
        val, err = parse_and_validate(current_key, user_input)
        if err:
            reply = err + f" Please re-enter {LABELS[current_key]}."
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            st.session_state.inputs[current_key] = val
            st.session_state.step += 1
            if st.session_state.step == len(ORDER):
                # Show summary
                summary = "✅ Please confirm your details:\n"
                for k in ORDER:
                    summary += f"- **{LABELS[k]}:** {st.session_state.inputs[k]}\n"
                summary += "\nType YES to confirm or RESTART to start over."
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.session_state.await_confirm = True
            else:
                next_q = f"{LABELS[ORDER[st.session_state.step]]}?"
                st.session_state.messages.append({"role": "assistant", "content": next_q})

    else:
        if user_input.strip().lower() == "restart":
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.await_confirm = False
            st.session_state.messages.append({"role": "assistant", "content": "🔄 Restarted. Let's begin again."})
        elif user_input.strip().lower() == "yes":
            # Predict
            X = scaler.transform([[st.session_state.inputs[k] for k in ORDER]])
            pred_pct = model.predict_proba(X)[0][1] * 100.0
            result_msg = f"🧠 Your predicted heart disease risk is **{pred_pct:.2f}%**."
            if pred_pct > 70:
                result_msg += "\n⚠️ High risk. Please consult a medical professional."
            elif pred_pct > 40:
                result_msg += "\n🔍 Moderate risk. A check-up is recommended."
            else:
                result_msg += "\n✅ Low risk. Keep up the healthy lifestyle."
            st.session_state.messages.append({"role": "assistant", "content": result_msg})

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie([pred_pct, 100 - pred_pct], labels=['At Risk', 'No Risk'],
                   colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # PDF download
            pdf = generate_pdf(st.session_state.inputs, pred_pct)
            st.download_button("📄 Download PDF Report", data=pdf,
                               file_name="heart_risk_report.pdf", mime="application/pdf")

            # Reset
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.await_confirm = False
        else:
            st.session_state.messages.append({"role": "assistant", "content": "❌ Please type YES or RESTART."})
