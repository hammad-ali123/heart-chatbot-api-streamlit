import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =========================
# Load LR model & scaler
# =========================
# Make sure these filenames match the ones you uploaded
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# Schema & labels
# =========================
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

# Realistic ranges/allowed values
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

ASSISTANT_AVATAR = "🩺"
USER_AVATAR = "🙂"

# =========================
# Utilities
# =========================
def add_assistant_once(text: str):
    """Append assistant message only if it's not identical to the last assistant message."""
    last_assistant = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break
    if last_assistant != text:
        st.session_state.messages.append({"role": "assistant", "content": text})

def parse_and_validate(key, raw):
    spec = SCHEMA[key]
    try:
        val = int(float(raw)) if spec["type"] == int else float(raw)
    except ValueError:
        return None, "❌ Please enter a valid number."

    if "allowed" in spec and val not in spec["allowed"]:
        return None, f"❌ Value must be one of: {', '.join(map(str, spec['allowed']))}."
    if "min" in spec and val < spec["min"]:
        return None, f"❌ Value too low. Min is {spec['min']}."
    if "max" in spec and val > spec["max"]:
        return None, f"❌ Value too high. Max is {spec['max']}."
    return val, None

def generate_pdf(input_data, prediction_pct):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    t = c.beginText(40, 750)
    t.setFont("Helvetica", 12)

    t.textLine("Heart Disease Risk Assessment Report")
    t.textLine("--------------------------------------")
    for k in ORDER:
        t.textLine(f"{LABELS[k]}: {input_data[k]}")
    t.textLine("")
    t.textLine(f"Predicted Risk: {round(prediction_pct, 2)}%")
    t.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(t)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def reset_all():
    st.session_state.phase = "ask"          # ask | confirm | done
    st.session_state.step = 0
    st.session_state.inputs = {}
    st.session_state.messages = []
    st.session_state.greeted = False

# =========================
# Session init
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "phase" not in st.session_state:
    reset_all()
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# =========================
# Header + Restart
# =========================
st.title("🫀 Heart Disease Risk Chatbot")
st.caption("I’ll ask a few questions and estimate your heart disease risk. "
           "Inputs are range‑checked to keep them realistic.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔄 Restart chat"):
        reset_all()
        st.rerun()

# =========================
# Greeting & first prompt (once)
# =========================
if not st.session_state.greeted:
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hello! I’m your virtual triage assistant."})
    st.session_state.messages.append({"role": "assistant", "content": "We’ll go step‑by‑step. Please answer with numbers exactly as requested."})
    first_q = f"**{LABELS[ORDER[0]]}?**"
    add_assistant_once(first_q)
    st.session_state.greeted = True

# =========================
# Render history
# =========================
for m in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if m["role"] == "assistant" else USER_AVATAR
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# =========================
# One chat input for the whole app
# =========================
user_text = st.chat_input("Your answer…")
if user_text:
    # Always echo the user message exactly once
    st.session_state.messages.append({"role": "user", "content": user_text})

    if st.session_state.phase == "ask":
        # Validate the current step
        key = ORDER[st.session_state.step]
        value, err = parse_and_validate(key, user_text)
        if err:
            add_assistant_once(err + f" Please re‑enter **{LABELS[key]}**.")
        else:
            # Store value and move forward
            st.session_state.inputs[key] = value
            st.session_state.step += 1

            if st.session_state.step == len(ORDER):
                # All inputs collected → show summary & ask to confirm
                summary = "✅ Please confirm your details:\n"
                for k in ORDER:
                    summary += f"- **{LABELS[k]}:** {st.session_state.inputs[k]}\n"
                summary += "\nType **YES** to confirm or **RESTART** to start over."
                add_assistant_once(summary)
                st.session_state.phase = "confirm"
            else:
                # Ask the next question (only once)
                next_key = ORDER[st.session_state.step]
                add_assistant_once(f"**{LABELS[next_key]}?**")

    elif st.session_state.phase == "confirm":
        t = user_text.strip().lower()
        if t == "restart":
            reset_all()
            st.rerun()
        elif t == "yes":
            # Predict
            X = scaler.transform([[st.session_state.inputs[k] for k in ORDER]])
            pred_pct = float(model.predict_proba(X)[0][1] * 100.0)

            result_msg = f"🧠 **Predicted heart disease risk: {pred_pct:.2f}%**"
            if pred_pct > 70:
                result_msg += "\n⚠️ High risk. Please consult a medical professional."
            elif pred_pct > 40:
                result_msg += "\n🔍 Moderate risk. A check-up is recommended."
            else:
                result_msg += "\n✅ Low risk. Keep up the healthy lifestyle."
            add_assistant_once(result_msg)

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie([pred_pct, 100 - pred_pct],
                   labels=['At Risk', 'No Risk'],
                   colors=['#e74c3c', '#2ecc71'],
                   autopct='%1.1f%%',
                   startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # PDF
            pdf = generate_pdf(st.session_state.inputs, pred_pct)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf,
                file_name="heart_risk_report.pdf",
                mime="application/pdf"
            )

            st.session_state.phase = "done"
            add_assistant_once("If you’d like another assessment, type anything or press **Restart chat**.")
        else:
            add_assistant_once("❌ Please type **YES** to confirm or **RESTART** to start over.")

    else:  # phase == "done"
        add_assistant_once("Type **RESTART** to begin again, or use the button above.")

