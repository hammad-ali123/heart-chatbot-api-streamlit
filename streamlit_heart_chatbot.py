import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -----------------------------
# Load model & scaler (LR files)
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Config & schema
# -----------------------------
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

ASSISTANT_AVATAR = "🩺"
USER_AVATAR = "🙂"

# -----------------------------
# Helpers
# -----------------------------
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

def generate_pdf(input_data, prediction_pct):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for k in ORDER:
        text.textLine(f"{LABELS[k]}: {input_data[k]}")
    text.textLine("")
    text.textLine(f"Predicted Risk: {round(prediction_pct, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def append_assistant(msg):
    st.session_state.messages.append({"role": "assistant", "content": msg})

def append_user(msg):
    st.session_state.messages.append({"role": "user", "content": msg})

def ensure_prompt_for_current_step():
    """Append exactly one prompt for the current step (or confirmation), no duplicates."""
    if st.session_state.await_confirm:
        needed = "Type **YES** to confirm or **RESTART** to start over."
    else:
        key = ORDER[st.session_state.step]
        needed = f"**{LABELS[key]}?**"

    # If messages empty or last assistant message is different, append
    last_assistant = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break
    if last_assistant != needed:
        append_assistant(needed)

# -----------------------------
# Session state init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = {}
    st.session_state.await_confirm = False
    st.session_state.greeted = False

# -----------------------------
# Header (always visible)
# -----------------------------
st.title("🫀 Heart Disease Risk Chatbot")
st.caption("Answer a few questions and I’ll estimate your heart disease risk. "
           "Values are validated to keep things realistic.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔄 Restart"):
        st.session_state.step = 0
        st.session_state.inputs = {}
        st.session_state.await_confirm = False
        st.session_state.messages = []
        st.session_state.greeted = False
        st.experimental_rerun()

# -----------------------------
# Greeting & first prompt
# -----------------------------
if not st.session_state.greeted:
    append_assistant("👋 Hello! I’m your virtual triage assistant.")
    append_assistant("We’ll go step‑by‑step. Please enter numbers exactly as requested.")
    st.session_state.greeted = True

# Make sure we have exactly one active prompt
ensure_prompt_for_current_step()

# -----------------------------
# Render chat history
# -----------------------------
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# -----------------------------
# Chat input (single point)
# -----------------------------
user_text = st.chat_input("Your answer…")
if user_text:
    append_user(user_text)

    if not st.session_state.await_confirm:
        # Validate current step input
        current_key = ORDER[st.session_state.step]
        val, err = parse_and_validate(current_key, user_text)
        if err:
            append_assistant(err + f" Please re‑enter **{LABELS[current_key]}**.")
        else:
            st.session_state.inputs[current_key] = val
            st.session_state.step += 1

            if st.session_state.step == len(ORDER):
                # All answers collected → show summary and ask to confirm
                summary = "✅ Please confirm your details:\n"
                for k in ORDER:
                    summary += f"- **{LABELS[k]}:** {st.session_state.inputs[k]}\n"
                summary += "\nType **YES** to confirm or **RESTART** to start over."
                append_assistant(summary)
                st.session_state.await_confirm = True
            else:
                # Ask the next question (only once)
                next_key = ORDER[st.session_state.step]
                append_assistant(f"**{LABELS[next_key]}?**")

    else:
        # Awaiting confirmation
        t = user_text.strip().lower()
        if t == "restart":
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.await_confirm = False
            st.session_state.messages = []
            st.session_state.greeted = False
            st.experimental_rerun()

        elif t == "yes":
            X = scaler.transform([[st.session_state.inputs[k] for k in ORDER]])
            pred_pct = float(model.predict_proba(X)[0][1] * 100.0)

            result_msg = f"🧠 **Predicted heart disease risk: {pred_pct:.2f}%**"
            if pred_pct > 70:
                result_msg += "\n⚠️ High risk. Please consult a medical professional."
            elif pred_pct > 40:
                result_msg += "\n🔍 Moderate risk. A check-up is recommended."
            else:
                result_msg += "\n✅ Low risk. Keep up the healthy lifestyle."
            append_assistant(result_msg)

            # Visual pie
            fig, ax = plt.subplots()
            ax.pie([pred_pct, 100 - pred_pct],
                   labels=['At Risk', 'No Risk'],
                   colors=['#e74c3c', '#2ecc71'],
                   autopct='%1.1f%%', startangle=90)
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

            # Prepare for next session (keep conversation visible)
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.await_confirm = False
            append_assistant("If you’d like to run another assessment, type anything or press **Restart**.")
        else:
            append_assistant("❌ Please type **YES** to confirm or **RESTART** to start over.")

# Ensure only one active prompt is visible (no duplicates)
ensure_prompt_for_current_step()
