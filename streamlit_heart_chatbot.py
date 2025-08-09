import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ========= Load model & scaler =========
model = joblib.load("lr_model.pkl")
scaler = joblib.load("lr_scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Chatbot", page_icon="🫀")
st.title("🫀 Heart Disease Risk Chatbot (Chat-Style)")

st.markdown(
    "Please answer each question. "
    "**Allowed values** are shown in brackets. I’ll re-ask anything that looks invalid."
)

# ========= Validation schema (tight but sensible) =========
SCHEMA = {
    "age":       {"type": int,   "min": 1,   "max": 120, "prompt": "What is your age? (1–120)"},
    "sex":       {"type": int,   "allowed": {0, 1},      "prompt": "Biological sex? (0 = Female, 1 = Male)"},
    "cp":        {"type": int,   "allowed": {0,1,2,3},   "prompt": "Chest pain type? (0,1,2,3)"},
    "trestbps":  {"type": int,   "min": 80,  "max": 250, "prompt": "Resting blood pressure (mm Hg)? (80–250)"},
    "chol":      {"type": int,   "min": 100, "max": 700, "prompt": "Cholesterol (mg/dl)? (100–700)"},
    "fbs":       {"type": int,   "allowed": {0,1},       "prompt": "Fasting blood sugar > 120 mg/dl? (0 = No, 1 = Yes)"},
    "restecg":   {"type": int,   "allowed": {0,1,2},     "prompt": "Resting ECG result? (0,1,2)"},
    "thalach":   {"type": int,   "min": 60,  "max": 250, "prompt": "Max heart rate achieved? (60–250)"},
    "exang":     {"type": int,   "allowed": {0,1},       "prompt": "Exercise-induced angina? (0 = No, 1 = Yes)"},
    "oldpeak":   {"type": float, "min": 0.0, "max": 6.0, "prompt": "ST depression induced by exercise? (0.0–6.0)"},
    "slope":     {"type": int,   "allowed": {0,1,2},     "prompt": "Slope of peak exercise ST segment? (0,1,2)"},
    "ca":        {"type": int,   "allowed": {0,1,2,3,4}, "prompt": "Number of major vessels coloured by fluoroscopy? (0–4)"},
    "thal":      {"type": int,   "allowed": {0,1,2},     "prompt": "Thalassemia? (0=Normal, 1=Fixed, 2=Reversible)"},
}

ORDER = list(SCHEMA.keys())

# Pretty labels for the PDF/summary
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

# ========= Helpers =========
def parse_and_validate(key: str, text: str):
    """Parse input and validate against schema. Return (value, error_msg_or_None)."""
    spec = SCHEMA[key]
    typ = spec["type"]
    # Parse
    try:
        if typ is int:
            val = int(float(text))  # allow "120.0" -> 120
        elif typ is float:
            val = float(text)
        else:
            return None, "Internal type error."
    except ValueError:
        return None, "Please enter a valid number."

    # Check allowed set
    if "allowed" in spec and val not in spec["allowed"]:
        allowed_str = ", ".join(str(a) for a in sorted(spec["allowed"]))
        return None, f"Value must be one of: {allowed_str}."

    # Check range
    if "min" in spec and val < spec["min"]:
        return None, f"Value too low. Minimum is {spec['min']}."
    if "max" in spec and val > spec["max"]:
        return None, f"Value too high. Maximum is {spec['max']}."
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
    text.textLine(f"")
    text.textLine(f"Predicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ========= Session state =========
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.inputs = {}
    st.session_state.await_confirm = False  # after collecting all inputs

# ========= Interaction flow =========
# 1) Ask all questions with validation
if st.session_state.step < len(ORDER) and not st.session_state.await_confirm:
    key = ORDER[st.session_state.step]
    prompt = SCHEMA[key]["prompt"]
    user_text = st.chat_input(prompt)
    if user_text:
        val, err = parse_and_validate(key, user_text)
        if err:
            st.error(f"❌ {err} Please try again. {prompt}")
        else:
            st.session_state.inputs[key] = val
            st.session_state.step += 1

# 2) After all fields collected, show summary and ask to confirm
if st.session_state.step == len(ORDER) and not st.session_state.await_confirm:
    st.markdown("### ✅ Please confirm your details")
    for k in ORDER:
        st.write(f"- **{LABELS[k]}:** {st.session_state.inputs[k]}")
    st.info("Type **YES** to confirm, or **restart** to start over.")
    st.session_state.await_confirm = True

# 3) Handle confirmation or restart
if st.session_state.await_confirm:
    confirm = st.chat_input("Type YES to confirm or RESTART to start again")
    if confirm:
        lowered = confirm.strip().lower()
        if lowered == "restart":
            st.session_state.step = 0
            st.session_state.inputs = {}
            st.session_state.await_confirm = False
            st.warning("Restarted. Let's begin again.")
        elif lowered == "yes":
            # Final **server-side** validation gate before prediction
            for k in ORDER:
                val, err = parse_and_validate(k, str(st.session_state.inputs[k]))
                if err:
                    st.error(f"Validation failed for {LABELS[k]}: {err}")
                    st.session_state.step = ORDER.index(k)
                    st.session_state.await_confirm = False
                    break
            else:
                # All good: predict
                input_list = [st.session_state.inputs[k] for k in ORDER]
                X = scaler.transform([input_list])
                pred_pct = model.predict_proba(X)[0][1] * 100.0

                st.success(f"🧠 Your predicted heart disease risk is **{pred_pct:.2f}%**.")
                if pred_pct > 70:
                    st.warning("⚠️ High risk. Please consult a medical professional.")
                elif pred_pct > 40:
                    st.info("🔍 Moderate risk. A check-up is recommended.")
                else:
                    st.info("✅ Low risk. Keep up the healthy lifestyle!")

                # Pie chart
                labels = ['At Risk', 'No Risk']
                sizes = [round(pred_pct, 2), 100 - round(pred_pct, 2)]
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, colors=['red', 'green'],
                       autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.markdown("### 🧩 Risk Distribution")
                st.pyplot(fig)

                # PDF
                pdf = generate_pdf(st.session_state.inputs, pred_pct)
                st.download_button("📄 Download PDF Report",
                                   data=pdf,
                                   file_name="heart_risk_report.pdf",
                                   mime="application/pdf")

                # Reset for next user
                st.session_state.step = 0
                st.session_state.inputs = {}
                st.session_state.await_confirm = False
        else:
            st.error("Please type **YES** to confirm or **RESTART** to start over.")
