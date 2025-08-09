# app.py — Chat style with validation, immediate results (pie + PDF), restart

import streamlit as st
import numpy as np
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Heart Disease Risk Chatbot", page_icon="🩺", layout="wide")

# ---------- Model loader (LR preferred, fallback to XGB/random forest etc.) ----------
def load_model_and_scaler():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, "LR"
    except Exception:
        pass
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler, "MODEL"

MODEL, SCALER, MODEL_KIND = load_model_and_scaler()

# ---------- Features & validation ----------
FIELDS = [
    dict(key="age",      label="Age?", kind="int",    min=1,   max=120, help=None),
    dict(key="sex",      label="Sex (0=Female, 1=Male)?", kind="choice", choices=[0,1]),
    dict(key="cp",       label="Chest Pain Type (0–3)?", kind="choice", choices=[0,1,2,3]),
    dict(key="trestbps", label="Resting Blood Pressure (mm Hg)?", kind="int",    min=80,  max=220),
    dict(key="chol",     label="Cholesterol (mg/dl)?",   kind="int",    min=100, max=700),
    dict(key="fbs",      label="Fasting Blood Sugar > 120 (0=No, 1=Yes)?", kind="choice", choices=[0,1]),
    dict(key="restecg",  label="Resting ECG (0–2)?",     kind="choice", choices=[0,1,2]),
    dict(key="thalach",  label="Max Heart Rate Achieved (bpm)?", kind="int",    min=60,  max=250),
    dict(key="exang",    label="Exercise‑Induced Angina (0=No, 1=Yes)?", kind="choice", choices=[0,1]),
    dict(key="oldpeak",  label="ST Depression by Exercise (e.g., 1.0)?", kind="float",  min=0.0, max=10.0, step=0.1),
    dict(key="slope",    label="Slope of Peak Exercise ST (0–2)?", kind="choice", choices=[0,1,2]),
    dict(key="ca",       label="Major Vessels Colored (0–4)?", kind="choice", choices=[0,1,2,3,4]),
    dict(key="thal",     label="Thalassemia (0=Normal,1=Fixed,2=Reversible)?", kind="choice", choices=[0,1,2]),
]
FEATURE_ORDER = [f["key"] for f in FIELDS]

FULL_NAMES = {
    "age":"Age", "sex":"Sex (0=Female, 1=Male)", "cp":"Chest Pain Type (0–3)",
    "trestbps":"Resting BP (mm Hg)", "chol":"Cholesterol (mg/dl)", "fbs":"Fasting Blood Sugar >120",
    "restecg":"Resting ECG (0–2)", "thalach":"Max Heart Rate (bpm)", "exang":"Exercise‑Induced Angina",
    "oldpeak":"ST Depression by Exercise", "slope":"Slope of Peak Exercise ST",
    "ca":"Major Vessels Colored (0–4)", "thal":"Thalassemia (0=Normal,1=Fixed,2=Reversible)"
}

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "stage" not in st.session_state:
    st.session_state.stage = "asking"   # asking | done

# ---------- Helpers ----------
def add_msg(role, text):
    st.session_state.messages.append({"role": role, "text": text})

def current_field():
    return FIELDS[st.session_state.idx]

def ask_question_once():
    """Ask the current question exactly once."""
    q = current_field()["label"]
    add_msg("assistant", q)

def parse_and_validate(raw, field):
    raw = raw.strip()
    kind = field["kind"]

    if kind == "choice":
        try:
            v = int(raw)
        except ValueError:
            return None, f"❌ Please enter one of: {', '.join(map(str, field['choices']))}."
        if v not in field["choices"]:
            return None, f"❌ Value must be one of: {', '.join(map(str, field['choices']))}."
        return v, None

    if kind == "int":
        try: v = int(raw)
        except ValueError: return None, "❌ Please enter a whole number."
        if v < field["min"] or v > field["max"]:
            return None, f"❌ Value out of range ({field['min']}–{field['max']})."
        return v, None

    if kind == "float":
        try: v = float(raw)
        except ValueError: return None, "❌ Please enter a number (e.g., 1.0)."
        if v < field["min"] or v > field["max"]:
            return None, f"❌ Value out of range ({field['min']}–{field['max']})."
        # round to sensible step for display
        step = field.get("step", 0.1)
        v = round(v/step)*step
        return v, None

    return None, "❌ Invalid input."

def predict_prob(answers):
    x = [answers[k] for k in FEATURE_ORDER]
    x_scaled = SCALER.transform([x])
    if hasattr(MODEL, "predict_proba"):
        p = MODEL.predict_proba(x_scaled)[0][1]
    else:
        # e.g., SVM decision_function → sigmoid fallback
        z = MODEL.decision_function(x_scaled)[0]
        p = 1.0/(1.0+np.exp(-z))
    return float(p)

def risk_text(pct):
    if pct >= 70:  return "⚠️ High risk. Please consult a medical professional."
    if pct >= 40:  return "🔍 Moderate risk. A check‑up is recommended."
    return "✅ Low risk. Keep up the healthy lifestyle!"

def make_pdf(answers, pct):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    t = c.beginText(40, 750)
    t.setFont("Helvetica", 12)
    t.textLine("Heart Disease Risk Assessment Report")
    t.textLine("------------------------------------")
    for k in FEATURE_ORDER:
        t.textLine(f"{FULL_NAMES[k]}: {answers[k]}")
    t.textLine("")
    t.textLine(f"Predicted risk: {pct:.2f}%")
    c.drawText(t)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def show_results_block():
    pct = predict_prob(st.session_state.answers)*100.0

    with st.chat_message("assistant"):
        st.markdown(f"**🧠 Predicted heart disease risk:** **{pct:.2f}%**")
        st.markdown(risk_text(pct))

        # Pie chart inside the chat bubble
        fig, ax = plt.subplots()
        parts = [pct, 100-pct]
        ax.pie(parts, labels=["At Risk", "No Risk"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # PDF download
        pdf = make_pdf(st.session_state.answers, pct)
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name="heart_risk_report.pdf", mime="application/pdf")

    add_msg("assistant", "If you’d like another assessment, type **restart** or press the **Restart chat** button above.")
    st.session_state.stage = "done"

def restart():
    for k in ["messages","intro_shown","idx","answers","stage"]:
        if k in st.session_state: del st.session_state[k]
    st.rerun()

# ---------- Header + Restart ----------
col1, col2 = st.columns([0.78, 0.22])
with col1:
    st.markdown("## 🩺 Heart Disease Risk Chatbot")
    st.caption("I’ll guide you through 13 quick questions and show your result right away.")
with col2:
    st.button("🔁 Restart chat", use_container_width=True, on_click=restart)

# ---------- Render prior messages ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

# ---------- Intro & first prompt (once) ----------
if not st.session_state.intro_shown:
    add_msg("assistant", "👋 Hello! I’m your virtual triage assistant.")
    add_msg("assistant", "We’ll go step‑by‑step. Please answer with numbers exactly as requested.")
    st.session_state.intro_shown = True
    ask_question_once()

# ---------- Chat input ----------
raw = st.chat_input("Your answer…")

if raw is not None:
    # echo user message
    with st.chat_message("user"):
        st.markdown(raw)
    add_msg("user", raw)

    # If already finished, allow restart by typing 'restart'
    if st.session_state.stage == "done":
        if raw.strip().lower() == "restart":
            restart()
        else:
            add_msg("assistant", "Type **restart** to begin again, or use the button above.")
            with st.chat_message("assistant"):
                st.markdown("Type **restart** to begin again, or use the button above.")
    else:
        # parse/validate current answer
        field = current_field()
        val, err = parse_and_validate(raw, field)
        if err:
            add_msg("assistant", err)
            with st.chat_message("assistant"):
                st.markdown(err)
        else:
            st.session_state.answers[field["key"]] = val
            st.session_state.idx += 1

            # Next question or results
            if st.session_state.idx < len(FIELDS):
                ask_question_once()
                with st.chat_message("assistant"):
                    st.markdown(FIELDS[st.session_state.idx]["label"])
            else:
                show_results_block()
