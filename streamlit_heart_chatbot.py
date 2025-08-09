# app.py
import os
import io
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ----------------------------
# Model / Scaler (works for LR, XGB, RF, etc.)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("model.pkl")         # your trained model (any sklearn/xgb)
    scaler = joblib.load("scaler.pkl")       # the scaler that matches that model
    return model, scaler

MODEL, SCALER = load_artifacts()

# ----------------------------
# App chrome
# ----------------------------
st.set_page_config(page_title="Heart Disease Risk Chatbot", page_icon="🩺", layout="wide")

header_left, header_right = st.columns([1, 0.35])
with header_left:
    st.markdown(
        "<h1 style='margin-bottom:0'>🩺 Heart Disease Risk Chatbot</h1>"
        "<p style='color:#C3C7D1;margin-top:6px'>"
        "I’ll guide you through 13 quick questions and show your result right away."
        "</p>",
        unsafe_allow_html=True,
    )
with header_right:
    # Restart that does not call st.rerun() inside callback
    def _reset_state():
        st.session_state.clear()
        st.session_state["_just_reset"] = True

    st.button("🔁 Restart chat", use_container_width=True, on_click=_reset_state)

if st.session_state.get("_just_reset"):
    st.session_state.pop("_just_reset")
    st.rerun()

# ----------------------------
# Chat helpers
# ----------------------------
ASSISTANT_ICON = "🩺"
USER_ICON = "🙂"
ERROR_ICON = "❌"

def assistant(msg: str):
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        st.markdown(msg)

def user(msg: str):
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(msg)

def error_line(msg: str):
    with st.chat_message("assistant", avatar=ASSISTANT_ICON):
        st.markdown(f"{ERROR_ICON} {msg}")

# ----------------------------
# Questions (order matters)
# ----------------------------
QUESTIONS = [
    {"key": "age",       "prompt": "Age?",                                   "type": "int",   "min": 1,   "max": 120},
    {"key": "sex",       "prompt": "Sex (0=Female, 1=Male)?",                "type": "choice","choices": [0, 1]},
    {"key": "cp",        "prompt": "Chest Pain Type (0–3)?",                  "type": "choice","choices": [0,1,2,3]},
    {"key": "trestbps",  "prompt": "Resting Blood Pressure (mm Hg)?",         "type": "int",   "min": 70,  "max": 220},
    {"key": "chol",      "prompt": "Cholesterol (mg/dl)?",                    "type": "int",   "min": 100, "max": 700},
    {"key": "fbs",       "prompt": "Fasting Blood Sugar > 120 (0=No, 1=Yes)?","type": "choice","choices": [0, 1]},
    {"key": "restecg",   "prompt": "Resting ECG (0–2)?",                      "type": "choice","choices": [0,1,2]},
    {"key": "thalach",   "prompt": "Max Heart Rate Achieved (bpm)?",          "type": "int",   "min": 60,  "max": 250},
    {"key": "exang",     "prompt": "Exercise-Induced Angina (0=No, 1=Yes)?",  "type": "choice","choices": [0, 1]},
    {"key": "oldpeak",   "prompt": "ST Depression by Exercise?",              "type": "float", "min": 0.0, "max": 6.5},
    {"key": "slope",     "prompt": "Slope of Peak Exercise ST (0–2)?",        "type": "choice","choices": [0,1,2]},
    {"key": "ca",        "prompt": "Major Vessels Colored (0–4)?",            "type": "choice","choices": [0,1,2,3,4]},
    {"key": "thal",      "prompt": "Thalassemia (0=Normal,1=Fixed,2=Reversible)?","type":"choice","choices":[0,1,2]},
]

FEATURE_ORDER = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

FIELD_LABELS = {
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
    "ca": "Major Vessels Colored (0–4)",
    "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)",
}

# ----------------------------
# State init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "result" not in st.session_state:
    st.session_state.result = None   # (risk_pct, figure_bytes, pdf_bytes)

# replay history
for role, text in st.session_state.messages:
    with st.chat_message(role, avatar=(ASSISTANT_ICON if role=="assistant" else USER_ICON)):
        st.markdown(text)

# show the current question (unless finished)
if st.session_state.step < len(QUESTIONS) and st.session_state.result is None:
    assistant(QUESTIONS[st.session_state.step]["prompt"])

# ----------------------------
# Parsing / validation
# ----------------------------
def parse_and_validate(text: str, q: dict):
    try:
        if q["type"] == "int":
            val = int(text)
            if val < q["min"] or val > q["max"]:
                return None, f"Value out of range. Allowed: {q['min']}–{q['max']}."
            return val, None
        if q["type"] == "float":
            val = float(text)
            if val < q["min"] or val > q["max"]:
                return None, f"Value out of range. Allowed: {q['min']}–{q['max']}."
            return val, None
        if q["type"] == "choice":
            val = int(text)
            if val not in q["choices"]:
                return None, f"Value must be one of: {', '.join(map(str, q['choices']))}."
            return val, None
    except ValueError:
        return None, "Please enter a number."
    return None, "Invalid input."

# ----------------------------
# Prediction + outputs
# ----------------------------
def make_prediction(answers: dict):
    # prepare features in the required order
    x = [answers[k] for k in FEATURE_ORDER]
    Xs = SCALER.transform([x])
    # probability of positive class
    proba = float(MODEL.predict_proba(Xs)[0][1]) * 100.0
    return proba

def render_pie(risk_pct: float):
    # small, clean pie
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    sizes = [risk_pct, max(0.0, 100 - risk_pct)]
    labels = ["At Risk", "No Risk"]
    explode = (0.05, 0.0)
    ax.pie(
        sizes, labels=labels, explode=explode,
        autopct="%1.1f%%", startangle=90, pctdistance=0.8
    )
    ax.axis("equal")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def render_pdf(answers: dict, risk_pct: float):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    t = c.beginText(40, 750)
    t.setFont("Helvetica", 12)
    t.textLine("Heart Disease Risk Assessment Report")
    t.textLine("-------------------------------------")
    for k in FEATURE_ORDER:
        t.textLine(f"{FIELD_LABELS[k]}: {answers[k]}")
    t.textLine("")
    t.textLine(f"Predicted Risk: {risk_pct:.2f}%")
    t.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(t)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def show_results():
    risk, fig_png, pdf_bytes = st.session_state.result
    # headline
    if risk >= 70:
        note = "⚠️ **High risk.** Please consult a medical professional."
    elif risk >= 40:
        note = "🔎 **Moderate risk.** A check‑up is recommended."
    else:
        note = "✅ **Low risk.** Keep up the healthy lifestyle!"

    assistant(f"🧠 **Predicted heart disease risk:** **{risk:.2f}%**  \n{note}")

    # columns for chart + downloads
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown("#### Risk distribution")
        st.image(fig_png, use_column_width=False)
    with c2:
        st.markdown("#### Download your report")
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name="heart_risk_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# If we already computed a result in a previous turn, show it persistently
if st.session_state.result is not None:
    show_results()

# ----------------------------
# Chat input (single consumption)
# ----------------------------
text = st.chat_input("Your answer…")
if text is not None:
    # Echo user message
    st.session_state.messages.append(("user", text))
    user(text)

    # If already finished (result shown), treat any input as restart prompt
    if st.session_state.result is not None:
        assistant("Type anything or press **Restart chat** (top‑right) to start again.")
        st.stop()

    # Otherwise, validate current question
    i = st.session_state.step
    q = QUESTIONS[i]
    value, err = parse_and_validate(text.strip(), q)
    if err:
        error_line(err)
        st.session_state.messages.append(("assistant", f"{ERROR_ICON} {err}"))
        st.stop()

    # Save answer and advance to next question
    st.session_state.answers[q["key"]] = value
    st.session_state.step += 1

    # If there are more questions, ask the next one
    if st.session_state.step < len(QUESTIONS):
        nxt = QUESTIONS[st.session_state.step]["prompt"]
        st.session_state.messages.append(("assistant", nxt))
        assistant(nxt)
        st.stop()

    # All questions answered → compute result once
    risk_pct = make_prediction(st.session_state.answers)
    fig_png = render_pie(risk_pct).getvalue()
    pdf_bytes = render_pdf(st.session_state.answers, risk_pct).getvalue()
    st.session_state.result = (risk_pct, fig_png, pdf_bytes)

    # Show results (and keep them visible on future reruns)
    show_results()
    st.stop()
