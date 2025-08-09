import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------- Model / Scaler ----------
MODEL_PATH = "model.pkl"     # compatible with LR or XGB saved as joblib
SCALER_PATH = "scaler.pkl"   # StandardScaler saved as joblib
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------- Page ----------
st.set_page_config(page_title="Heart Risk Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Heart Disease Risk Chatbot")
st.caption("Answer step‑by‑step. I’ll compute your risk right after the last question.")

# ---------- Questions & validation ----------
QUESTIONS = [
    # key, prompt, min, max, allowed
    ("age",       "Age?",                                  1,   120, None),
    ("sex",       "Sex (0=Female, 1=Male)?",               0,   1,   {0,1}),
    ("cp",        "Chest Pain Type (0–3)?",                0,   3,   {0,1,2,3}),
    ("trestbps",  "Resting Blood Pressure (mm Hg)?",       80,  250, None),
    ("chol",      "Cholesterol (mg/dl)?",                  100, 700, None),
    ("fbs",       "Fasting Blood Sugar > 120 (0=No,1=Yes)?",0,  1,   {0,1}),
    ("restecg",   "Resting ECG (0–2)?",                    0,   2,   {0,1,2}),
    ("thalach",   "Max Heart Rate Achieved (bpm)?",        60,  250, None),
    ("exang",     "Exercise‑Induced Angina (0=No,1=Yes)?", 0,   1,   {0,1}),
    ("oldpeak",   "ST Depression by Exercise (e.g., 1.4)?",0.0, 6.5, None),
    ("slope",     "Slope of Peak Exercise ST (0–2)?",      0,   2,   {0,1,2}),
    ("ca",        "Major Vessels Colored (0–4)?",          0,   4,   {0,1,2,3,4}),
    ("thal",      "Thalassemia (0=Normal,1=Fixed,2=Reversible)?", 0, 2, {0,1,2}),
]

ORDER = [q[0] for q in QUESTIONS]

# ---------- Utils ----------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "done" not in st.session_state:
        st.session_state.done = False
    if "just_started" not in st.session_state:
        st.session_state.just_started = True

def say_bot(text):
    st.session_state.messages.append({"role": "assistant", "content": text})

def say_user(text):
    st.session_state.messages.append({"role": "user", "content": text})

def validate_value(key, raw):
    """Return (ok, value_or_errmsg). Coerce ints/floats as needed."""
    # decide type
    is_float = (key == "oldpeak")
    try:
        val = float(raw) if is_float else int(float(raw))
    except Exception:
        return False, "❌ Please enter a number."

    # get question meta
    for k, prompt, vmin, vmax, allowed in QUESTIONS:
        if k == key:
            # allowed set check first (for categorical)
            if allowed is not None and val not in allowed:
                opts = ", ".join(map(str, sorted(list(allowed))))
                return False, f"❌ Value must be one of: {opts}. Please re‑enter {prompt}"
            # range check
            if val < vmin or val > vmax:
                return False, f"❌ Value out of range ({vmin}–{vmax}). Please re‑enter {prompt}"
            # cast back to int if needed
            if not is_float:
                val = int(val)
            return True, val
    return False, "❌ Unknown field."

def answers_to_array(ans_dict):
    seq = [
        ans_dict["age"], ans_dict["sex"], ans_dict["cp"], ans_dict["trestbps"],
        ans_dict["chol"], ans_dict["fbs"], ans_dict["restecg"], ans_dict["thalach"],
        ans_dict["exang"], ans_dict["oldpeak"], ans_dict["slope"], ans_dict["ca"],
        ans_dict["thal"],
    ]
    return np.array(seq, dtype=float).reshape(1, -1)

def predict_and_show():
    # Assemble -> scale -> predict
    X = answers_to_array(st.session_state.answers)
    Xs = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(Xs)[0, 1]) * 100.0
    else:
        # For models without predict_proba, fall back to decision function
        p = float(model.decision_function(Xs))
        prob = 100.0 / (1.0 + np.exp(-p))
    risk = round(prob, 2)

    # Text result
    if risk > 70:
        verdict = "⚠️ High risk. Please consult a medical professional."
    elif risk > 40:
        verdict = "🔍 Moderate risk. A check‑up is recommended."
    else:
        verdict = "✅ Low risk. Keep up the healthy lifestyle!"

    say_bot(f"🧠 **Predicted heart disease risk:** **{risk}%**\n\n{verdict}")

    # Pie chart
    labels = ["At Risk", "No Risk"]
    sizes = [risk, 100 - risk]
    colors = ["#e74c3c", "#2ecc71"]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # PDF download
    pdf = make_pdf(st.session_state.answers, risk)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf,
        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

    # Restart option
    st.divider()
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🔄 Restart chat"):
            for k in ["messages", "idx", "answers", "done", "just_started"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    with col2:
        say_bot("If you’d like another assessment, type anything or press **Restart chat** above.")

def make_pdf(data, risk):
    labels = {
        "age": "Age",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 (0=No,1=Yes)",
        "restecg": "Resting ECG (0–2)",
        "thalach": "Max Heart Rate Achieved (bpm)",
        "exang": "Exercise‑Induced Angina (0=No,1=Yes)",
        "oldpeak": "ST Depression by Exercise",
        "slope": "Slope of Peak Exercise ST (0–2)",
        "ca": "Major Vessels Colored (0–4)",
        "thal": "Thalassemia (0=Normal,1=Fixed,2=Reversible)",
    }
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    t = c.beginText(40, 750)
    t.setFont("Helvetica", 12)
    t.textLine("Heart Disease Risk Assessment Report")
    t.textLine("--------------------------------------")
    for k in ORDER:
        t.textLine(f"{labels[k]}: {data[k]}")
    t.textLine(f"\nPredicted Risk: {risk}%")
    t.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(t); c.showPage(); c.save()
    buf.seek(0)
    return buf

# ---------- Init ----------
init_state()

# First welcome + first question (once)
if st.session_state.just_started:
    say_bot("Hi! I’ll ask you 13 quick questions and then show your result right away. "
            "You can answer with numbers only. Let’s begin.")
    # ask first
    first_prompt = QUESTIONS[0][1]
    say_bot(f"{first_prompt}")
    st.session_state.just_started = False

# Render history
for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg["role"]=="assistant" else "user"):
        st.markdown(msg["content"])

# If already finished, do nothing (buttons shown above)
if st.session_state.done:
    st.stop()

# Chat input
user_text = st.chat_input("Your answer...")
if user_text is not None:
    # Show user bubble
    say_user(user_text)

    # Validate current question
    key, prompt, *_ = QUESTIONS[st.session_state.idx]
    ok, val = validate_value(key, user_text)

    if not ok:
        say_bot(val)                 # val contains the error message and prompt
    else:
        # Save, advance
        st.session_state.answers[key] = val
        st.session_state.idx += 1

        # If finished, show results immediately
        if st.session_state.idx >= len(QUESTIONS):
            st.session_state.done = True
            predict_and_show()
        else:
            # Ask next question
            next_prompt = QUESTIONS[st.session_state.idx][1]
            say_bot(next_prompt)

    # Re-render with updated messages
    st.rerun()
