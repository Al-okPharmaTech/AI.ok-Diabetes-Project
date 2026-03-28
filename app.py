import streamlit as st
import joblib
import numpy as np

# --- STEP 1: Load the 'Frozen' Brains ---
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# --- STEP 2: The Help Tooltips (Clinical Ranges) ---
help_text = {
    "Glucose": "Normal Post-Prandial: 70–140 mg/dL. Prediabetes: 140–199 mg/dL.",
    "BloodPressure": "Normal Systolic: <120 mmHg. Hypertension: >140 mmHg.",
    "BMI": "Normal: 18.5–24.9. Overweight: 25–29.9. Obese: >30.",
    "Age": "Age of the patient in years.",
    "Insulin": "Normal Fasting: <25 mIU/L.",
    "Pregnancies": "Number of times pregnant.",
    "SkinThickness": "Triceps skin fold thickness (mm).",
    "Pedigree": "Diabetes Pedigree Function (Genetic risk score)."
}

# --- STEP 3: The AI.ok Branding ---
st.set_page_config(page_title="AI.ok Diagnostic", page_icon="🩺")
st.title("🩺 AI.ok | Diabetes Assistant")
st.write(f"Developed by: **Alok Shah** | *Providing Explainable AI (XAI) for Pharmacy Research*")
st.markdown("---")

# --- STEP 4: Reset Logic Initialization ---
if 'reset' not in st.session_state:
    st.session_state.reset = False

def trigger_reset():
    st.session_state.reset = True

# --- STEP 5: Create the Input UI ---
inputs = {}
cols = st.columns(2)

for i, col_name in enumerate(columns):
    with cols[i % 2]:
        # If reset was clicked, we show 0.0, otherwise we allow current input
        val = 0.0 if st.session_state.reset else 0.0
        inputs[col_name] = st.number_input(
            f"Enter {col_name}", 
            value=val, 
            key=f"{col_name}_{st.session_state.reset}", # Key change forces refresh
            help=help_text.get(col_name, "Clinical metric")
        )

# Reset the trigger after inputs are rendered
if st.session_state.reset:
    st.session_state.reset = False

st.markdown("---")

# --- STEP 6: Action Buttons ---
btn_col1, btn_col2 = st.columns([1, 4]) # Smaller column for Reset, larger for Report

with btn_col1:
    st.button("Clear All", on_click=trigger_reset, help="Reset all fields to zero")

with btn_col2:
    generate_btn = st.button("Generate Clinical Report")

# --- STEP 7: The Calculation & XAI ---
if generate_btn:
    features = np.array([inputs[col] for col in columns]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1] * 100
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.error(f"⚠️ High Risk: {prob:.2f}% Probability of Diabetes")
        st.subheader("🔍 Why this result? (XAI Analysis)")
        if inputs['Glucose'] > 140: st.write("- Blood Glucose is above normal range.")
        if inputs['BMI'] > 30: st.write("- BMI indicates high metabolic risk.")
    else:
        st.success(f"✅ Low Risk: {prob:.2f}% Probability of Diabetes")

# --- STEP 8: The Professional Disclaimer ---
st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This AI tool (AI.ok) is a research prototype developed for educational and pharmacy informatics study. It is **not** a substitute for professional medical diagnosis or clinical advice. Always consult a certified healthcare professional.")
