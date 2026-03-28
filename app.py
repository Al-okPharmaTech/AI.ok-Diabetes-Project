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

# --- STEP 4: Create the Input UI with Tooltips ---
inputs = {}
cols = st.columns(2)

for i, col_name in enumerate(columns):
    with cols[i % 2]:
        # The 'help' parameter adds the hover-over info!
        inputs[col_name] = st.number_input(
            f"Enter {col_name}", 
            value=0.0, 
            help=help_text.get(col_name, "Clinical metric")
        )

st.markdown("---")

# --- STEP 5: The Calculation & XAI ---
if st.button("Generate Clinical Report"):
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

# --- STEP 6: The Professional Disclaimer ---
st.markdown("---")
st.caption("⚠️ **Medical Disclaimer:** This AI tool (AI.ok) is a research prototype developed for educational and pharmacy informatics study. It is **not** a substitute for professional medical diagnosis or clinical advice. Always consult a certified healthcare professional.")