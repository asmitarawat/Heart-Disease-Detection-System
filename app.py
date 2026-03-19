import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model_joblib_heart')

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Header
st.markdown("<h1 style='text-align: center; color: #e74c3c;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter patient details below to predict heart disease risk</p>", unsafe_allow_html=True)
st.markdown("---")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Info")
    age      = st.slider("Age", 20, 80, 52)
    sex      = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            help="0=No pain, 1=Mild, 2=Moderate, 3=Severe")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 125)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 212)
    fbs      = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    restecg  = st.selectbox("Resting ECG Results", [0, 1, 2])

with col2:
    st.subheader("🫀 Heart Metrics")
    thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 168)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope    = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca       = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal     = st.selectbox("Thal", [0, 1, 2, 3])

st.markdown("---")

# Predict button centered
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    predict_btn = st.button("🔍 Predict Now", use_container_width=True)

if predict_btn:
    new_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp],
        'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
        'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })

    result = model.predict(new_data)[0]
    prob   = model.predict_proba(new_data)[0][1]

    st.markdown("---")
    if result == 0:
        st.success(f"✅ No Heart Disease Detected — {(1-prob)*100:.1f}% confidence")
        st.balloons()
    else:
        st.error(f"⚠️ Possibility of Heart Disease — {prob*100:.1f}% probability")
        st.warning("Please consult a doctor for a proper diagnosis.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: 12px;'>Built by Asmita Rawat · Heart Disease Detection ML Project</p>", unsafe_allow_html=True)
