import streamlit as st
import pandas as pd
import joblib

#Load the trained model and expected feature list
model = joblib.load('heart_disease_model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

#Custom styling
st.markdown("""
<style>
    :root {
        --green: #1B5E4F;
        --green-dark: #164A3E;
        --coral: #E8604C;
        --bg: #F7F7F5;
        --ink: #1F2937;
    }
    .stApp {
        background-color: #EDEBE5;
    }
    .main-header {
        background: linear-gradient(135deg, #1B5E4F 0%, #24785F 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .result-card {
        background: #FDFCFA;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 6px solid var(--green);
    }
    .result-card.risk {
        border-left-color: var(--coral);
    }
    .gauge-track {
        background: #E8E8E4;
        border-radius: 999px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin: 0.75rem 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280 !important;
        margin-bottom: 0.3rem;
    }
    [data-testid="stSidebar"] {
        background-color: #E3EFE9;
    }
    /* Force readable text colors regardless of system dark mode */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1F2937 !important;
    }
    .result-card p, .result-card h2 {
        color: #1F2937 !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="popover"] div[role="listbox"],
    [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [data-baseweb="menu"] li,
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background-color: #DCEBE5 !important;
        color: #1F2937 !important;
    }
    /* Number inputs and text inputs */
    input, textarea {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    /* Align +/- stepper buttons with input field height */
    [data-testid="stNumberInputStepUp"],
    [data-testid="stNumberInputStepDown"] {
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-baseweb="input"] {
        align-items: stretch !important;
    }

    /* predict box */
    .stButton > button {
        background-color: var(--green) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1rem !important;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(27, 94, 79, 0.3);
        margin-top: 0.5rem;
    }
    .stButton > button:hover {
        background-color: var(--green-dark) !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(27, 94, 79, 0.4);
    }
    .stButton > button p {
        color: #FFFFFF !important;
    }
    /* Tint only the small toolbar strip (collapse arrow area), not the whole sidebar */
    [data-testid="stSidebarCollapsedControl"] {
        background-color: #1B5E4F !important;
    }
</style>
""", unsafe_allow_html=True)

#Header
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p>Enter a patient's clinical measurements in the sidebar to flag early risk 
    and support preventive care decisions.</p>
</div>
""", unsafe_allow_html=True)

#Sidebar: inputs
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.6rem 0 1rem 0; border-top: 2px solid #1B5E4F; border-bottom: 2px solid #1B5E4F; margin: -3.5rem 0 1.2rem 0;">
        <div style="font-size: 1.25rem; font-weight: 700; color: #1B5E4F; margin-top: 0.5rem;">🩺 Patient Details</div>
        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.15rem;">
            Fill in each field, then predict below
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

    st.markdown('<hr style="border: none; border-top: 1px solid #C7D9D1; margin: 1.4rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="margin-top:0;">Clinical Findings</div>', unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                       format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina",
                                                3: "Non-Anginal Pain", 4: "Asymptomatic"}[x])
    exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2],
                            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality",
                                                     2: "Left Ventricular Hypertrophy"}[x])
    slope = st.selectbox("ST Slope", options=[1, 2, 3],
                          format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7],
                         format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

    predict_clicked = st.button("Predict Risk", use_container_width=True, type="primary")

#Build input dataframe matching training format
input_dict = {
    'age': age, 'sex': 1 if sex == "Male" else 0, 'trestbps': trestbps, 'chol': chol,
    'thalach': thalach, 'exang': 1 if exang == "Yes" else 0, 'oldpeak': oldpeak, 'ca': ca,
    'cp_2.0': 1 if cp == 2 else 0, 'cp_3.0': 1 if cp == 3 else 0, 'cp_4.0': 1 if cp == 4 else 0,
    'restecg_1.0': 1 if restecg == 1 else 0, 'restecg_2.0': 1 if restecg == 2 else 0,
    'slope_2.0': 1 if slope == 2 else 0, 'slope_3.0': 1 if slope == 3 else 0,
    'thal_6.0': 1 if thal == 6 else 0, 'thal_7.0': 1 if thal == 7 else 0,
}
input_df = pd.DataFrame([input_dict])
input_df = input_df[[col for col in model_features if col in input_df.columns]]

#result
col1, col2 = st.columns([1.3, 1])

with col1:
    if predict_clicked:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        card_class = "result-card risk" if prediction == 1 else "result-card"
        bar_color = "#E8604C" if prediction == 1 else "#1B5E4F"
        label = "Disease Present" if prediction == 1 else "No Disease Detected"
        icon = "⚠️" if prediction == 1 else "✅"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="section-label">Prediction Result</div>
            <h2 style="margin:0.2rem 0;">{icon} {label}</h2>
            <p style="color:#6B7280; margin-bottom:0.2rem;">Estimated probability of heart disease</p>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{probability*100:.1f}%; background:{bar_color};"></div>
            </div>
            <p style="font-size:1.4rem; font-weight:700; margin:0;">{probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card">
            <div class="section-label">Prediction Result</div>
            <h2 style="margin:0.2rem 0; color:#9CA3AF;">Awaiting input</h2>
            <p style="color:#6B7280;">Fill in the patient's details in the sidebar and click 
            <strong>Predict Risk</strong> to see a result here.</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-label">About this model</div>', unsafe_allow_html=True)
    st.markdown("""
    Trained on the UCI Cleveland Heart Disease dataset using a feature-selected 
    Gradient Boosting classifier, prioritizing recall to minimize missed diagnoses.
    """)