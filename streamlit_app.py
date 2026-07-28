import streamlit as st
import pandas as pd
import joblib
#Load the trained model and expected feature list
model = joblib.load('heart_disease_model.pkl')
model_features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Heart Disease Risk Predictor")

st.title(" Heart Disease Risk Predictor")
st.write("""
 Enter the patient's clinical measurements below to get a prediction.
""")

st.header("Patient Clinical Measurements")

#Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

with col2:
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                       format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 
                                                3: "Non-Anginal Pain", 4: "Asymptomatic"}[x])
    restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2],
                            format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 
                                                     2: "Left Ventricular Hypertrophy"}[x])
    slope = st.selectbox("ST Slope", options=[1, 2, 3],
                          format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7],
                         format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])

#Build input dataframe matching training format
input_dict = {
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'exang': 1 if exang == "Yes" else 0,
    'oldpeak': oldpeak,
    'ca': ca,
    'cp_2.0': 1 if cp == 2 else 0,
    'cp_3.0': 1 if cp == 3 else 0,
    'cp_4.0': 1 if cp == 4 else 0,
    'restecg_1.0': 1 if restecg == 1 else 0,
    'restecg_2.0': 1 if restecg == 2 else 0,
    'slope_2.0': 1 if slope == 2 else 0,
    'slope_3.0': 1 if slope == 3 else 0,
    'thal_6.0': 1 if thal == 6 else 0,
    'thal_7.0': 1 if thal == 7 else 0,
}

input_df = pd.DataFrame([input_dict])

#Keep only columns the model expects
input_df = input_df[[col for col in model_features if col in input_df.columns]]

#Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f" Disease Present — Probability: {probability:.1%}")
    else:
        st.success(f" No Disease Detected — Probability of disease: {probability:.1%}")

    st.progress(probability)