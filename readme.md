#heart disease risk predictor model 
Predicts risk of heart disease based on clincal measurements, this is built to help care providers flag at-risk patients early for early preventive intervention.

#Dataset UCI Machine Learning Repository (Heart Disease Dataset) 
https://archive.ics.uci.edu/dataset/45/heart+disease

#Steps
Exploratory data analysis and data cleaning
Compared Logistic Regression, Random Forest and Gradient Boosting
Feature selection and hyperparameter tuning using RandomizedSearchCV
Chose Gradient Boosting as final model

#Web App
Streamlit link:

#Requirements
Python 3.13 
streamlit==1.58.0 
pandas==2.3.3 
scikit-learn==1.9.0 
joblib==1.5.3 
numpy==2.4.4 
matplotlib==3.10.8 
seaborn==0.13.2

#Web App
Link:https://mldp-heart-disease-model.streamlit.app/ 

#Files
`MLDP Program Codes.ipynb` full analysis and model development notebook
`streamlit_app.py` Streamlit web application
`heart_disease_model.pkl` and `model_features.pkl` saved trained model
`heart_disease_model.pkl` actual model
`model_features.pkl` all features for streamlit