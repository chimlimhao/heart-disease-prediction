import gradio as gr
import numpy as np
import pandas as pd
from joblib import load
import os

# Specify the direct path to the models directory
models_dir = 'models'

# Load the model files with absolute paths
model = load(os.path.join(models_dir, 'heart_disease_model.joblib'))
scaler = load(os.path.join(models_dir, 'scaler.joblib'))

# Load feature names from file
with open(os.path.join(models_dir, 'feature_names.txt'), 'r') as f:
    feature_names = f.read().split(',')

def predict_heart_disease(age, sex, chest_pain, resting_bp, cholesterol, 
                         fasting_bs, resting_ecg, max_hr, 
                         exercise_angina, oldpeak, st_slope):
    # Create a dataframe with the input
    data = pd.DataFrame({
        'Age': [age],
        'Sex_M': [1 if sex == "Male" else 0],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'MaxHR': [max_hr],
        'Oldpeak': [oldpeak],
        'ChestPainType_ATA': [1 if chest_pain == "ATA" else 0],
        'ChestPainType_NAP': [1 if chest_pain == "NAP" else 0],
        'ChestPainType_TA': [1 if chest_pain == "TA" else 0],
        'RestingECG_Normal': [1 if resting_ecg == "Normal" else 0],
        'RestingECG_ST': [1 if resting_ecg == "ST" else 0],
        'ExerciseAngina_Y': [1 if exercise_angina == "Yes" else 0],
        'ST_Slope_Flat': [1 if st_slope == "Flat" else 0],
        'ST_Slope_Up': [1 if st_slope == "Up" else 0]
    })
    
    # Scale numerical features
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # Make prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]
    
    return {
        "No Heart Disease": float(probability[0]),
        "Heart Disease": float(probability[1])
    }

# Create Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Slider(25, 80, label="Age", info="Patient's age in years"),
        gr.Radio(["Male", "Female"], label="Sex", 
                info="Men have higher risk before 55, women after menopause"),
        gr.Radio(["ATA", "NAP", "TA", "ASY"], label="Chest Pain Type", 
                info="TA: Typical Angina | ATA: Atypical Angina | NAP: Non-Anginal Pain | ASY: Asymptomatic"),
        gr.Number(label="Resting Blood Pressure (mmHg)", 
                 info="Normal range: 90-120 mmHg. High BP is a risk factor"),
        gr.Number(label="Cholesterol (mg/dl)", 
                 info="Normal: <200 mg/dL, Borderline: 200-239, High: >240"),
        gr.Radio(["0", "1"], label="Fasting Blood Sugar > 120 mg/dl", 
                info="1 = diabetic, 0 = normal. Diabetes increases heart disease risk"),
        gr.Radio(["Normal", "ST", "LVH"], label="Resting ECG", 
                info="Normal: No abnormalities | ST: ST-T wave abnormality | LVH: Left ventricular hypertrophy"),
        gr.Number(label="Maximum Heart Rate", 
                 info="Max rate achieved during exercise. Normal: 220-age"),
        gr.Radio(["Yes", "No"], label="Exercise Induced Angina", 
                info="Chest pain during exercise is an important indicator of coronary disease"),
        gr.Number(label="ST Depression (Oldpeak)", 
                 info="ST depression induced by exercise. Higher values indicate ischemia"),
        gr.Radio(["Up", "Flat", "Down"], label="ST Slope", 
                info="Up: good sign | Flat: concerning | Down: suggests ischemia")
    ],
    outputs=gr.Label(label="Heart Disease Prediction"),
    title="Heart Disease Risk Prediction",
    description="""This tool estimates the risk of heart disease based on clinical data.
                  Enter the patient's information to get a prediction. 
                  Note: This is for educational purposes only and not a medical diagnosis.""",
    article="""
    <div style="text-align: left; max-width: 800px; margin: 0 auto;">
        <h3>Understanding Heart Disease Risk Factors</h3>
        <p>Heart disease risk is influenced by multiple factors:</p>
        <ul>
            <li><strong>Age & Sex</strong>: Risk increases with age, and varies by sex</li>
            <li><strong>Chest Pain</strong>: Different types indicate varying risk levels</li>
            <li><strong>Blood Pressure & Cholesterol</strong>: Key modifiable risk factors</li>
            <li><strong>Diabetes</strong>: Significantly increases heart disease risk</li>
            <li><strong>Exercise Response</strong>: How the heart responds to stress</li>
        </ul>
        <p>This model was trained on the UCI Heart Disease dataset with 918 patients.</p>
    </div>
    """
)

# Launch the interface
iface.launch(share=True)