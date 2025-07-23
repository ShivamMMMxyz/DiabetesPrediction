import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, render_template
from datetime import datetime

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Scaler ---
try:
    model_path = 'gradient_boosting_model.joblib' 
    model = joblib.load(model_path)
    
    scaler_path = 'scaler.joblib'
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Make sure '{model_path}' and '{scaler_path}' are in the correct directory.")
    model = None
    scaler = None

# --- Define the column order from the training data ---
# This list has been corrected based on the error message.
model_columns = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'gender_Male', 'smoking_history_current',
    'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', # CORRECTED: Changed 'no info' to 'never'
    'smoking_history_not current'
]

columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# --- Define Routes ---

@app.route('/')
def home():
    """Renders the main input form page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and makes a prediction."""
    if model is None or scaler is None:
        return "Model or scaler not loaded. Please check the server logs.", 500

    try:
        # --- 1. Get Data from Form ---
        patient_name = request.form.get('patient_name')
        patient_id = request.form.get('patient_id')
        
        form_data = {
            'gender': request.form.get('gender'),
            'age': float(request.form.get('age')),
            'hypertension': int(request.form.get('hypertension')),
            'heart_disease': int(request.form.get('heart_disease')),
            'smoking_history': request.form.get('smoking_history'),
            'bmi': float(request.form.get('bmi')),
            'HbA1c_level': float(request.form.get('hba1c_level')),
            'blood_glucose_level': int(request.form.get('blood_glucose_level'))
        }

        # --- 2. Preprocess the Input Data ---
        input_df = pd.DataFrame([form_data])
        input_df = pd.get_dummies(input_df)

        # This reindex step is crucial. It aligns the input data to the model's expected structure.
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Scale the numerical features
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        # --- 3. Make Prediction ---
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # --- 4. Prepare Results for Display ---
        result = {
            'patient_name': patient_name,
            'patient_id': patient_id,
            'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'has_diabetes': prediction[0] == 1
        }
        
        if result['has_diabetes']:
            result['message'] = "The model predicts that the patient HAS DIABETES."
            result['confidence'] = f"{prediction_proba[0][1]*100:.2f}%"
        else:
            result['message'] = "The model predicts that the patient DOES NOT HAVE DIABETES."
            result['confidence'] = f"{prediction_proba[0][0]*100:.2f}%"
        
        return render_template('result.html', prediction_result=result)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        return render_template('index.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True)
