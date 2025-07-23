## Complete Code with Interactive Prediction

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# Load the dataset
try:
    df = pd.read_csv('diabetes_prediction_dataset.csv')
except FileNotFoundError:
    print("Error: 'diabetes_prediction_dataset.csv' not found.")
    print("Please download the dataset and place it in the same directory as the script.")
    exit()

# ==============================================================================
# DATASET OVERVIEW & PREPARATION
# ==============================================================================
print("="*50)
print("DATASET OVERVIEW & PREPARATION")
print("="*50)

# Remove duplicates and 'Other' gender category
df = df.drop_duplicates()
df = df[df['gender'] != 'Other']
print(f"Dataset shape after cleaning: {df.shape}")

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

df_processed = df.copy()
df_processed = pd.get_dummies(df_processed, columns=['gender', 'smoking_history'], drop_first=True)
print("Categorical encoding completed.")

# ==============================================================================
# TRAIN-TEST SPLIT AND SCALING
# ==============================================================================
print("\n" + "="*50)
print("TRAIN-TEST SPLIT AND SCALING")
print("="*50)

X = df_processed.drop('diabetes', axis=1)
y = df_processed['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

joblib.dump(scaler, 'scaler.joblib')
print("\nFeature scaling completed and scaler saved as 'scaler.joblib'.")

# ==============================================================================
# MODEL TRAINING, EVALUATION, AND SAVING
# ==============================================================================
print("\n" + "="*50)
print("MODEL TRAINING, EVALUATION, AND SAVING")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    model_filename = f'{name.replace(" ", "_").lower()}_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    results[name] = {'accuracy': accuracy, 'auc': auc_score}

    print(f"{name} -> Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    print(f"\nClassification Report for {name}:\n{classification_report(y_test, y_pred)}")

results_df = pd.DataFrame(results).T.sort_values(by='auc', ascending=False)
print("\nModel Performance Summary:")
print(results_df)

