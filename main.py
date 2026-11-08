import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("--- AI-Powered Drug Safety Risk Assessment ---")
print("Simulating AI's ability to predict potential drug risks before administration.")

# 1. Simulate Data for Drug Safety
# Features represent various factors influencing drug safety:
# [Chemical_Property_A, Patient_Age, Existing_Condition_Score, Drug_Interaction_Score]
# Target: Risk (0 = Low Risk, 1 = High Risk)
# This synthetic data mimics the diverse sources (clinical reports, EHR, etc.) mentioned in the article.
X = np.array([
    [0.5, 30, 0.1, 0.2], # Scenario 1: Low risk profile
    [0.8, 65, 0.7, 0.9], # Scenario 2: High risk profile
    [0.3, 25, 0.0, 0.1], # Scenario 3: Low risk profile
    [0.7, 50, 0.5, 0.6], # Scenario 4: Medium-high risk profile
    [0.4, 40, 0.2, 0.3], # Scenario 5: Low risk profile
    [0.9, 70, 0.8, 0.95],# Scenario 6: Very high risk profile
    [0.2, 20, 0.0, 0.05],# Scenario 7: Low risk profile
    [0.6, 55, 0.6, 0.7], # Scenario 8: High risk profile
    [0.55, 45, 0.3, 0.4],# Scenario 9: Low risk profile
    [0.85, 68, 0.75, 0.88] # Scenario 10: High risk profile
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) # Corresponding risk labels

# 2. Split data into training and testing sets (good practice for ML models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a simple AI model (Logistic Regression)
# This step demonstrates the "learning" aspect of AI, where the model identifies patterns
# in historical data to predict future risks, as described in the article.
print("\nTraining AI model to learn from historical drug safety data...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# (Optional) Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on simulated test data: {accuracy:.2f}")

# 4. Predict risk for new, unseen drug/patient scenarios
# This is the core demonstration of AI catching risks "before the first dose".
# The AI analyzes new input features and predicts a risk level.

# Scenario A: A new drug for a relatively healthy, younger patient
new_drug_patient_data_low_risk = np.array([[0.35, 32, 0.15, 0.25]]) 

# Scenario B: A new drug for an elderly patient with multiple existing conditions
new_drug_patient_data_high_risk = np.array([[0.78, 68, 0.8, 0.9]]) 

predicted_risk_low = model.predict(new_drug_patient_data_low_risk)[0]
predicted_risk_high = model.predict(new_drug_patient_data_high_risk)[0]

risk_labels = {0: "Low Risk", 1: "High Risk"}

print(f"\n--- Predicting Risk for New Scenarios ---")

print(f"\nScenario A: New drug for a relatively healthy, younger patient.")
print(f"   Features: Chemical_Prop_A=0.35, Age=32, Existing_Cond_Score=0.15, Drug_Interaction_Score=0.25")
print(f"   Predicted Risk: {risk_labels[predicted_risk_low]} (AI identifies this as potentially safe)")

print(f"\nScenario B: New drug for an elderly patient with multiple existing conditions.")
print(f"   Features: Chemical_Prop_A=0.78, Age=68, Existing_Cond_Score=0.8, Drug_Interaction_Score=0.9")
print(f"   Predicted Risk: {risk_labels[predicted_risk_high]} (AI flags this as potentially risky)")

print("\nThis example illustrates how an AI model can analyze various factors to proactively identify")
print("potential drug safety risks before a drug is administered, making drug development safer and more efficient.")
