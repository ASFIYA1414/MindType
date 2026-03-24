import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------
# Load trained model
# -------------------------------------

model = joblib.load("stress_model.pkl")

print("Model loaded successfully.")


# -------------------------------------
# Load MindType dataset
# -------------------------------------

data = pd.read_csv("mindtype_dataset.csv")

print("MindType dataset shape:", data.shape)


# -------------------------------------
# Convert stress labels (1–5 → binary)
# -------------------------------------

data["stress_level"] = data["stress_level"].apply(
    lambda x: 1 if x >= 4 else 0
)


# -------------------------------------
# Select features
# -------------------------------------

X_test = data[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate"
]]

y_test = data["stress_level"]


print("Testing samples:", len(X_test))


# -------------------------------------
# Predict
# -------------------------------------

predictions = model.predict(X_test)


# -------------------------------------
# Evaluation
# -------------------------------------

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))