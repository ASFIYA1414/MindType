import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("public_training_dataset.csv")

X = data[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate"
]]

y = data["stress_level"]

print("Training samples:", len(X))

# Train final model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Final model trained.")

# Save model
joblib.dump(model, "stress_model.pkl")

print("Model saved as stress_model.pkl")