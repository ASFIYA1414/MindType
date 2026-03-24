import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------
# Load training dataset (public)
# -------------------------------------

train_data = pd.read_csv("public_training_dataset.csv")

X_train = train_data.drop("stress_level", axis=1)
y_train = train_data["stress_level"]

print("Training samples:", len(X_train))


# -------------------------------------
# Train model
# -------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


# -------------------------------------
# Load MindType dataset (your dataset)
# -------------------------------------

mindtype = pd.read_csv("mindtype_dataset.csv")

# Convert 5-level stress to binary
mindtype["stress_level"] = mindtype["stress_level"].apply(
    lambda x: 1 if x >= 4 else 0
)

# Use only the same features as training
X_test = mindtype[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate"
]]

y_test = mindtype["stress_level"]

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