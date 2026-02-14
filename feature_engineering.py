import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Remove rows with NaN values (first delta row usually has NaN)
df = df.dropna()

# Define features (X) and label (y)
X = df[[
    "avg_hold",
    "avg_pause",
    "backspace_rate",
    "kpm",
    "delta_kpm",
    "delta_pause",
    "delta_backspace"
]]

y = df["stress_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
