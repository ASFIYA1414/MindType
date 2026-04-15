import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------------
# LOAD DATA
# -------------------------------------
data = pd.read_csv("final_desktop_dataset.csv")

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", data.shape)

# -------------------------------------
# FIX LABELS (IMPORTANT)
# -------------------------------------

data = data.dropna(subset=["stress_level"])
data["stress_level"] = pd.to_numeric(data["stress_level"], errors="coerce")
data = data.dropna(subset=["stress_level"])

# Mapping: 0,1,2 → 0 (Calm), 3,4,5 → 1 (Stress)
data["stress_level"] = data["stress_level"].apply(
    lambda x: 0 if x in [0, 1, 2] else 1
)

print("\nClass distribution:\n", data["stress_level"].value_counts())

# -------------------------------------
# HANDLE NaNs
# -------------------------------------
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(data.median(numeric_only=True))

# -------------------------------------
# FEATURE ENGINEERING
# -------------------------------------
data["hold_pause_ratio"] = data["avg_hold"] / (data["avg_pause"] + 1e-6)
data["typing_efficiency"] = data["kpm"] / (data["backspace_rate"] + 1e-6)
data["pause_per_key"] = data["avg_pause"] / (data["kpm"] + 1e-6)

# -------------------------------------
# FEATURES & LABEL
# -------------------------------------
X = data[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate",
    "hold_pause_ratio",
    "typing_efficiency",
    "pause_per_key"
]]

y = data["stress_level"]

# -------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# -------------------------------------
# MODEL (BEST ONE)
# -------------------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=30,
    random_state=42,
    eval_metric="logloss"
)

# -------------------------------------
# TRAIN
# -------------------------------------
model.fit(X_train, y_train)

# -------------------------------------
# TEST
# -------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------
# RESULTS
# -------------------------------------
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))