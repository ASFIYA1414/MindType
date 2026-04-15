import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


# -------------------------------------
# Load Dataset
# -------------------------------------

data = pd.read_csv("final_desktop_dataset.csv")
# Shuffle (extra safety)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# -------------------------------
# FIX LABELS (CUSTOM MAPPING)
# -------------------------------

# Remove rows where label is missing
data = data.dropna(subset=["stress_level"])

# Convert to numeric (important safety)
data["stress_level"] = pd.to_numeric(data["stress_level"], errors="coerce")

# Drop again if any conversion created NaNs
data = data.dropna(subset=["stress_level"])

# Apply mapping
data["stress_level"] = data["stress_level"].apply(
    lambda x: 0 if x in [0, 1] else 1
)

print("\nFixed Class Distribution:\n", data["stress_level"].value_counts())

print("Dataset shape:", data.shape)
print("\nClass distribution:\n", data["stress_level"].value_counts())


# -------------------------------------
# FEATURE ENGINEERING
# -------------------------------------

data["hold_pause_ratio"] = data["avg_hold"] / (data["avg_pause"] + 1e-6)
data["typing_efficiency"] = data["kpm"] / (data["backspace_rate"] + 1e-6)
data["pause_per_key"] = data["avg_pause"] / (data["kpm"] + 1e-6)


# -------------------------------------
# Features and Labels
# -------------------------------------

X = data[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate",
    "hold_pause_ratio",      # NEW
    "typing_efficiency",     # NEW
    "pause_per_key"          # NEW
]]

y = data["stress_level"]


# -------------------------------------
# Cross Validation Setup
# -------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Correct F1 for binary classification
f1 = make_scorer(f1_score)


# -------------------------------------
# Models (IMPROVED 🔥)
# -------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "SVM (RBF)": SVC(
        kernel="rbf",
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(),

    "XGBoost (BEST)": XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=5,   # handles imbalance
        random_state=42,
        eval_metric="logloss"
    )
}


# -------------------------------------
# Evaluate Models
# -------------------------------------

results = []

for name, model in models.items():

    accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    f1_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=f1
    )

    results.append({
        "Model": name,
        "Accuracy Mean": accuracy_scores.mean(),
        "Accuracy Std": accuracy_scores.std(),
        "F1 Mean": f1_scores.mean(),
        "F1 Std": f1_scores.std()
    })


# -------------------------------------
# Show Results
# -------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison (5-Fold CV):\n")

print(results_df.sort_values(by="Accuracy Mean", ascending=False))
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train best model (XGBoost)
best_model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=30,
    random_state=42,
    eval_metric="logloss"
)

best_model.fit(X_train, y_train)

# Predict
y_pred = best_model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))