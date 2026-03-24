import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


# -------------------------------------
# Load Dataset
# -------------------------------------

data = pd.read_csv("public_training_dataset.csv")

print("Dataset shape:", data.shape)


# -------------------------------------
# Features and Labels
# -------------------------------------

X = data[[
    "avg_hold",
    "hold_variance",
    "avg_pause",
    "kpm",
    "backspace_rate"
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

f1 = make_scorer(f1_score)


# -------------------------------------
# Models
# -------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM (RBF)": SVC(kernel="rbf"),
    "Gradient Boosting": GradientBoostingClassifier()
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

print(results_df.sort_values(by="F1 Mean", ascending=False))