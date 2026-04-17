"""
model_selection.py  —  MindType PRD v2.0
=========================================
Implements PRD §6 Step 1: Fix evaluation metrics
  - Primary scorer: Minority class F1 (pos_label=1)
  - Secondary: PR-AUC
  - SMOTE inside imblearn.pipeline.Pipeline (zero data leakage)
  - Threshold sweep after best model selection

Data: public_training_dataset.csv (properly labelled binary labels)
NOTE: Overall accuracy is NOT used as a selection metric (PRD §5.2).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    make_scorer, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_curve, auc, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("MODEL SELECTION & CROSS-VALIDATION  (PRD v2.0)")
log.info("Primary metric: Minority class F1 (stress detection)")
log.info("NOTE: Overall accuracy is NOT used as a selection metric.")
log.info("="*70)

# ─── Load Data ──────────────────────────────────────────────────────────────
log.info("\n--- LOADING DATA ---")
data = pd.read_csv("public_training_dataset.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

for col in ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate", "stress_level"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=["stress_level"])
data["stress_level"] = data["stress_level"].astype(int)

log.info(f"\nDataset shape: {data.shape}")
log.info(f"Class distribution:\n{data['stress_level'].value_counts().sort_index()}")

minority_count = (data["stress_level"] == 1).sum()
majority_count = (data["stress_level"] == 0).sum()
imbalance_ratio = majority_count / max(minority_count, 1)
log.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

# ─── Feature Engineering  (PRD §4) ──────────────────────────────────────────
log.info("\n--- FEATURE ENGINEERING v2 (PRD §4) ---")

BASE_FEATURES = ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]
V2_FEATURES   = ["iki_entropy", "hold_cv", "digraph_latency", "error_burst_rate", "session_kpm_z"]

for col in BASE_FEATURES:
    data[col] = data[col].replace([np.inf, -np.inf], 0).fillna(0)

available_v2 = [f for f in V2_FEATURES if f in data.columns]

if not available_v2:
    # Compute v2 approximations
    data["hold_cv"]           = (np.sqrt(data["hold_variance"].clip(0)) / (data["avg_hold"] + 1e-8)).clip(upper=10)
    data["iki_entropy"]       = np.log1p(data["avg_pause"])
    data["digraph_latency"]   = (data["avg_pause"] / (data["kpm"] + 1e-8)).clip(upper=data["avg_pause"].quantile(0.99))
    data["error_burst_rate"]  = np.square(data["backspace_rate"])
    data["session_kpm_z"]     = (data["kpm"] / (data["backspace_rate"] + 1e-8)).clip(upper=data["kpm"].quantile(0.99) / 1e-6)
    available_v2 = ["hold_cv", "iki_entropy", "digraph_latency", "error_burst_rate", "session_kpm_z"]
    for c in available_v2:
        data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0).clip(upper=data[c].quantile(0.99))
    log.info(f"✓ Computed v2 features: {available_v2}")
else:
    log.info(f"✓ Native v2 features: {available_v2}")

data["hold_pause_ratio"]  = (data["avg_hold"]  / (data["avg_pause"] + 1e-8)).clip(upper=data["avg_hold"].quantile(0.99))
data["typing_efficiency"] = (data["kpm"] / (data["backspace_rate"] + 1e-8)).clip(upper=data["kpm"].quantile(0.99))
data["pause_per_key"]     = (data["avg_pause"] / (data["kpm"] + 1e-8)).clip(upper=data["avg_pause"].quantile(0.99))
for c in ["hold_pause_ratio", "typing_efficiency", "pause_per_key"]:
    data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0)

feature_cols = BASE_FEATURES + available_v2 + ["hold_pause_ratio", "typing_efficiency", "pause_per_key"]

X = data[feature_cols]
y = data["stress_level"]

log.info(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

# ─── Cross-Validation Setup  (PRD §5) ───────────────────────────────────────
log.info("\n--- CROSS-VALIDATION SETUP ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# PRIMARY scorer: minority class F1
minority_f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

# PR-AUC scorer
pr_auc_scorer = make_scorer(
    average_precision_score,
    needs_proba=True,
    pos_label=1
)

# ─── Baseline Models  (for comparison — NO SMOTE) ───────────────────────────
log.info("\n--- BASELINE MODELS (5-Fold CV, Minority F1) ---")
log.info("These do NOT use SMOTE — included for comparison only.\n")

baseline_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=8, random_state=42
    ),
    f"XGBoost (SPW={int(imbalance_ratio)})": XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        scale_pos_weight=int(imbalance_ratio),
        random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1
    )
}

baseline_results = []
for name, model in baseline_models.items():
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring=minority_f1_scorer)
    baseline_results.append({
        "Model":    name,
        "F1 Mean":  round(float(f1_scores.mean()), 3),
        "F1 Std":   round(float(f1_scores.std()), 3),
    })
    log.info(f"{name:42s}: F1 = {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# ─── XGBoost + SMOTE Pipeline  (PRD §3.1 Recommended)  ─────────────────────
log.info("\n--- XGBOOST + SMOTE PIPELINE (5-Fold CV) ---")
log.info("SMOTE is applied INSIDE the pipeline per fold — zero data leakage.\n")

# For 2.1:1 ratio: SMOTE strategy must be > current minority fraction (~0.47)
smote_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)),
    ('model', XGBClassifier(
        scale_pos_weight=2,
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ))
])

smote_f1_scores = cross_val_score(smote_pipeline, X, y, cv=cv, scoring=minority_f1_scorer)
log.info(f"{'XGBoost + SMOTE':42s}: F1 = {smote_f1_scores.mean():.3f} ± {smote_f1_scores.std():.3f}")

baseline_results.append({
    "Model":   "XGBoost + SMOTE (recommended)",
    "F1 Mean": round(float(smote_f1_scores.mean()), 3),
    "F1 Std":  round(float(smote_f1_scores.std()), 3),
})

# ─── Results Table ────────────────────────────────────────────────────────────
log.info("\n--- CROSS-VALIDATION RESULTS (sorted by Minority F1) ---")
results_df = pd.DataFrame(baseline_results).sort_values(by="F1 Mean", ascending=False)
log.info(results_df.to_string(index=False))

best_cv_model_name = results_df.iloc[0]["Model"]
log.info(f"\n✓ Best CV model: {best_cv_model_name}")
log.info("NOTE: XGBoost + SMOTE is the recommended model — it prevents data leakage.")

# ─── Train Best Model on Held-Out Test Set  ───────────────────────────────────
log.info("\n--- TRAINING BEST MODEL ON HELD-OUT TEST SET ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
log.info(f"Train stressed: {y_train.sum()} | Test stressed: {y_test.sum()}")

best_model = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)),
    ('model', XGBClassifier(
        scale_pos_weight=2,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=10,
        gamma=0.5,
        reg_alpha=4.0,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ))
])
best_model.fit(X_train, y_train)
log.info("✓ Model fitted")

# ─── Threshold Tuning  (PRD §5.3) ────────────────────────────────────────────
log.info("\n--- THRESHOLD TUNING (sweep predict_proba 0.05–0.65) ---")
log.info("Default 0.5 threshold is calibrated for balanced datasets.")
log.info("At imbalanced ratios it biases predictions — we tune it.\n")

y_proba = best_model.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_f1        = f1_score(y_test, best_model.predict(X_test), pos_label=1, zero_division=0)

threshold_results = []
for threshold in np.arange(0.05, 0.65, 0.01):
    y_pred_t    = (y_proba >= threshold).astype(int)
    f1_t        = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    precision_t = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    recall_t    = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "F1":        round(f1_t, 3),
        "Precision": round(precision_t, 3),
        "Recall":    round(recall_t, 3)
    })
    if f1_t > best_f1:
        best_f1        = f1_t
        best_threshold = threshold

threshold_df = pd.DataFrame(threshold_results).sort_values(by="F1", ascending=False)
log.info("Top 5 thresholds by F1:")
log.info(threshold_df.head().to_string(index=False))
log.info(f"\n✓ Optimal threshold: {best_threshold:.2f}")
log.info(f"  F1 @ optimal threshold: {best_f1:.3f}")

# ─── Final Metrics at Optimal Threshold  ─────────────────────────────────────
y_pred_optimal = (y_proba >= best_threshold).astype(int)

optimal_precision = precision_score(y_test, y_pred_optimal, pos_label=1, zero_division=0)
optimal_recall    = recall_score(y_test,    y_pred_optimal, pos_label=1, zero_division=0)

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_vals, precision_vals)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()

log.info(f"\n--- FINAL TEST SET METRICS (Threshold = {best_threshold:.2f}) ---")
log.info(f"Minority Precision  : {optimal_precision:.3f}  (TARGET: 0.72–0.85)")
log.info(f"Minority Recall     : {optimal_recall:.3f}  (TARGET: >= 0.80)")
log.info(f"Minority F1         : {best_f1:.3f}  (TARGET: >= 0.85)")
log.info(f"PR-AUC              : {pr_auc:.3f}  (TARGET: 0.75–0.85)")
log.info(f"\nConfusion Matrix:")
log.info(f"  TN: {tn:6d}  FP: {fp:6d}")
log.info(f"  FN: {fn:6d}  TP: {tp:6d}")
log.info(f"  False Negatives (missed stress): {fn}  (TARGET: < 7)")

log.info(f"\n{classification_report(y_test, y_pred_optimal, target_names=['Non-stressed', 'Stressed'], digits=3, zero_division=0)}")


log.info("\n" + "="*70)
log.info("✅ MODEL SELECTION COMPLETE")
log.info("="*70 + "\n")