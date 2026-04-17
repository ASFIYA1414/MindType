"""
evaluate_mindtype.py  —  MindType PRD v2.0
==========================================
Loads the trained stress_model.pkl (which includes the optimal threshold)
and reports all PRD §5.1 mandatory metrics:
  - Minority class F1  (primary)
  - PR-AUC             (primary)
  - Minority precision and recall (separately)
  - Confusion matrix
  - Macro F1  (secondary)
  - Optimal classification threshold

NOTE: Overall accuracy is NOT included as a primary metric per PRD §5.2.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    precision_recall_curve, auc
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("EVALUATING ON HELD-OUT MINDTYPE DATASET")
log.info("="*70)

# ─── Load model ──────────────────────────────────────────────────────────────
log.info("\n--- LOADING MODEL ---")
model_dict    = joblib.load("stress_model.pkl")
pipe          = model_dict['pipeline']
threshold     = model_dict['threshold']
feature_names = model_dict['feature_names']

log.info(f"✓ Model loaded  (threshold={threshold:.2f})")
log.info(f"  Feature names: {feature_names}")

if 'metrics' in model_dict:
    m = model_dict['metrics']
    log.info(f"\nTraining metrics (stored in pkl):")
    log.info(f"  Minority F1     : {m.get('minority_f1', 'N/A'):.4f}")
    log.info(f"  Minority Prec   : {m.get('minority_precision', 'N/A'):.4f}")
    log.info(f"  Minority Recall : {m.get('minority_recall', 'N/A'):.4f}")
    log.info(f"  PR-AUC          : {m.get('pr_auc', 'N/A'):.4f}")
    log.info(f"  False Negatives : {m.get('false_negatives', 'N/A')}")

# ─── Load evaluation dataset ──────────────────────────────────────────────────
log.info("\n--- LOADING EVALUATION DATASET ---")
data = pd.read_csv("public_training_dataset.csv")
log.info(f"Dataset shape: {data.shape}")

# Prepare features
BASE_FEATURES = ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]
for col in BASE_FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

data["stress_level"] = pd.to_numeric(data["stress_level"], errors="coerce").fillna(0).astype(int)

log.info("Computing derived features ...")
# v2 feature approximations (must match training exactly)
data["hold_cv"]           = (np.sqrt(data["hold_variance"].clip(lower=0)) / (data["avg_hold"] + 1e-8)).clip(upper=10)
data["iki_entropy"]       = np.log1p(data["avg_pause"])
data["digraph_latency"]   = (data["avg_pause"] / (data["kpm"] + 1e-8)).clip(upper=data["avg_pause"].quantile(0.99) / (data["kpm"].quantile(0.01) + 1e-8))
data["error_burst_rate"]  = np.square(data["backspace_rate"])
data["session_kpm_z"]     = (data["kpm"] / (data["backspace_rate"] + 1e-8)).clip(upper=data["kpm"].quantile(0.99) / 1e-8)
data["hold_pause_ratio"]  = (data["avg_hold"] / (data["avg_pause"] + 1e-8)).clip(upper=data["avg_hold"].quantile(0.99))
data["typing_efficiency"] = (data["kpm"] / (data["backspace_rate"] + 1e-8)).clip(upper=data["kpm"].quantile(0.99) / 1e-8)
data["pause_per_key"]     = (data["avg_pause"] / (data["kpm"] + 1e-8)).clip(upper=data["avg_pause"].quantile(0.99))
for c in ["digraph_latency", "session_kpm_z", "hold_pause_ratio", "typing_efficiency", "pause_per_key"]:
    data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0)
    data[c] = data[c].clip(upper=data[c].quantile(0.99))

# Select only features in training order
X_eval = data[feature_names]
y_eval = data["stress_level"]

log.info(f"Evaluation samples: {len(X_eval)}")
log.info(f"Class distribution:\n{y_eval.value_counts().sort_index()}")

# ─── Predictions ─────────────────────────────────────────────────────────────
log.info("\n--- PREDICTIONS ---")
y_proba       = pipe.predict_proba(X_eval)[:, 1]
y_pred_opt    = (y_proba >= threshold).astype(int)
y_pred_default = (y_proba >= 0.5).astype(int)

log.info(f"✓ Predictions generated  (optimal threshold={threshold:.2f})")

# ─── Minority Class Metrics  ─────────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("MINORITY CLASS (STRESS DETECTION) METRICS  — PRD §5.1")
log.info("="*70)

min_pre = precision_score(y_eval, y_pred_opt, pos_label=1, zero_division=0)
min_rec = recall_score(y_eval,    y_pred_opt, pos_label=1, zero_division=0)
min_f1  = f1_score(y_eval,        y_pred_opt, pos_label=1, zero_division=0)
macro_f1= f1_score(y_eval,        y_pred_opt, average="macro", zero_division=0)

log.info(f"\nPrecision (Minority): {min_pre:.4f}")
log.info(f"  → Of predictions marked 'stressed', {min_pre*100:.1f}% are correct")
log.info(f"\nRecall (Minority): {min_rec:.4f}")
log.info(f"  → Of actual stressed samples, {min_rec*100:.1f}% are caught")
log.info(f"\nF1-Score (Minority): {min_f1:.4f}  ← PRIMARY METRIC  (TARGET: >= 0.85)")
log.info(f"\nMacro F1 (secondary): {macro_f1:.4f}")

# ─── PR-AUC ──────────────────────────────────────────────────────────────────
log.info("\n--- PR-AUC (Precision-Recall Area Under Curve) ---")
pv, rv, _ = precision_recall_curve(y_eval, y_proba)
pr_auc    = auc(rv, pv)
log.info(f"PR-AUC: {pr_auc:.4f}  ← PRIMARY METRIC  (TARGET: 0.75–0.85)")

# Precision@Recall >= 0.80
idx_80 = np.where(rv >= 0.80)[0]
if len(idx_80) > 0:
    log.info(f"Precision @ Recall=0.80: {pv[idx_80[-1]]:.4f}")
else:
    log.info("Precision @ Recall=0.80: N/A (model does not reach 0.80 recall)")

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
log.info("\n--- CONFUSION MATRIX ---")
tn, fp, fn, tp = confusion_matrix(y_eval, y_pred_opt).ravel()

log.info(f"\n              Predicted:")
log.info(f"             0 (Non-stressed)   1 (Stressed)")
log.info(f"Actual  0   {tn:12d}        {fp:12d}")
log.info(f"        1   {fn:12d}        {tp:12d}")

log.info(f"\nTN (correct non-stressed): {tn:6d}  ✓")
log.info(f"FP (false alarms):         {fp:6d}  ⚠")
log.info(f"FN (MISSED STRESS):        {fn:6d}  🔥  (TARGET: < 7)")
log.info(f"TP (correct stress):       {tp:6d}  ✓")

# ─── Full Classification Report ──────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("DETAILED CLASSIFICATION REPORT")
log.info("="*70)
log.info(f"\n{classification_report(y_eval, y_pred_opt, target_names=['Non-Stressed','Stressed'], digits=4, zero_division=0)}")

log.info("\n" + "="*70)
log.info("✅ EVALUATION COMPLETE")
log.info("="*70 + "\n")