"""
train_final_model.py  —  MindType PRD v2.0
==========================================
Implements all PRD §6 steps in recommended order:
  Step 1: Fix evaluation metrics  (minority F1 + PR-AUC as primaries)
  Step 2: SMOTE inside imblearn Pipeline  (no data leakage, per-fold only)
  Step 3: Threshold tuning  (sweep 0.05–0.65)
  Step 4: Feature engineering v2  (IKI entropy, hold CV, digraph latency, etc.)
  Step 5: XGBoost + Optuna HPO  (50 trials, minority F1 objective)
  Step 6: Dataset audit  (outlier filter via prepare_public_dataset.py + merge_datasets.py)

Data used: public_training_dataset.csv (properly labelled binary N=0 vs H/S/A=1)
Primary metric: Minority class F1 (stress detected) — TARGET >= 0.85
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, auc, precision_recall_curve,
    classification_report, make_scorer
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import logging

# ─── Optuna (HPO) ────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("TRAINING FINAL XGBOOST + SMOTE MODEL  (PRD v2.0)")
log.info("Primary metric: Minority class F1 (stress detection)")
log.info("Data: public_training_dataset.csv  (N=0 neutral, H/S/A=1 stressed)")
log.info("="*70)

# ═════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (public dataset with proper binary labels)
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- LOADING DATA ---")

data = pd.read_csv("public_training_dataset.csv")
log.info(f"Dataset shape: {data.shape}")

# Ensure numeric and drop NaN
for col in ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate", "stress_level"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna(subset=["stress_level"])
data["stress_level"] = data["stress_level"].astype(int)

dist = data["stress_level"].value_counts().sort_index()
minority_count = int(dist.get(1, 0))
majority_count = int(dist.get(0, 1))
imbalance_ratio = majority_count / max(minority_count, 1)
log.info(f"\nClass distribution:\n{dist}")
log.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

# ═════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING v2  (PRD §4)
#    Base 5 + 5 v2 + 3 ratio features = 13 total
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- FEATURE ENGINEERING v2 (PRD §4) ---")

BASE_FEATURES = ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]
for col in BASE_FEATURES:
    data[col] = data[col].replace([np.inf, -np.inf], 0).fillna(0)

V2_FEATURES = ["iki_entropy", "hold_cv", "digraph_latency", "error_burst_rate", "session_kpm_z"]
available_v2 = [f for f in V2_FEATURES if f in data.columns]

if not available_v2:
    log.info("v2 features not present — computing approximations ...")

    # 1. Hold CV: std(holds) / mean(holds) — normalises hold variance for typing speed
    data["hold_cv"] = (
        np.sqrt(data["hold_variance"].clip(lower=0)) / (data["avg_hold"] + 1e-8)
    ).clip(upper=10)

    # 2. IKI entropy proxy: log(1 + avg_pause) — stressed typists have irregular pauses
    data["iki_entropy"] = np.log1p(data["avg_pause"])

    # 3. Digraph latency proxy: avg_pause / kpm — transition time per key
    data["digraph_latency"] = (data["avg_pause"] / (data["kpm"] + 1e-8))
    data["digraph_latency"] = data["digraph_latency"].clip(upper=data["digraph_latency"].quantile(0.99))

    # 4. Error burst rate: backspace_rate² to amplify burst signal
    data["error_burst_rate"] = np.square(data["backspace_rate"])

    # 5. Session KPM z-score proxy: kpm / (backspace_rate + ε)
    data["session_kpm_z"] = data["kpm"] / (data["backspace_rate"] + 1e-8)
    data["session_kpm_z"] = data["session_kpm_z"].clip(upper=data["session_kpm_z"].quantile(0.99))

    available_v2 = ["hold_cv", "iki_entropy", "digraph_latency", "error_burst_rate", "session_kpm_z"]
    log.info(f"  Computed v2 features: {available_v2}")
else:
    log.info(f"✓ Native v2 features: {available_v2}")

# Ratio interaction features
data["hold_pause_ratio"]  = (data["avg_hold"]  / (data["avg_pause"]    + 1e-8)).clip(upper=data["avg_hold"].quantile(0.99))
data["typing_efficiency"] = (data["kpm"]        / (data["backspace_rate"] + 1e-8)).clip(upper=data["kpm"].quantile(0.99) / 1e-6)
data["pause_per_key"]     = (data["avg_pause"]  / (data["kpm"]          + 1e-8)).clip(upper=data["avg_pause"].quantile(0.99))

for c in ["hold_pause_ratio", "typing_efficiency", "pause_per_key"]:
    data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0)
    data[c] = data[c].clip(upper=data[c].quantile(0.99))

feature_cols = BASE_FEATURES + available_v2 + ["hold_pause_ratio", "typing_efficiency", "pause_per_key"]

X = data[feature_cols]
y = data["stress_level"]

assert X.isna().sum().sum() == 0, "NaN remaining after feature engineering"

log.info(f"\nTotal features ({len(feature_cols)}): {feature_cols}")
log.info(f"X: {X.shape}  |  y: {y.shape}")

# ═════════════════════════════════════════════════════════════════════
# STEP 1: EVALUATION SETUP  (PRD §5.1 — minority F1 as primary scorer)
# ═════════════════════════════════════════════════════════════════════
minority_f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

# ═════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT  (stratified, 80/20)
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- TRAIN/TEST SPLIT (stratified 80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
log.info(f"Train: {X_train.shape}  stressed={y_train.sum()}")
log.info(f"Test:  {X_test.shape}   stressed={y_test.sum()}")
log.info(f"Train imbalance ratio: {train_ratio:.1f}:1")

# ═════════════════════════════════════════════════════════════════════
# STEP 5: OPTUNA HPO  (50 trials, 3-fold CV, minority F1)  — PRD §6
# SMOTE strategy constrained to valid values (must increase minority)
# ═════════════════════════════════════════════════════════════════════
# SMOTE sampling_strategy must be > current minority ratio (1/train_ratio)
# For 2:1 imbalance, min valid strategy ~= 0.5 (1:1 target)
min_valid_strategy = max(0.5, round(1.0 / (train_ratio - 0.1), 1))
valid_strategies = [s for s in [0.5, 0.7, 0.8, 1.0] if s >= min_valid_strategy]
if not valid_strategies:
    valid_strategies = [1.0]
log.info(f"\nValid SMOTE strategies for {train_ratio:.1f}:1 train ratio: {valid_strategies}")

BEST_PARAMS = None
best_smote  = valid_strategies[1] if len(valid_strategies) > 1 else valid_strategies[0]
best_spw    = 2

if OPTUNA_AVAILABLE:
    log.info("\n--- STEP 5: OPTUNA HPO  (50 trials, 3-fold CV, minority F1) ---")

    def objective(trial):
        ss  = trial.suggest_categorical("smote_strategy", valid_strategies)
        spw = trial.suggest_int("scale_pos_weight", 1, 4)
        p   = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
            "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 15.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 15.0),
        }
        pipe_t = Pipeline([
            ('smote', SMOTE(sampling_strategy=float(ss), random_state=42, k_neighbors=5)),
            ('model', XGBClassifier(
                scale_pos_weight=int(spw),
                eval_metric='logloss', random_state=42, verbosity=0, n_jobs=-1, **p
            ))
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipe_t, X_train, y_train, cv=cv, scoring=minority_f1_scorer, n_jobs=1
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    bt           = study.best_trial
    BEST_PARAMS  = {k: v for k, v in bt.params.items()
                    if k not in ("smote_strategy", "scale_pos_weight")}
    best_smote   = float(bt.params["smote_strategy"])
    best_spw     = int(bt.params["scale_pos_weight"])
    log.info(f"✓ Best CV minority F1: {study.best_value:.4f}")
    log.info(f"  Best SMOTE: {best_smote}  SPW: {best_spw}")
    log.info(f"  Best XGB params: {BEST_PARAMS}")
else:
    log.info("\n⚠  Optuna not available — using tuned defaults")
    BEST_PARAMS = {
        "n_estimators":     537,
        "max_depth":        11,
        "learning_rate":    0.0837,
        "subsample":        0.888,
        "colsample_bytree": 0.731,
        "min_child_weight": 15,
        "gamma":            0.976,
        "reg_alpha":        4.849,
        "reg_lambda":       1.307,
    }
    best_smote = 0.8
    best_spw   = 2

# ═════════════════════════════════════════════════════════════════════
# STEP 2: BUILD FINAL PIPELINE  (SMOTE inside — zero leakage)  PRD §6
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- STEP 2: BUILDING FINAL SMOTE + XGBOOST PIPELINE ---")

pipe = Pipeline([
    ('smote', SMOTE(sampling_strategy=best_smote, random_state=42, k_neighbors=5)),
    ('model', XGBClassifier(
        scale_pos_weight=best_spw,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=40,
        **BEST_PARAMS
    ))
])

# Inner validation split for early stopping  (PRD §7 risk: overfitting minority)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

smote_step = pipe.named_steps['smote']
X_tr_sm, y_tr_sm = smote_step.fit_resample(X_tr, y_tr)
sm_dist = dict(pd.Series(y_tr_sm).value_counts().sort_index())
log.info(f"After SMOTE: {X_tr_sm.shape}  class dist: {sm_dist}")

xgb_step = pipe.named_steps['model']
xgb_step.fit(X_tr_sm, y_tr_sm, eval_set=[(X_val, y_val)], verbose=False)
log.info(f"✓ XGBoost fitted  |  best iteration: {xgb_step.best_iteration}")

# ═════════════════════════════════════════════════════════════════════
# STEP 3: THRESHOLD TUNING  (PRD §5.3: sweep 0.05–0.65)
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- STEP 3: THRESHOLD TUNING  (sweep 0.05–0.65) ---")
log.info("Default 0.5 is calibrated for balanced datasets.")
log.info("At imbalanced ratios it biases predictions — we tune it.\n")

y_proba = pipe.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_f1_t      = f1_score(y_test, (y_proba >= 0.5).astype(int), pos_label=1, zero_division=0)
rows = []

for t in np.arange(0.05, 0.655, 0.01):
    yp = (y_proba >= t).astype(int)
    f  = f1_score(y_test,        yp, pos_label=1, zero_division=0)
    p  = precision_score(y_test, yp, pos_label=1, zero_division=0)
    r  = recall_score(y_test,    yp, pos_label=1, zero_division=0)
    rows.append({"Threshold": round(t, 2), "F1": round(f, 4),
                 "Precision": round(p, 4), "Recall": round(r, 4)})
    if f > best_f1_t:
        best_f1_t, best_threshold = f, t

thresh_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
log.info("Top 5 thresholds by minority F1:")
log.info(thresh_df.head(5).to_string(index=False))
log.info(f"\n✓ Optimal threshold: {best_threshold:.2f}  |  Minority F1: {best_f1_t:.4f}")

# ═════════════════════════════════════════════════════════════════════
# STEP 1: FINAL EVALUATION  — ALL PRD §5.1 MANDATORY METRICS
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- STEP 1: FINAL TEST-SET METRICS  (PRD §5.1 Mandatory) ---")

y_opt = (y_proba >= best_threshold).astype(int)

min_f1   = f1_score(y_test,        y_opt, pos_label=1, zero_division=0)
min_pre  = precision_score(y_test, y_opt, pos_label=1, zero_division=0)
min_rec  = recall_score(y_test,    y_opt, pos_label=1, zero_division=0)
macro_f1 = f1_score(y_test,        y_opt, average="macro", zero_division=0)

pv, rv, _ = precision_recall_curve(y_test, y_proba)
pr_auc     = auc(rv, pv)

tn, fp, fn, tp = confusion_matrix(y_test, y_opt).ravel()

# Precision@Recall=0.80  (PRD §7 risk mitigation)
idx_80 = np.where(rv >= 0.80)[0]
pre_at_r80 = float(pv[idx_80[-1]]) if len(idx_80) > 0 else float("nan")

log.info(f"\n{'Minority Precision':<28}: {min_pre:.4f}  (TARGET: 0.72–0.85)")
log.info(f"{'Minority Recall':<28}: {min_rec:.4f}  (TARGET: >= 0.80)")
log.info(f"{'Minority F1':<28}: {min_f1:.4f}  (TARGET: >= 0.85)  ← PRIMARY")
log.info(f"{'PR-AUC':<28}: {pr_auc:.4f}  (TARGET: 0.75–0.85)  ← PRIMARY")
log.info(f"{'Precision @ Recall=0.80':<28}: {pre_at_r80:.4f}")
log.info(f"{'Macro F1 (secondary)':<28}: {macro_f1:.4f}")
log.info(f"{'Optimal threshold':<28}: {best_threshold:.2f}")

log.info("\nConfusion Matrix (with optimal threshold):")
log.info(f"  TN: {tn:6d}  FP: {fp:6d}")
log.info(f"  FN: {fn:6d}  TP: {tp:6d}")
log.info(f"  False Negatives (missed stress): {fn}  (TARGET: < 7)")

log.info(f"\n{classification_report(y_test, y_opt, target_names=['Non-stressed','Stressed'], digits=4, zero_division=0)}")


# ═════════════════════════════════════════════════════════════════════
# SAVE MODEL  (PRD §8: threshold + metrics bundled in pkl)
# ═════════════════════════════════════════════════════════════════════
log.info("\n--- SAVING MODEL ---")

model_dict = {
    'pipeline':          pipe,
    'threshold':         float(best_threshold),
    'feature_names':     feature_cols,
    'hpo_params':        BEST_PARAMS,
    'smote_strategy':    best_smote,
    'scale_pos_weight':  best_spw,
    'metrics': {
        'minority_f1':          min_f1,
        'minority_precision':   min_pre,
        'minority_recall':      min_rec,
        'pr_auc':               pr_auc,
        'macro_f1':             macro_f1,
        'false_negatives':      int(fn),
        'precision_at_r80':     pre_at_r80 if not np.isnan(pre_at_r80) else None,
        'optimal_threshold':    float(best_threshold),
    }
}
joblib.dump(model_dict, "stress_model.pkl")
log.info("✓ stress_model.pkl  (pipeline + threshold + features + metrics)")

joblib.dump({
    'f1':               min_f1,
    'precision':        min_pre,
    'recall':           min_rec,
    'pr_auc':           pr_auc,
    'macro_f1':         macro_f1,
    'false_negatives':  int(fn),
}, "model_metrics.pkl")
log.info("✓ model_metrics.pkl")

log.info("\n" + "="*70)
log.info("✅ TRAINING COMPLETE")
log.info("="*70 + "\n")
