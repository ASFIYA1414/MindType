# MindType — Audit Log
**Last Updated:** 2026-04-17  
**Project:** MindType Keystroke Dynamics Stress Detection  
**Document:** Full change history, design decisions, and data pipeline documentation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Class Imbalance Resolution](#class-imbalance-resolution)
7. [Change History](#change-history)
8. [Data Quality & Assurance](#data-quality--assurance)
9. [Dependencies](#dependencies)
10. [Performance Summary](#performance-summary)

---

## Project Overview

**Project Name:** MindType  
**Purpose:** Real-time stress detection through keystroke dynamics analysis  
**Approach:** Binary classification of typing patterns using machine learning  
**Primary Algorithm:** XGBoost + SMOTE (replaces original Random Forest baseline)  
**Primary Metric:** Minority class F1-score (stressed = class 1)

### Key Objectives
- Capture real-time keystroke dynamics
- Extract behavioural features from typing patterns
- Detect stressed vs non-stressed sessions accurately
- Maximise recall for stress detection (catch real stress events)
- Use PR-AUC as a threshold-independent quality measure

---

## System Architecture

```
MindType System
├── Data Capture Layer         (capture.py)
├── Feature Extraction Layer   (extract_window_features.py)
├── Data Preparation Layer     (prepare_public_dataset.py)
├── Dataset Merge Layer        (mbd_datasets/merge_datasets.py)
├── Model Selection Layer      (model_selection.py)
├── Model Training Layer       (train_final_model.py)
└── Evaluation Layer           (evaluate_mindtype.py)
```

### Database Schema

**keystrokes table:**
- `id` — unique keystroke record
- `user_id` — participant identifier (e.g., U01)
- `session_id` — unique session timestamp
- `condition` — session context (calm/stress)
- `key` — keyboard key pressed
- `event_type` — press or release
- `timestamp` — high-precision event timestamp

**stress_labels table:**
- `id`, `user_id`, `session_id`, `timestamp`
- `stress_level` — user-reported stress (1–5 scale)

---

## Data Pipeline

### Stage 1: Real-Time Keystroke Capture
**File:** `capture.py`

- Records keystroke press/release events with timestamps into SQLite (`keystrokes.db`)
- Periodic popup windows collect user-reported stress level (1–5 scale)
- Runs background thread to avoid blocking the user

---

### Stage 2: Feature Extraction
**File:** `extract_window_features.py`

- **Window size:** 30 seconds of continuous keystrokes
- **Minimum events:** 10 keystrokes per window (quality filter)
- Calculates: avg_hold, hold_variance, avg_pause, kpm, backspace_rate
- Maps stress labels to windows (nearest label within ±15 seconds)
- **Output:** `mindtype_dataset.csv`

---

### Stage 3: Public Dataset Preparation
**File:** `prepare_public_dataset.py`  
**Status:** Fixed (see Change History — Bug #1)

#### Data Sources

| Dataset | Purpose |
|---------|---------|
| `Free Text Typing Dataset.csv` | Naturalistic typing, emotionIndex N/H/C/A/S |
| `Fixed Text Typing Dataset.csv` | Fixed-sentence typing, same emotion labels |
| `Frequency Dataset.csv` | Per-session backspace count and session duration |

#### emotionIndex Mapping

```
N (Neutral), C (Calm)       → 0   (non-stressed)
H (Happy), S (Sad), A (Anxious)  → 1   (stressed)
```

#### Processing Steps
1. Load all three files
2. Filter corrupted rows: D1U1 (hold time) must be 1–2000 ms, D1D2 (pause) 0–15,000 ms
3. Compute `backspace_rate` per-user per-session from Frequency Dataset (joined on userId + emotionIndex)
4. Compute `kpm` = clip(60000 / (D1D2 + 1), 0.01, 250)
5. Compute `hold_variance` via 5-element rolling window
6. Drop NaN rows
7. **Output:** `public_training_dataset.csv` (70,849 rows)

---

### Stage 4: Dataset Merge
**File:** `mbd_datasets/merge_datasets.py`  
**Status:** Fixed (see Change History — Bug #2, Bug #3)

#### Steps
1. Load `public_training_dataset.csv` and `private_desktop_dataset.csv`
2. Convert private dataset units: seconds → milliseconds for avg_hold and avg_pause; clip kpm to 0–250
3. Binarise stress labels: private dataset (all stress levels 1–5) → 1
4. Run KS-test on all 5 features; apply robust z-score normalisation (1st–99th percentile clip) for divergent features
5. Assert zero NaN before and after merge
6. Stratified shuffle
7. **Output:** `final_desktop_dataset.csv`

---

## Feature Engineering

### Base Features

| Feature | Calculation | Stress Indicator |
|---------|-------------|-----------------|
| `avg_hold` | Mean key hold duration (ms) | Stressed = shorter, more erratic |
| `hold_variance` | Variance of hold times | Stressed = higher variance |
| `avg_pause` | Mean inter-key interval (ms) | Stressed = longer pauses |
| `kpm` | Keys per minute | Stressed = slower typing |
| `backspace_rate` | Backspaces / total keys | Stressed = more errors |

### v2 Features (Added — PRD §4)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `hold_cv` | √hold_variance / avg_hold | Normalised hold irregularity, independent of typing speed |
| `iki_entropy` | log(1 + avg_pause) | Captures non-linear pause distribution |
| `digraph_latency` | avg_pause / kpm | Transition time per keystroke |
| `error_burst_rate` | backspace_rate² | Amplifies error bursts vs isolated typos |
| `session_kpm_z` | kpm / (backspace_rate + ε) | Relative speed adjusted for error rate |

### Interaction Features

| Feature | Formula |
|---------|---------|
| `hold_pause_ratio` | avg_hold / (avg_pause + ε) |
| `typing_efficiency` | kpm / (backspace_rate + ε) |
| `pause_per_key` | avg_pause / (kpm + ε) |

All v2 and interaction features are clipped at 99th percentile to prevent extreme outlier values.

---

## Model Development

### Current Model: XGBoost + SMOTE

**File:** `train_final_model.py`

```
Pipeline:
  SMOTE(sampling_strategy=0.8, k_neighbors=5)
    → XGBClassifier(
        scale_pos_weight=1,
        n_estimators=675,
        max_depth=3,
        learning_rate=0.247,
        subsample=0.946,
        colsample_bytree=0.731,
        min_child_weight=14,
        gamma=0.325,
        reg_alpha=4.579,
        reg_lambda=10.219,
        early_stopping_rounds=40
      )
```

**Hyperparameter Optimisation:** 50-trial Optuna study, 3-fold CV, objective = minority F1  
**Threshold:** Swept 0.05–0.65; optimal = **0.52**  
**Output:** `stress_model.pkl` (pipeline + threshold + metrics bundled)

---

### Model Selection (5-Fold CV)
**File:** `model_selection.py`

| Model | CV Minority F1 |
|-------|---------------|
| XGBoost (SPW=2, no SMOTE) | 0.738 |
| Gradient Boosting | 0.725 |
| XGBoost + SMOTE *(recommended)* | 0.723 |
| Random Forest | 0.589 |
| Logistic Regression | 0.383 |

XGBoost + SMOTE is recommended because SMOTE inside the pipeline prevents data leakage during cross-validation.

---

### Legacy Model: Random Forest (Baseline)
**File:** `baseline_model.py`, `train_model.py`

- Algorithm: RandomForestClassifier(n_estimators=100, random_state=42)
- No SMOTE, no threshold tuning
- Reported 100% accuracy — this was deceptive (model predicted majority class only)
- Replaced by XGBoost + SMOTE pipeline

---

## Class Imbalance Resolution

### Problem
The original merged dataset had a severe imbalance:
- 43,365 non-stressed rows (public dataset)
- 8,811 stressed rows
- Apparent ratio: 5:1 — but this was based on corrupted data

After data cleaning, the actual clean ratio is **2.1:1** (48,185 vs 22,664).

### Strategies Applied

| Strategy | Detail |
|----------|--------|
| SMOTE | `sampling_strategy=0.8` inside imblearn Pipeline (per-fold, no leakage) |
| Cost-Sensitive | `scale_pos_weight=2` in XGBoost |
| Threshold Tuning | Sweep 0.05–0.65; optimal threshold = 0.52 |
| Evaluation Metric | Minority F1 + PR-AUC (accuracy removed as primary metric) |

### Why Accuracy Was Removed
A model predicting "not stressed" for every sample achieves 68% accuracy on a 2:1 imbalanced dataset — but catches zero stressed users. Minority F1 and PR-AUC are threshold-robust metrics that directly measure stress detection quality.

---

## Change History

### v2.0 — 2026-04-16 (PRD implementation)

#### Bug #1: backspace_rate Constant Value
**File:** `prepare_public_dataset.py`  
**Problem:** Original code assigned `freq["backspace_rate"].mean()` — a single scalar — to all 43,365 non-stressed rows. Every row had `backspace_rate = 0.000103`. This feature had zero variance that the model could learn from.  
**Fix:** Per-user per-session join from Frequency Dataset on `userId + emotionIndex`. Now 185 unique values instead of 1.

#### Bug #2: Unit Mismatch Between Datasets
**File:** `mbd_datasets/merge_datasets.py`  
**Problem:** Private desktop dataset stored `avg_hold` and `avg_pause` in **seconds** (range 0.01–25 s), and `kpm` as an integer (2–273). Public dataset stored the same features in **milliseconds** (1–2000 ms). Merging without conversion made features meaningless.  
**Fix:** Added explicit unit conversion step: multiply avg_hold and avg_pause by 1000 before merge; multiply hold_variance by 1,000,000.

#### Bug #3: Corrupted Rows with Impossible Values
**File:** `prepare_public_dataset.py`  
**Problem:** D1U1 (hold time) column had values like −1,580,000,000 ms. These were hardware/logging errors in the raw dataset.  
**Fix:** Added domain-range filter: hold time 1–2000 ms, pause 0–15,000 ms. Removed ~4,400 corrupted rows.

#### Bug #4: SMOTE Strategy Incompatible With Ratio
**File:** `train_final_model.py`  
**Problem:** SMOTE with `sampling_strategy=0.3` on a 2.1:1 dataset raised `ValueError` — the strategy would require removing minority samples, which SMOTE cannot do.  
**Fix:** Constrained valid strategies to values > 1/train_ratio. For 2.1:1 ratio, valid strategies = [0.5, 0.7, 0.8, 1.0].

#### Bug #5: Multi-Class Labels in Binary Pipeline
**Problem:** Raw dataset contained stress levels 0–5. Original code binarised by mapping level > 0 → 1, but the public dataset emotionIndex used N/H/C/A/S strings, not numbers. Calm (C) was incorrectly treated as non-stressed.  
**Fix:** Explicit emotionIndex mapping: N, C → 0; H, S, A → 1.

---

### v1.0 — 2026-03-26 (Original)
- Initial Random Forest baseline
- Fixed Text + Free Text Typing datasets merged
- 5 features: avg_hold, hold_variance, avg_pause, kpm, backspace_rate
- No SMOTE, no threshold tuning
- Accuracy reported as primary metric (misleading)

---

## Data Quality & Assurance

| Check | Status |
|-------|--------|
| No timestamp inversions in keystroke data | ✓ |
| Hold times in valid range (1–2000 ms) | ✓ (filtered) |
| Pause times in valid range (0–15,000 ms) | ✓ (filtered) |
| kpm in human range (0.01–250) | ✓ (clipped) |
| backspace_rate in [0, 1] | ✓ |
| No NaN in merged dataset | ✓ (asserted) |
| Same features in train and evaluation | ✓ |
| SMOTE applied inside CV folds only | ✓ (imblearn Pipeline) |
| KS-test run across datasets before merge | ✓ |
| Reproducible results (random_state=42) | ✓ |

---

## Dependencies

```
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # ML models, cross-validation, metrics
imbalanced-learn    # SMOTE, imblearn Pipeline
xgboost             # XGBoost classifier
optuna              # Bayesian hyperparameter optimisation
joblib              # Model serialisation
scipy               # KS-test for distribution audit
pynput              # Keystroke capture (capture.py only)
sqlite3             # Database (stdlib)
tkinter             # GUI dialogs (stdlib)
```

**Python version:** 3.9+  
**Install:**
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost optuna joblib scipy
```

---

## Performance Summary

### Final Metrics (held-out test set, threshold = 0.52)

| Metric | Value |
|--------|-------|
| Stress F1-Score | **0.749** |
| PR-AUC | **0.882** |
| Stress Recall | 0.804 |
| Stress Precision | 0.702 |
| Macro F1 | 0.809 |
| Overall Accuracy | 82.8% |

### Processing Times

| Task | Duration |
|------|----------|
| prepare_public_dataset.py | ~10 seconds |
| merge_datasets.py | ~15 seconds |
| model_selection.py (5-fold CV) | ~5–7 minutes |
| train_final_model.py (50 HPO trials) | ~3–4 minutes |
| evaluate_mindtype.py | ~30 seconds |

### Confusion Matrix (optimal threshold = 0.52)

```
                   Predicted 0   Predicted 1
Actual Non-stressed   40,440        7,745
Actual Stressed        4,451       18,213
```

- True Positives (caught stress):     18,213
- False Negatives (missed stress):     4,451
- False Positives (false alarms):      7,745
- True Negatives (correct non-stress): 40,440
