# MindType — Keystroke Dynamics Stress Detection

> Detecting cognitive stress in real time through typing patterns, using machine learning on keystroke dynamics data.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Feature Engineering](#feature-engineering)
- [Class Imbalance Strategy](#class-imbalance-strategy)

---

## Overview

MindType is a machine learning system that detects **cognitive stress** by analyzing how a person types — key hold times, pauses between keys, typing speed, and error patterns. No cameras, no wearables — just the keyboard.

The system was designed to solve a real-world class imbalance problem: in typing data, stressed sessions are less frequent than non-stressed ones. The model is specifically tuned to detect these rare stressed sessions accurately using SMOTE oversampling and XGBoost with Optuna hyperparameter optimization.

**Primary goal:** Maximize the **F1-score for the stressed class** — catching as many real stress events as possible.

---

## How It Works

```
User types
    ↓
Keystroke events captured (hold times, pause intervals, kpm, error rate)
    ↓
13 features extracted (base + v2 + interaction)
    ↓
XGBoost classifier trained with SMOTE oversampling
    ↓
Probability score → threshold (0.52) → Stressed / Not Stressed
```

---

## Dataset

### Sources

| Dataset | Description | Rows |
|---------|-------------|------|
| Fixed Text Typing Dataset | Participants typed fixed sentences under 5 emotional conditions | 46,871 keystrokes |
| Free Text Typing Dataset | Free-form typing under same emotional conditions | 28,412 keystrokes |
| Frequency Dataset | Per-session backspace and total-time stats | 478 sessions |
| Private Desktop Dataset | Real-world stressed typing sessions | 2,420 samples |

### Label Encoding

```
Neutral (N), Calm (C)       →  0  (non-stressed)
Happy (H), Sad (S), Anxious (A)  →  1  (stressed)
```

### Final Class Distribution

```
Non-stressed (0):  48,185  samples  (68%)
Stressed     (1):  22,664  samples  (32%)
Imbalance ratio:   ~2.1 : 1
```

---

## Model Pipeline

```
Raw keystroke events
        ↓
  Feature Extraction       (extract_window_features.py)
        ↓
  Dataset Preparation      (prepare_public_dataset.py)
        ↓
  Dataset Merge            (mbd_datasets/merge_datasets.py)
        ↓
  Model Selection / CV     (model_selection.py)
        ↓
  SMOTE + XGBoost + HPO    (train_final_model.py)
        ↓
  Evaluation               (evaluate_mindtype.py)
```

**Algorithm:** XGBoost + SMOTE inside `imblearn.Pipeline` (no data leakage)  
**HPO:** 50-trial Optuna study, objective = minority class F1  
**Threshold:** Tuned via sweep 0.05–0.65, optimal = **0.52**

---

## Results

| Metric | Value |
|--------|-------|
| Stress F1-Score | **0.749** |
| PR-AUC | **0.882** |
| Stress Recall | 0.804 |
| Stress Precision | 0.702 |
| Macro F1 | 0.809 |
| Overall Accuracy | 82.8% |

### Model Comparison (5-Fold CV, Stress F1)

| Model | CV F1 |
|-------|-------|
| XGBoost (SPW=2) | 0.738 |
| Gradient Boosting | 0.725 |
| XGBoost + SMOTE *(recommended)* | 0.723 |
| Random Forest | 0.589 |
| Logistic Regression | 0.383 |

---

## Project Structure

```
MindType-1/
│
├── docs/
│   ├── README.md                    ← Project overview (this file, copy)
│   └── AUDIT_LOG.md                 ← Full change history and design decisions
│
├── README.md                        ← Project overview
├── AUDIT_LOG.md                     ← Change history
├── MindType_Imbalance_PRD.txt       ← Product Requirements Document
│
├── Fixed Text Typing Dataset.csv    ← Raw public keystroke data
├── Free Text Typing Dataset.csv     ← Raw public free-text data
├── Frequency Dataset.csv            ← Per-session backspace statistics
├── private_desktop_dataset.csv      ← Real-world stressed sessions
│
├── prepare_public_dataset.py        ← Build public_training_dataset.csv
├── public_training_dataset.csv      ← Cleaned, labelled public dataset
│
├── mbd_datasets/
│   ├── merge_datasets.py            ← KS-test audit + merge datasets
│   └── mindtype_dataset*.csv        ← Intermediate MindType exports
│
├── final_desktop_dataset.csv        ← Merged final training data
│
├── extract_window_features.py       ← Sliding-window feature extraction
├── model_selection.py               ← 5-fold CV model comparison
├── train_final_model.py             ← XGBoost + SMOTE + Optuna → model
├── evaluate_mindtype.py             ← Full evaluation with all metrics
│
├── stress_model.pkl                 ← Trained model + threshold (generated)
├── model_metrics.pkl                ← Metrics dict (generated)
│
├── capture.py                       ← Live keystroke capture utility
├── baseline_model.py                ← Original baseline for reference
└── venv/                            ← Python virtual environment
```

---

## Setup & Installation

```bash
cd MindType-1
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn xgboost optuna joblib scipy
```

---

## How to Run

Run in this exact order:

```bash
# Activate environment
source venv/bin/activate

# Step 1 — Rebuild public dataset (fixes data pipeline bugs)
python3 prepare_public_dataset.py

# Step 2 — Merge datasets with KS-test distribution audit
python3 mbd_datasets/merge_datasets.py

# Step 3 — Compare models (5-fold cross-validation)
python3 model_selection.py

# Step 4 — Train final XGBoost + SMOTE + Optuna model
python3 train_final_model.py

# Step 5 — Evaluate on held-out data
python3 evaluate_mindtype.py
```

**Runtimes:** Steps 1–2: ~15 sec | Step 3: ~6 min | Step 4: ~4 min | Step 5: ~30 sec

---

## Feature Engineering

### Base Features (original 5)

| Feature | Description |
|---------|-------------|
| `avg_hold` | Average key hold time (ms) |
| `hold_variance` | Variance of hold times |
| `avg_pause` | Average inter-key interval (ms) |
| `kpm` | Keys per minute |
| `backspace_rate` | Fraction of backspace keypresses |

### v2 Features (added)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `hold_cv` | std(holds) / mean(holds) | Normalised hold irregularity |
| `iki_entropy` | log(1 + avg_pause) | Inter-key timing unpredictability |
| `digraph_latency` | avg_pause / kpm | Transition time per keystroke |
| `error_burst_rate` | backspace_rate² | Amplifies error clusters |
| `session_kpm_z` | kpm / (backspace_rate + ε) | Relative speed vs error rate |

### Interaction Features

| Feature | Formula |
|---------|---------|
| `hold_pause_ratio` | avg_hold / avg_pause |
| `typing_efficiency` | kpm / (backspace_rate + ε) |
| `pause_per_key` | avg_pause / kpm |

---

## Class Imbalance Strategy

| Technique | Setting | Purpose |
|-----------|---------|---------|
| SMOTE | `sampling_strategy=0.8` | Synthetic minority oversampling |
| Cost-sensitive learning | `scale_pos_weight=2` | Penalise missed stress more |
| Threshold tuning | Optimal = 0.52 | Maximise stress class F1 |
| Primary metric | Minority F1 + PR-AUC | Accuracy not used (misleading at imbalance) |

> See `docs/AUDIT_LOG.md` for full design decisions and change history.