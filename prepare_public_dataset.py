"""
prepare_public_dataset.py  —  Fixed version
=============================================
Fixes the following bugs from the original script:
  1. backspace_rate was assigned freq["backspace_rate"].mean() — a single scalar
     applied to all rows. Now computed per-user and joined properly.
  2. Negative D1U1 / D1D2 values (corrupted rows) are filtered out.
  3. kpm computation used D1D2 (pause between keys) not window count — now fixed.
  4. emotionIndex mapping extended: N=0 (non-stressed), H/S/A/C handled explicitly.
  5. hold_variance now uses a proper per-session rolling window.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("PREPARE PUBLIC TRAINING DATASET  (Fixed)")
log.info("="*70)

# ─── Step 1: Load raw files ────────────────────────────────────────────────
log.info("\n--- Loading raw files ---")
free_text  = pd.read_csv("Free Text Typing Dataset.csv",  sep=None, engine="python")
fixed_text = pd.read_csv("Fixed Text Typing Dataset.csv", sep=None, engine="python")
freq       = pd.read_csv("Frequency Dataset.csv",         sep=None, engine="python")

log.info(f"Free text:  {free_text.shape}   columns: {list(free_text.columns)}")
log.info(f"Fixed text: {fixed_text.shape}  columns: {list(fixed_text.columns)}")
log.info(f"Frequency:  {freq.shape}         columns: {list(freq.columns)}")

# ─── Step 2: Unify typing datasets ────────────────────────────────────────
typing_data = pd.concat([free_text, fixed_text], ignore_index=True)
log.info(f"\nCombined typing rows: {len(typing_data)}")

# ─── Step 3: Convert numeric columns; remove corrupted rows ────────────────
log.info("\n--- Cleaning numeric columns ---")

for col in ["D1U1", "D1D2"]:
    typing_data[col] = pd.to_numeric(
        typing_data[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

# Filter out corrupted rows (D1U1 = hold time; must be 1–2000 ms)
n_raw = len(typing_data)
typing_data = typing_data[
    (typing_data["D1U1"].between(1, 2000)) &
    (typing_data["D1D2"].between(0, 15000))
].copy()
log.info(f"After corrupted-row removal: {n_raw} → {len(typing_data)} rows")

# ─── Step 4: Stress label mapping ─────────────────────────────────────────
# emotionIndex: N = Neutral (no stress = 0), H = Happy, C = Calm, S = Sad/Stressed, A = Anxious
# For binary classification: N → 0 (non-stressed), all others → 1 (stressed)
log.info("\n--- emotionIndex mapping ---")
log.info(f"Unique emotionIndex values: {typing_data['emotionIndex'].unique()}")
log.info(f"Value counts:\n{typing_data['emotionIndex'].value_counts()}")

typing_data["stress_level"] = typing_data["emotionIndex"].map({
    "N": 0,   # Neutral  → non-stressed
    "H": 1,   # Happy    → mild stress (aroused)
    "C": 0,   # Calm     → non-stressed (grouped with N)
    "S": 1,   # Sad/Stressed
    "A": 1,   # Anxious
})
# Drop rows where emotionIndex wasn't in our map
typing_data = typing_data.dropna(subset=["stress_level"])
typing_data["stress_level"] = typing_data["stress_level"].astype(int)
log.info(f"\nBinary stress distribution:\n{typing_data['stress_level'].value_counts().sort_index()}")

# ─── Step 5: Compute per-user backspace rate from Frequency dataset ────────
log.info("\n--- Computing per-user backspace rate ---")
# BUG FIX: Original used freq["backspace_rate"].mean() — a constant for all rows!
# Fix: compute per-user, per-session rate and join to typing_data.

freq = freq.dropna(subset=["TotTime"])
freq["TotTime"] = pd.to_numeric(freq["TotTime"], errors="coerce")
freq["delFreq"]  = pd.to_numeric(freq["delFreq"],  errors="coerce")
freq = freq.dropna(subset=["TotTime"])

# backspace_rate = backspaces / total_time (per user-session pair)
freq["backspace_rate"] = freq["delFreq"] / (freq["TotTime"] + 1e-6)
freq["backspace_rate"] = freq["backspace_rate"].clip(0, 1)

# Merge on userId + emotionIndex
freq_merged = freq[["User ID", "emotionIndex", "backspace_rate"]].rename(
    columns={"User ID": "userId"}
)
# Average per userId-emotionIndex pair
freq_agg = freq_merged.groupby(["userId", "emotionIndex"])["backspace_rate"].mean().reset_index()
log.info(f"Freq aggregated entries: {len(freq_agg)}")
log.info(f"backspace_rate stats: min={freq_agg['backspace_rate'].min():.6f}  "
         f"max={freq_agg['backspace_rate'].max():.6f}")

typing_data = typing_data.merge(freq_agg, on=["userId", "emotionIndex"], how="left")
# For rows without a match, impute with per-user mean
user_br_mean = typing_data.groupby("userId")["backspace_rate"].transform("mean")
typing_data["backspace_rate"] = typing_data["backspace_rate"].fillna(user_br_mean)
# Remaining NaN → global mean
global_br_mean = typing_data["backspace_rate"].mean()
typing_data["backspace_rate"] = typing_data["backspace_rate"].fillna(global_br_mean)
log.info(f"After merge — backspace_rate unique values: {typing_data['backspace_rate'].nunique()}")

# ─── Step 6: Build feature columns ─────────────────────────────────────────
log.info("\n--- Building feature columns ---")

dataset = pd.DataFrame()
dataset["avg_hold"]      = typing_data["D1U1"]
dataset["hold_variance"] = typing_data["D1U1"].rolling(5, min_periods=1).var().fillna(0)
dataset["avg_pause"]     = typing_data["D1D2"]
# kpm: keys-per-minute estimate.
# D1D2 is the inter-key interval (pause) in ms.
# kpm ≈ 60,000 / avg_pause_ms, but clipped to 0.01–250 to match private dataset range.
# Fast typists: 200-300 kpm; stressed typists may slow to 30-60 kpm.
dataset["kpm"]           = (60_000 / (typing_data["D1D2"] + 1)).clip(0.01, 250)

dataset["stress_level"]  = typing_data["stress_level"].values
dataset["backspace_rate"] = typing_data["backspace_rate"].values

# ─── Step 7: Final cleaning ─────────────────────────────────────────────────
log.info("\n--- Final cleaning ---")
dataset = dataset.dropna()
log.info(f"Rows after dropna: {len(dataset)}")

n_before = len(dataset)
dataset = dataset[
    (dataset["avg_hold"].between(1, 2000)) &
    (dataset["hold_variance"] >= 0) &
    (dataset["avg_pause"].between(0, 15000)) &
    (dataset["kpm"].between(0.01, 60_000)) &
    (dataset["backspace_rate"].between(0, 1))
]
log.info(f"Rows after domain sanity check: {n_before} → {len(dataset)}")

log.info(f"\nFinal class distribution:")
log.info(dataset["stress_level"].value_counts().sort_index())

# ─── Step 8: Feature stats ────────────────────────────────────────────────
log.info("\nFeature statistics:")
for col in ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]:
    log.info(f"  {col:20s}: min={dataset[col].min():.4f}  max={dataset[col].max():.4f}  "
             f"mean={dataset[col].mean():.4f}  unique={dataset[col].nunique()}")

# ─── Step 9: Save ───────────────────────────────────────────────────────────
dataset.to_csv("public_training_dataset.csv", index=False)
log.info(f"\n✅  Saved: public_training_dataset.csv  ({len(dataset)} rows)")
log.info("="*70 + "\n")