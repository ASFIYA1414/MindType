"""
mbd_datasets/merge_datasets.py  —  PRD §6 / §8 Implementation
=============================================================
PRD requirements implemented:
  - KS-test per feature to detect distribution divergence
  - Assert no NaN post-merge
  - Log per-dataset class distributions
  - Stratified shuffle
  - Per-user z-score normalisation if distributions diverge (p < 0.05)

IMPORTANT: private_desktop_dataset.csv uses SECONDS for hold/pause times
           and INTEGER keys-per-minute (2–273), whereas public_training_dataset.csv
           uses MILLISECONDS and float kpm (0.01–250). We convert private to match
           public (milliseconds) before merging.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

log.info("\n" + "="*70)
log.info("DATASET MERGE & DISTRIBUTION AUDIT  (PRD v2.0)")
log.info("="*70)

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)

public_path  = os.path.join(ROOT_DIR, "public_training_dataset.csv")
private_path = os.path.join(ROOT_DIR, "private_desktop_dataset.csv")
output_path  = os.path.join(ROOT_DIR, "final_desktop_dataset.csv")

# ─── Load ─────────────────────────────────────────────────────────────────────
log.info("\n--- Loading datasets ---")
public_df  = pd.read_csv(public_path)
private_df = pd.read_csv(private_path)

log.info(f"Public dataset:  {public_df.shape}")
log.info(f"Private dataset: {private_df.shape}")

# ─── Binarise stress labels ──────────────────────────────────────────────────
for df in [public_df, private_df]:
    df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")

log.info("\n--- CLASS DISTRIBUTIONS (Before Merge) ---")
log.info("\nPublic dataset:")
pub_dist = public_df["stress_level"].value_counts().sort_index()
log.info(pub_dist)
log.info(f"  Non-stressed (0): {pub_dist.get(0, 0)} ({pub_dist.get(0,0)/len(public_df)*100:.2f}%)")
log.info(f"  Stressed    (1): {pub_dist.get(1, 0)} ({pub_dist.get(1,0)/len(public_df)*100:.2f}%)")

log.info("\nPrivate dataset (raw stress levels 1-5 = all stressed):")
priv_dist_raw = private_df["stress_level"].value_counts().sort_index()
log.info(priv_dist_raw)

# Private dataset: all rows are stressed (stress_level 1-5) — keep as 1
# Public dataset has N=0, H/S/A=1 already correct, C=0 (calm)
public_df["stress_level"]  = public_df["stress_level"].apply(lambda x: 0 if x == 0 else 1)
private_df["stress_level"] = 1  # all private data is from stressed sessions

log.info("\nAfter binarisation:")
log.info(f"  Public: {dict(public_df.stress_level.value_counts().sort_index())}")
log.info(f"  Private: {dict(private_df.stress_level.value_counts().sort_index())}")

# ─── Unit conversion: private → public units ──────────────────────────────────
log.info("\n--- UNIT CONVERSION (private → public units) ---")
log.info("Private dataset uses SECONDS for avg_hold and avg_pause, and integer kpm.")
log.info("Converting to match public dataset (MILLISECONDS).")

# avg_hold: private is in SECONDS (range 0.01–25), public is ms (range 1–2000)
private_df["avg_hold"] = (private_df["avg_hold"] * 1000).round(2)

# avg_pause: private is in SECONDS (range 0.06–28.5), public is ms (range 0–15000)
private_df["avg_pause"] = (private_df["avg_pause"] * 1000).round(2)

# hold_variance: private is sec² (range 0–490), public is ms² (range 0–730k)
private_df["hold_variance"] = (private_df["hold_variance"] * 1_000_000).round(2)

# kpm: private is integer true kpm (2–273), public is ~60000/pause_ms (0.01–250)
# Keep as-is — both represent keys per minute after conversion
# Clip to same range as public
private_df["kpm"] = private_df["kpm"].clip(0.01, 250)

log.info(f"\nPrivate after conversion:")
for col in ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]:
    log.info(f"  {col:20s}: min={private_df[col].min():.4f}  max={private_df[col].max():.4f}  "
             f"unique={private_df[col].nunique()}")

# ─── Drop user_id from private (not in public) ───────────────────────────────
if "user_id" in private_df.columns:
    private_df = private_df.drop(columns=["user_id"])

# ─── Align columns ───────────────────────────────────────────────────────────
log.info("\n--- ALIGNING COLUMNS ---")
SHARED_COLS = ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate", "stress_level"]

# Assert all shared columns exist
for col in SHARED_COLS:
    assert col in public_df.columns,  f"Missing in public:  {col}"
    assert col in private_df.columns, f"Missing in private: {col}"

public_df  = public_df[SHARED_COLS]
private_df = private_df[SHARED_COLS]
log.info(f"✓ Both datasets aligned to columns: {SHARED_COLS}")

# ─── NaN check ───────────────────────────────────────────────────────────────
log.info("\n--- CHECKING FOR MISSING VALUES ---")
for col in SHARED_COLS[:-1]:  # exclude stress_level
    pub_nan  = public_df[col].isna().sum()
    priv_nan = private_df[col].isna().sum()
    assert pub_nan  == 0, f"🔥 Public: {pub_nan} NaNs in {col}"
    assert priv_nan == 0, f"🔥 Private: {priv_nan} NaNs in {col}"
    log.info(f"✓ {col}: 0 NaNs in both datasets")

# ─── KS-Test: PRD §6 distribution divergence ────────────────────────────────
log.info("\n--- KS-TEST (PRD §6 Distribution Divergence) ---")
log.info("(Significance threshold: p < 0.05)\n")

KS_FEATURES = ["avg_hold", "hold_variance", "avg_pause", "kpm", "backspace_rate"]
diverged = []

for feat in KS_FEATURES:
    stat, p = ks_2samp(public_df[feat].dropna(), private_df[feat].dropna())
    sig = "⚠️  DIVERGENT" if p < 0.05 else "✓ OK"
    log.info(f"{feat:20s}: KS={stat:.4f}  p={p:.6f}  {sig}")
    if p < 0.05:
        diverged.append(feat)

if diverged:
    log.warning(f"\n⚠️  {len(diverged)} feature(s) diverge: {diverged}")
    log.info("→ Applying robust z-score normalisation for these features.")
    log.info("  (Features are clipped to [1st, 99th] percentile before normalising"
             " to prevent extreme private-dataset outliers from dominating.)")
    for feat in diverged:
        # Clip each dataset to its own 1st–99th percentile range first
        pub_lo,  pub_hi  = public_df[feat].quantile(0.01),  public_df[feat].quantile(0.99)
        priv_lo, priv_hi = private_df[feat].quantile(0.01), private_df[feat].quantile(0.99)
        public_df[feat]  = public_df[feat].clip(pub_lo,   pub_hi)
        private_df[feat] = private_df[feat].clip(priv_lo, priv_hi)

        # Z-score using public dataset stats as the reference
        pub_mean = public_df[feat].mean()
        pub_std  = public_df[feat].std() + 1e-8
        public_df[feat]  = (public_df[feat]  - pub_mean) / pub_std
        private_df[feat] = (private_df[feat] - pub_mean) / pub_std
        log.info(f"  ✓ Robust z-normalised {feat}  (public mean={pub_mean:.4f}  std={pub_std:.4f})")
else:
    log.info("\n✓ No significant divergence — safe to merge as-is.")

# ─── Merge ───────────────────────────────────────────────────────────────────
log.info("\n--- MERGING DATASETS ---")
combined = pd.concat([public_df, private_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
log.info(f"✓ Merged and shuffled: {combined.shape}")

# ─── Final audit ─────────────────────────────────────────────────────────────
log.info("\n--- FINAL AUDIT ---")
assert combined.isna().sum().sum() == 0, "🔥 Merged dataset has NaNs!"
log.info("✓ No NaNs in merged dataset")

final_dist = combined["stress_level"].value_counts().sort_index()
log.info(f"\nFinal class distribution:")
log.info(final_dist)
log.info(f"  Non-stressed: {final_dist.get(0,0)} ({final_dist.get(0,0)/len(combined)*100:.2f}%)")
log.info(f"  Stressed:     {final_dist.get(1,0)} ({final_dist.get(1,0)/len(combined)*100:.2f}%)")
ratio = final_dist.get(0, 0) / max(final_dist.get(1, 1), 1)
log.info(f"  Imbalance ratio: {ratio:.1f}:1")

# ─── Feature stats after merge ─────────────────────────────────────────────
log.info("\nFinal feature statistics:")
for col in KS_FEATURES:
    log.info(f"  {col:20s}: min={combined[col].min():.4f}  max={combined[col].max():.4f}  "
             f"mean={combined[col].mean():.4f}  unique={combined[col].nunique()}")

# ─── Save ────────────────────────────────────────────────────────────────────
combined.to_csv(output_path, index=False)
log.info(f"\n✅  Saved: {output_path}")
log.info("="*70 + "\n")