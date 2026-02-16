import pandas as pd
import numpy as np

# Load CMU dataset
df = pd.read_csv("DSL-StrongPasswordData.csv")

# -----------------------------
# Separate feature types
# -----------------------------

hold_cols = [col for col in df.columns if col.startswith("H.")]
dd_cols = [col for col in df.columns if col.startswith("DD.")]
ud_cols = [col for col in df.columns if col.startswith("UD.")]

# -----------------------------
# Compute statistical features
# -----------------------------

df_features = pd.DataFrame()

df_features["avg_hold"] = df[hold_cols].mean(axis=1)
df_features["std_hold"] = df[hold_cols].std(axis=1)

df_features["avg_dd"] = df[dd_cols].mean(axis=1)
df_features["std_dd"] = df[dd_cols].std(axis=1)

df_features["avg_ud"] = df[ud_cols].mean(axis=1)
df_features["std_ud"] = df[ud_cols].std(axis=1)

# Typing speed proxy (inverse of avg hold time)
df_features["typing_speed_proxy"] = 1 / (df_features["avg_hold"] + 1e-5)

# Add baseline stress label
df_features["stress_level"] = 0

# Save processed dataset
df_features.to_csv("cmu_processed_baseline.csv", index=False)

print("CMU dataset processed successfully ✅")
print("Shape:", df_features.shape)
