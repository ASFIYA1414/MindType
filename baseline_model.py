import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("DSL-StrongPasswordData.csv")

print("Original Shape:", df.shape)

# Identify timing columns
hold_cols = [col for col in df.columns if col.startswith("H.")]
dd_cols = [col for col in df.columns if col.startswith("DD.")]
ud_cols = [col for col in df.columns if col.startswith("UD.")]

print("Hold columns:", len(hold_cols))
print("DD columns:", len(dd_cols))
print("UD columns:", len(ud_cols))

# Create processed dataset
processed = pd.DataFrame()

# Keep user + session
processed["user"] = df["subject"]
processed["session"] = df["sessionIndex"]

# Aggregate behavioral features
processed["avg_hold"] = df[hold_cols].mean(axis=1)
processed["std_hold"] = df[hold_cols].std(axis=1)

processed["avg_dd"] = df[dd_cols].mean(axis=1)
processed["std_dd"] = df[dd_cols].std(axis=1)

processed["avg_ud"] = df[ud_cols].mean(axis=1)
processed["std_ud"] = df[ud_cols].std(axis=1)

# Typing speed proxy
processed["typing_speed_proxy"] = 1 / (
    processed["avg_hold"] +
    processed["avg_dd"] +
    processed["avg_ud"]
)

# No stress yet → baseline
processed["stress_level"] = 0

print("\nProcessed Shape:", processed.shape)
print(processed.head())

# Save
processed.to_csv("cmu_phase1_processed.csv", index=False)

print("\nPhase 1 processing completed ✅")
