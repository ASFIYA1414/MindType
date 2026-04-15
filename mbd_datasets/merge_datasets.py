import pandas as pd

# Load datasets
public_df = pd.read_csv("public_training_dataset.csv")
private_df = pd.read_csv("private_desktop_dataset.csv")

print("Public shape:", public_df.shape)
print("Private shape (before fix):", private_df.shape)

# -------------------------------
# 🔥 REMOVE EXTRA COLUMN
# -------------------------------
if "user_id" in private_df.columns:
    private_df = private_df.drop(columns=["user_id"])

print("Private shape (after fix):", private_df.shape)

# -------------------------------
# ENSURE SAME COLUMN ORDER
# -------------------------------
private_df = private_df[public_df.columns]

# -------------------------------
# MERGE
# -------------------------------
combined_df = pd.concat([public_df, private_df], ignore_index=True)

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Final dataset shape:", combined_df.shape)

# -------------------------------
# SAVE
# -------------------------------
combined_df.to_csv("final_desktop_dataset.csv", index=False)

print("Final dataset ready ✅")