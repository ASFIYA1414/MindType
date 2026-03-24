import pandas as pd
import numpy as np

# -------------------------------------
# STEP 1 — Load datasets
# -------------------------------------

free_text = pd.read_csv(
    "Free Text Typing Dataset.csv",
    sep=None,
    engine="python"
)

fixed_text = pd.read_csv(
    "Fixed Text Typing Dataset.csv",
    sep=None,
    engine="python"
)

freq = pd.read_csv(
    "Frequency Dataset.csv",
    sep=None,
    engine="python"
)

print("Free text columns:", free_text.columns)
print("Fixed text columns:", fixed_text.columns)
print("Frequency columns:", freq.columns)


# -------------------------------------
# STEP 2 — Combine typing datasets
# -------------------------------------

typing_data = pd.concat([free_text, fixed_text], ignore_index=True)

print("Total typing rows:", len(typing_data))
print("Combined columns:", typing_data.columns)


# -------------------------------------
# STEP 3 — Convert numeric columns
# -------------------------------------

typing_data["D1U1"] = pd.to_numeric(
    typing_data["D1U1"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

typing_data["D1D2"] = pd.to_numeric(
    typing_data["D1D2"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)


# -------------------------------------
# STEP 4 — Create MindType features
# -------------------------------------

dataset = pd.DataFrame()

# Average hold time
dataset["avg_hold"] = typing_data["D1U1"]

# Hold variance
dataset["hold_variance"] = (
    typing_data["D1U1"]
    .rolling(5)
    .var()
    .fillna(0)
)

# Average pause time
dataset["avg_pause"] = typing_data["D1D2"]

# Typing speed approximation
dataset["kpm"] = 60 / (typing_data["D1D2"] + 1e-6)

# Stress labels
dataset["stress_level"] = typing_data["emotionIndex"].map({
    "N": 0,
    "H": 1
})


# -------------------------------------
# STEP 5 — Add backspace rate
# -------------------------------------

freq["backspace_rate"] = freq["delFreq"] / (freq["TotTime"] + 1e-6)

dataset["backspace_rate"] = freq["backspace_rate"].mean()


# -------------------------------------
# STEP 6 — Clean dataset
# -------------------------------------

dataset = dataset.dropna()

print("Final dataset size:", dataset.shape)


# -------------------------------------
# STEP 7 — Save dataset
# -------------------------------------

dataset.to_csv("public_training_dataset.csv", index=False)

print("Public training dataset created successfully.")