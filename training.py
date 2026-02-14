import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

df1 = pd.read_csv("user1/keystrokes.tsv", sep="\t")
cond1 = pd.read_csv("user1/usercondition.tsv", sep="\t")

df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
cond1 = cond1.loc[:, ~cond1.columns.str.contains('^Unnamed')]

df1["Press_Time"] = pd.to_datetime(df1["Press_Time"])
df1["Relase_Time"] = pd.to_datetime(df1["Relase_Time"])
cond1["Time"] = pd.to_datetime(cond1["Time"])

# Feature extraction
df1["hold_time"] = (df1["Relase_Time"] - df1["Press_Time"]).dt.total_seconds()
df1["pause"] = (df1["Press_Time"] - df1["Relase_Time"].shift(1)).dt.total_seconds()
df1["is_backspace"] = df1["Key"].apply(lambda x: 1 if "backspace" in str(x).lower() else 0)

# Window aggregation (5-minute)
dataset = df1.groupby(df1["Press_Time"].dt.floor("5min")).agg(
    avg_hold=("hold_time", "mean"),
    std_hold=("hold_time", "std"),
    avg_flight=("pause", "mean"),
    avg_pause=("pause", "mean"),
    backspace_rate=("is_backspace", "mean"),
    key_count=("Key", "count")
).reset_index().rename(columns={"Press_Time": "window"})

dataset["kpm"] = dataset["key_count"] / 5

# Merge with condition
cond1 = cond1.sort_values("Time").reset_index(drop=True)
cond1["stress_label"] = cond1["Stress_Val"].apply(lambda x: 1 if "Stressed" in str(x) else 0)

dataset = pd.merge_asof(
    dataset.sort_values("window"),
    cond1[["Time", "stress_label"]],
    left_on="window",
    right_on="Time",
    direction="nearest"
).drop(columns=["Time"])

dataset = dataset.dropna()

# -------------------------
# Step 2: Process user2
# -------------------------
df2 = pd.read_csv("user2/keystrokes.tsv", sep="\t")
cond2 = pd.read_csv("user2/usercondition.tsv", sep="\t")

df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
cond2 = cond2.loc[:, ~cond2.columns.str.contains('^Unnamed')]

df2["Press_Time"] = pd.to_datetime(df2["Press_Time"])
df2["Relase_Time"] = pd.to_datetime(df2["Relase_Time"])
cond2["Time"] = pd.to_datetime(cond2["Time"])

df2["hold_time"] = (df2["Relase_Time"] - df2["Press_Time"]).dt.total_seconds()
df2["pause"] = (df2["Press_Time"] - df2["Relase_Time"].shift(1)).dt.total_seconds()
df2["is_backspace"] = df2["Key"].apply(lambda x: 1 if "backspace" in str(x).lower() else 0)

dataset2 = df2.groupby(df2["Press_Time"].dt.floor("5min")).agg(
    avg_hold=("hold_time", "mean"),
    std_hold=("hold_time", "std"),
    avg_flight=("pause", "mean"),
    avg_pause=("pause", "mean"),
    backspace_rate=("is_backspace", "mean"),
    key_count=("Key", "count")
).reset_index().rename(columns={"Press_Time": "window"})

dataset2["kpm"] = dataset2["key_count"] / 5

cond2 = cond2.sort_values("Time").reset_index(drop=True)
cond2["stress_label"] = cond2["Stress_Val"].apply(lambda x: 1 if "Stressed" in str(x) else 0)

dataset2 = pd.merge_asof(
    dataset2.sort_values("window"),
    cond2[["Time", "stress_label"]],
    left_on="window",
    right_on="Time",
    direction="nearest"
).drop(columns=["Time"])

dataset2 = dataset2.dropna()


dataset["user_id"] = 1
dataset2["user_id"] = 2

final_dataset = pd.concat([dataset, dataset2], ignore_index=True)
print("Users in dataset:")
print(final_dataset["user_id"].value_counts())

# -------------------------
# Step 5: Prepare features and target
# -------------------------
X = final_dataset[[
    "avg_hold",
    "std_hold",
    "avg_flight",
    "backspace_rate",
    "kpm"
]]

y = final_dataset["stress_label"]

# -------------------------
# Step 6: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Step 7: Train model
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight='balanced', max_iter=1000))
])

pipeline.fit(X_train, y_train)

# -------------------------
# Step 8: Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# Step 9: Save trained model (optional)
# -------------------------
joblib.dump(pipeline, "stress_model.pkl")
print("Trained model saved as stress_model.pkl")
print(y.value_counts())