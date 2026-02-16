import pandas as pd

df = pd.read_csv("cmu_phase1_processed.csv")

# Remove extreme values using IQR
numeric_cols = df.select_dtypes(include=["float64"]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Shape after outlier removal:", df.shape)

df.to_csv("cmu_phase1_cleaned.csv", index=False)
