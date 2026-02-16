import pandas as pd

df = pd.read_csv("cmu_phase1_processed.csv")

print("Shape:", df.shape)

print("\nSummary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.select_dtypes(include=["float64", "int64"]).corr())
