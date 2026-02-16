import pandas as pd
import numpy as np

# Load processed CMU dataset
df = pd.read_csv("cmu_processed_baseline.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nChecking Missing Values:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.corr())
