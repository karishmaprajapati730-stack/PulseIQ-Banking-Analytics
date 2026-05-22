import pandas as pd
import numpy as np

print("Libraries imported successfully")

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

print("\nFirst 5 Rows:")
print(train.head())

print("\nDataset Info:")
print(train.info())

print("\nMissing Values:")
print(train.isnull().sum())

print("\nTarget Distribution:")
print(train['churn'].value_counts())