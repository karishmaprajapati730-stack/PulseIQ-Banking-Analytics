import pandas as pd
import numpy as np

# Load datasets
train = pd.read_csv("data/raw/ChurnZero_dataset_v1.csv")
test = pd.read_csv("data/raw/ChurnZero_test_v1.csv")

# Dataset Shape
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# First 5 Rows
print("\nFirst 5 Rows:")
print(train.head())

# Dataset Information
print("\nDataset Info:")
print(train.info())

# Missing Values
print("\nMissing Values:")
print(train.isnull().sum())

# Target Distribution
print("\nTarget Distribution:")
print(train['churn'].value_counts())

# Target Percentage
print("\nTarget Percentage:")
print(train['churn'].value_counts(normalize=True) * 100)

# Duplicate Rows
print("\nDuplicate Rows:")
print(train.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(train.describe())

print("\nColumn Names:")
print(train.columns.tolist())

print("\nUnique Values Per Column:")
print(train.nunique().sort_values())

categorical_cols = train.select_dtypes(include='object').columns

print("\nCategorical Columns:")
print(categorical_cols)

numerical_cols = train.select_dtypes(exclude='object').columns

print("\nNumerical Columns:")
print(numerical_cols)

