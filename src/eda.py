import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
train = pd.read_csv("../data/raw/ChurnZero_dataset_v1.csv")

# Style
sns.set_style("whitegrid")

# =========================
# CHURN DISTRIBUTION
# =========================

plt.figure(figsize=(8,5))

sns.countplot(x='churn', data=train)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customer Count")

plt.show()

# =========================
# MISSING VALUES
# =========================

missing = train.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(12,6))

missing.plot(kind='bar')

plt.title("Missing Values by Feature")
plt.xlabel("Features")
plt.ylabel("Missing Count")

plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

numeric_df = train.select_dtypes(include=np.number)

plt.figure(figsize=(15,10))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x='churn', y='age', data=train)

plt.title("Customer Age vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x='churn', y='avg_monthly_balance', data=train)

plt.title("Average Monthly Balance vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x='churn', y='total_complaints', data=train)

plt.title("Complaints vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(x='churn', y='digital_engagement_index', data=train)

plt.title("Digital Engagement vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='avg_monthly_balance',
    data=train
)

plt.title("Balance vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='total_complaints',
    data=train
)

plt.title("Complaints vs Churn")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x='churn',
    y='digital_engagement_index',
    data=train
)

plt.title("Digital Engagement vs Churn")

plt.show()

correlation = train.corr(numeric_only=True)

churn_corr = correlation['churn'].sort_values(
    ascending=False
)

print("\nTop Correlation With Churn:")
print(churn_corr.head(15))

print("\nNegative Correlation:")
print(churn_corr.tail(15))



