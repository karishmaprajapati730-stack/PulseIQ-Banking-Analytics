import pandas as pd
import numpy as np

# =========================================================
# LOAD DATASETS
# =========================================================

train = pd.read_csv("../data/raw/ChurnZero_dataset_v1.csv")
test = pd.read_csv("../data/raw/ChurnZero_test_v1.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

# 1. Engagement Score
train['engagement_score'] = (
    train['mobile_app_login_count']
    + train['net_banking_transaction_count']
    + train['upi_transaction_count']
)

test['engagement_score'] = (
    test['mobile_app_login_count']
    + test['net_banking_transaction_count']
    + test['upi_transaction_count']
)

# 2. Complaint Ratio
train['complaint_ratio'] = (
    train['unresolved_complaint_count'] /
    (train['total_complaints'] + 1)
)

test['complaint_ratio'] = (
    test['unresolved_complaint_count'] /
    (test['total_complaints'] + 1)
)

# 3. Credit Utilization Difference
train['credit_utilization_diff'] = (
    train['credit_utilization_6m_avg']
    - train['credit_utilization_3m_avg']
)

test['credit_utilization_diff'] = (
    test['credit_utilization_6m_avg']
    - test['credit_utilization_3m_avg']
)

# 4. Average Transaction Value
train['avg_transaction_value'] = (
    train['total_trans_amt'] /
    (train['total_trans_count'] + 1)
)

test['avg_transaction_value'] = (
    test['total_trans_amt'] /
    (test['total_trans_count'] + 1)
)

# 5. Digital Engagement Combined
train['digital_engagement_combined'] = (
    train['digital_engagement_index']
    + train['digital_service_usage_score']
)

test['digital_engagement_combined'] = (
    test['digital_engagement_index']
    + test['digital_service_usage_score']
)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# =========================================================
# SAVE PROCESSED FILES
# =========================================================

train.to_csv(
    "../data/processed/train_processed.csv",
    index=False
)

test.to_csv(
    "../data/processed/test_processed.csv",
    index=False
)

print("\nFeature Engineering Completed Successfully!")
print("Processed files saved in data/processed/")