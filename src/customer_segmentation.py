import pandas as pd
import numpy as np

# =====================================================
# LOAD DATA
# =====================================================

train = pd.read_csv("../data/processed/train_processed.csv")

# =====================================================
# CREATE RISK SEGMENTS
# =====================================================

conditions = [

    (
        (train['complaint_ratio'] > 0.5)
        &
        (train['digital_engagement_combined'] < 50)
    ),

    (
        (train['engagement_score'] < 100)
        &
        (train['avg_monthly_balance'] < 5000)
    ),

    (
        (train['number_of_products'] <= 2)
    )
]

segments = [
    "High Complaint Risk",
    "Digitally Disengaged",
    "Low Product Usage"
]

train['customer_segment'] = np.select(
    conditions,
    segments,
    default="Stable Customer"
)

# =====================================================
# SEGMENT DISTRIBUTION
# =====================================================

segment_counts = train['customer_segment'].value_counts()

print("\n========== CUSTOMER SEGMENTS ==========\n")

print(segment_counts)

# =====================================================
# CHURN RATE BY SEGMENT
# =====================================================

segment_churn = train.groupby(
    'customer_segment'
)['churn'].mean() * 100

print("\n========== CHURN RATE BY SEGMENT ==========\n")

print(segment_churn)

# =====================================================
# SAVE RESULTS
# =====================================================

train.to_csv(
    "../outputs/reports/customer_segments.csv",
    index=False
)

print("\nCustomer segmentation completed!")