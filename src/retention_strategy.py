import pandas as pd

# =====================================================
# LOAD SEGMENTED DATA
# =====================================================

train = pd.read_csv(
    "../outputs/reports/customer_segments.csv"
)

# =====================================================
# RETENTION STRATEGY MAPPING
# =====================================================

strategy_map = {

    "High Complaint Risk":
        "Assign dedicated relationship manager and priority support.",

    "Digitally Disengaged":
        "Launch mobile banking onboarding and engagement campaigns.",

    "Low Product Usage":
        "Provide personalized cross-sell offers and loyalty rewards.",

    "Stable Customer":
        "Maintain regular engagement and premium service quality."
}

# =====================================================
# ASSIGN STRATEGIES
# =====================================================

train['retention_strategy'] = train[
    'customer_segment'
].map(strategy_map)

# =====================================================
# SAVE RESULTS
# =====================================================

train.to_csv(
    "../outputs/reports/customer_retention_strategy.csv",
    index=False
)

# =====================================================
# DISPLAY SAMPLE OUTPUT
# =====================================================

print("\n========== RETENTION STRATEGIES ==========\n")

print(
    train[
        [
            'customer_id',
            'customer_segment',
            'retention_strategy'
        ]
    ].head(10)
)

print("\nRetention strategy generation completed!")

