import pandas as pd
import joblib

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("../models/churn_model.pkl")

# ============================================
# LOAD TEST DATA
# ============================================

test = pd.read_csv("../data/processed/test_processed.csv")

# Save customer_id separately
customer_ids = test['customer_id']

# Remove customer_id
test = test.drop(columns=['customer_id'])

# ============================================
# APPLY SAME ENCODING
# ============================================

test = pd.get_dummies(test)

# ============================================
# MATCH TRAINING FEATURES
# ============================================

model_features = model.feature_names_in_

for col in model_features:
    if col not in test.columns:
        test[col] = 0

# Keep only required columns
test = test[model_features]

# ============================================
# MAKE PREDICTIONS
# ============================================

predictions = model.predict(test)

# ============================================
# CREATE SUBMISSION FILE
# ============================================

submission = pd.DataFrame({
    "customer_id": customer_ids,
    "prediction": predictions
})

# ============================================
# SAVE CSV
# ============================================

submission.to_csv(
    "../submissions/predictions.csv",
    index=False
)

print("Predictions saved successfully!")