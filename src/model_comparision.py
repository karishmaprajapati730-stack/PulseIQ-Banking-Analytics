import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

train = pd.read_csv("../data/processed/train_processed.csv")

# =====================================================
# FEATURES & TARGET
# =====================================================

X = train.drop(columns=['churn'])

if 'customer_id' in X.columns:
    X = X.drop(columns=['customer_id'])

# Convert categorical columns
X = pd.get_dummies(X)

y = train['churn']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}

# =====================================================
# MODEL COMPARISON
# =====================================================

results = []

for model_name, model in models.items():

    print(f"\n========== {model_name} ==========\n")

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    business_cost = (
        (fn * 40000)
        + (fp * 500)
    )

    # Store results
    results.append({
        "Model": model_name,
        "Recall": recall,
        "F1 Score": f1,
        "Business Cost": business_cost
    })

    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Business Cost: ₹{business_cost:,}")

# =====================================================
# FINAL RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========\n")

print(results_df)