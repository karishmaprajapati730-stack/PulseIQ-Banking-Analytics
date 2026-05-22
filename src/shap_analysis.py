import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

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
# LOAD MODEL
# =====================================================

model = joblib.load("../models/churn_model.pkl")

# =====================================================
# SHAP EXPLAINER
# =====================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

# =====================================================
# SHAP SUMMARY PLOT
# =====================================================

shap.summary_plot(
    shap_values,
    X_test
)