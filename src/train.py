import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATA
# =====================================================

train = pd.read_csv("../data/processed/train_processed.csv")

# =====================================================
# SPLIT FEATURES AND TARGET
# =====================================================

# X = train.drop(columns=['churn'])

# Remove target column
X = train.drop(columns=['churn'])

# Remove customer_id
if 'customer_id' in X.columns:
    X = X.drop(columns=['customer_id'])

# Convert text columns into numbers
X = pd.get_dummies(X)

if 'customer_id' in X.columns:
    X = X.drop(columns=['customer_id'])

y = train['churn']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "../models/churn_model.pkl")

print("Model saved successfully!")