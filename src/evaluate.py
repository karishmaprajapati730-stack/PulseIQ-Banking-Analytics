import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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
# LOAD MODEL
# =====================================================

model = joblib.load("../models/churn_model.pkl")

# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(y_test, predictions))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== TOP FEATURES ==========\n")

print(feature_importance.head(15))

# =====================================================
# FEATURE IMPORTANCE VISUALIZATION
# =====================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features
)

plt.title("Top 15 Important Features")

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()

# =====================================================
# BUSINESS COST ANALYSIS
# =====================================================

# False Negative Cost = 40000
# False Positive Cost = 500

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

business_cost = (
    (fn * 40000)
    + (fp * 500)
)

print("\n========== BUSINESS COST ANALYSIS ==========\n")

print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives : {tp}")

print(f"\nEstimated Business Cost: ₹{business_cost:,}")

# =====================================================
# THRESHOLD OPTIMIZATION
# =====================================================

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6]

print("\n========== THRESHOLD ANALYSIS ==========\n")

for threshold in thresholds:

    custom_predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        custom_predictions
    ).ravel()

    recall = recall_score(
        y_test,
        custom_predictions
    )

    cost = (
        (fn * 40000)
        + (fp * 500)
    )

    print(f"\nThreshold: {threshold}")
    print(f"Recall   : {recall:.4f}")
    print(f"FN       : {fn}")
    print(f"FP       : {fp}")
    print(f"Cost     : ₹{cost:,}")