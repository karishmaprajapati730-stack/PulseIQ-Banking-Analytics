import streamlit as st
import pandas as pd
import joblib

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("models/churn_model.pkl")

# =====================================================
# APP TITLE
# =====================================================

st.title("🏦 BankPulse-AI")
st.subheader("AI-Powered Customer Churn Prediction")

st.write(
    "Enter customer details to predict churn risk."
)

# =====================================================
# USER INPUTS
# =====================================================

age = st.slider("Age", 18, 80, 35)

balance = st.number_input(
    "Average Monthly Balance",
    min_value=0.0,
    value=10000.0
)

products = st.slider(
    "Number of Products",
    1,
    10,
    2
)

complaints = st.slider(
    "Total Complaints",
    0,
    20,
    1
)

engagement = st.slider(
    "Engagement Score",
    0,
    500,
    100
)

digital_logins = st.slider(
    "Total Digital Logins",
    0,
    1000,
    100
)

# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({

    'age': [age],

    'avg_monthly_balance': [balance],

    'number_of_products': [products],

    'total_complaints': [complaints],

    'engagement_score': [engagement],

    'total_digital_logins': [digital_logins]
})

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    st.write(f"### Churn Probability: {probability:.2%}")

    if prediction == 1:

        st.error(
            "⚠️ High Risk Customer Likely To Churn"
        )

    else:

        st.success(
            "✅ Customer Likely To Stay"
        )