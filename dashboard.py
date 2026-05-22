import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BankPulse Executive Dashboard",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to right,
        #0f172a,
        #111827
    );
    color: white;
}

.metric-card {
    background-color: #1F2937;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.05);
}

.big-title {
    font-size: 50px;
    font-weight: bold;
    color: #60A5FA;
}

.subtitle {
    font-size: 20px;
    color: #D1D5DB;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD REAL DATA
# =====================================================

train = pd.read_csv(
    "data/processed/train_processed.csv"
)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="big-title">🏦 BankPulse Executive Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Banking Retention Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.write("")

# =====================================================
# KPI CARDS
# =====================================================

total_customers = len(train)

churn_rate = round(
    train['churn'].mean() * 100,
    2
)

high_risk = train[
    train['churn'] == 1
].shape[0]

estimated_loss = high_risk * 40000

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{total_customers}</h2>
        <p>Total Customers</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{churn_rate}%</h2>
        <p>Churn Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{high_risk}</h2>
        <p>Churned Customers</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h2>₹{estimated_loss:,}</h2>
        <p>Potential Revenue Loss</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# CHURN TREND ANALYSIS
# =====================================================

st.subheader("📈 Churn Trend Analysis")

churn_counts = train['churn'].value_counts()

trend_data = pd.DataFrame({
    "Status": ["Non-Churn", "Churn"],
    "Customers": churn_counts.values
})

fig = px.bar(
    trend_data,
    x="Status",
    y="Customers",
    color="Status",
    title="Customer Churn Distribution"
)

fig = px.line(
    trend_data,
    x="Month",
    y="Churn Count",
    markers=True,
    title="Monthly Customer Churn Trend"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CUSTOMER RISK SEGMENTS
# =====================================================

segment_data = pd.DataFrame({

    "Segment": [
        "High Complaint Risk",
        "Digitally Disengaged",
        "Low Product Usage",
        "Stable Customers"
    ],

    "Customers": [
        train[train['complaint_ratio'] > 0.5].shape[0],

        train[
            train['digital_engagement_combined'] < 50
        ].shape[0],

        train[
            train['number_of_products'] <= 2
        ].shape[0],

        train[
            train['churn'] == 0
        ].shape[0]
    ]
})



# =====================================================
# BUSINESS COST ANALYSIS
# =====================================================

st.subheader("💰 Business Cost Impact")

cost_data = pd.DataFrame({
    "Category": [
        "False Negatives",
        "False Positives"
    ],
    "Cost": [
        4000000,
        50000
    ]
})

fig3 = px.bar(
    cost_data,
    x="Category",
    y="Cost",
    title="Business Loss Analysis"
)

st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# RETENTION STRATEGY
# =====================================================

st.subheader("🎯 AI Retention Recommendations")

strategy_data = pd.DataFrame({

    "Segment": [
        "High Complaint Risk",
        "Digitally Disengaged",
        "Low Product Usage"
    ],

    "Recommendation": [
        "Priority customer support",
        "Mobile onboarding campaign",
        "Cross-sell premium products"
    ]
})

st.dataframe(
    strategy_data,
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.write("---")

st.caption(
    "Built using Streamlit, Plotly, Machine Learning & Banking Analytics"
)