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

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>92%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>0.89</h2>
        <p>Recall Score</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>₹4.2M</h2>
        <p>Potential Savings</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h2>1,423</h2>
        <p>High Risk Customers</p>
    </div>
    """, unsafe_allow_html=True
)

st.write("")
st.write("---")

# =====================================================
# CHURN TREND ANALYSIS
# =====================================================

st.subheader("📈 Churn Trend Analysis")

trend_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Churn Count": [120, 140, 170, 160, 190, 175]
})

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

st.subheader("⚠️ Customer Risk Segmentation")

segment_data = pd.DataFrame({
    "Segment": [
        "High Complaint Risk",
        "Digitally Disengaged",
        "Low Product Usage",
        "Stable Customers"
    ],
    "Customers": [420, 350, 290, 6940]
})

fig2 = px.pie(
    segment_data,
    names="Segment",
    values="Customers",
    hole=0.5,
    title="Customer Segmentation"
)

st.plotly_chart(fig2, use_container_width=True)

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