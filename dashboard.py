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
# LOAD REAL DATA
# =====================================================

train = pd.read_csv(
    "data/processed/train_processed.csv"
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

.section-title {
    color: #93C5FD;
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
# SIDEBAR
# =====================================================

st.sidebar.title("🏦 BankPulse-AI")

st.sidebar.markdown("""
### Executive Analytics Dashboard

Real-time banking churn intelligence system.
""")

st.sidebar.success("✅ Dashboard Status: Active")

st.sidebar.write("---")

st.sidebar.write("### Key Modules")

st.sidebar.write("📊 KPI Analytics")
st.sidebar.write("📈 Churn Trends")
st.sidebar.write("⚠️ Risk Segmentation")
st.sidebar.write("💰 Business Cost Analysis")
st.sidebar.write("🎯 Retention Strategy")

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
        <p>High Risk Customers</p>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown(f"""
    <div class="metric-card">
        <h2>₹{estimated_loss:,}</h2>
        <p>Potential Revenue Loss</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("---")

# =====================================================
# CHURN DISTRIBUTION
# =====================================================

st.subheader("📈 Customer Churn Distribution")

churn_counts = train['churn'].value_counts()

trend_data = pd.DataFrame({

    "Status": [
        "Non-Churn",
        "Churn"
    ],

    "Customers": churn_counts.values
})

fig = px.bar(
    trend_data,
    x="Status",
    y="Customers",
    color="Status",
    title="Customer Churn Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

st.subheader("⚠️ Customer Risk Segmentation")

segment_data = pd.DataFrame({

    "Segment": [
        "High Complaint Risk",
        "Digitally Disengaged",
        "Low Product Usage",
        "Stable Customers"
    ],

    "Customers": [

        train[
            train['complaint_ratio'] > 0.5
        ].shape[0],

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

fig2 = px.pie(
    segment_data,
    names="Segment",
    values="Customers",
    hole=0.5,
    title="Customer Risk Segments"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================================
# AGE ANALYSIS
# =====================================================

st.subheader("👥 Age Distribution by Churn")

fig3 = px.box(
    train,
    x='churn',
    y='age',
    color='churn',
    title="Customer Age vs Churn"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================================
# COMPLAINT ANALYSIS
# =====================================================

st.subheader("📞 Complaints vs Churn")

fig4 = px.box(
    train,
    x='churn',
    y='total_complaints',
    color='churn',
    title="Customer Complaints Analysis"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =====================================================
# DIGITAL ENGAGEMENT ANALYSIS
# =====================================================

st.subheader("📱 Digital Engagement Analysis")

fig5 = px.box(
    train,
    x='churn',
    y='digital_engagement_combined',
    color='churn',
    title="Digital Engagement vs Churn"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

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

fig6 = px.bar(
    cost_data,
    x="Category",
    y="Cost",
    color="Category",
    title="Estimated Business Loss"
)

st.plotly_chart(
    fig6,
    use_container_width=True
)

# =====================================================
# RETENTION STRATEGY
# =====================================================

st.subheader("🎯 AI Retention Recommendations")

strategy_data = pd.DataFrame({

    "Customer Segment": [
        "High Complaint Risk",
        "Digitally Disengaged",
        "Low Product Usage"
    ],

    "AI Recommendation": [
        "Provide priority support and relationship manager.",
        "Launch mobile onboarding & engagement campaigns.",
        "Cross-sell banking products and loyalty rewards."
    ]
})

st.dataframe(
    strategy_data,
    use_container_width=True
)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.subheader("🧠 Executive Business Insights")

st.info("""
• Customers with high complaints demonstrate elevated churn probability.

• Digitally disengaged customers are more likely to leave the bank.

• Lower product utilization strongly correlates with churn behavior.

• AI-driven retention campaigns can significantly reduce business loss.
""")

# =====================================================
# FOOTER
# =====================================================

st.write("---")

st.caption(
    "Built using Streamlit, Plotly, Machine Learning & Banking Analytics"
)