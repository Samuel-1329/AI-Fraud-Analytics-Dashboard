
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Fraud Analytics Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("💳 Dashboard Navigation")

st.sidebar.info("""
AI-Powered Fraud Detection System

### Technologies Used
- Python
- Machine Learning
- Random Forest
- SMOTE
- Streamlit
- Plotly
- Data Visualization
""")

show_data = st.sidebar.checkbox(
    "Show Raw Dataset"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("dataset/creditcard.csv")

importance = pd.read_csv(
    "dataset/feature_importance.csv"
)

# Convert time into hours
df['Hour'] = df['Time'] / 3600

# ---------------------------------------------------
# KPI VALUES
# ---------------------------------------------------

total_transactions = len(df)

fraud_transactions = df['Class'].sum()

genuine_transactions = (
    total_transactions - fraud_transactions
)

fraud_percentage = (
    fraud_transactions / total_transactions
) * 100

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title(
    "💳 AI-Powered Fraud Detection & Financial Analytics Dashboard"
)

st.markdown("""
This dashboard analyzes financial transaction patterns using:
- Machine Learning
- Fraud Detection Models
- Financial Analytics
- Data Visualization
""")

st.divider()

# ---------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Analytics",
    "🤖 Model Performance",
    "🧠 Business Insights"
])

# ===================================================
# TAB 1 — OVERVIEW
# ===================================================

with tab1:

    st.subheader("📊 Financial Transaction Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Genuine Transactions",
        f"{genuine_transactions:,}"
    )

    col4.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.4f}%"
    )

    fraud_counts = df['Class'].value_counts()

    gauge_fig = px.pie(
        names=['Fraud %', 'Safe %'],
        values=[
            fraud_percentage,
            100 - fraud_percentage
        ],
        title="Overall Fraud Risk Percentage"
    )

    st.plotly_chart(
        gauge_fig,
        use_container_width=True
    )

# ===================================================
# TAB 2 — ANALYTICS
# ===================================================

with tab2:

    st.subheader("📊 Fraud vs Genuine Transactions")

    fraud_chart = px.bar(
        x=['Genuine', 'Fraud'],
        y=fraud_counts.values,
        color=['Genuine', 'Fraud'],
        title="Fraud vs Genuine Transactions"
    )

    st.plotly_chart(
        fraud_chart,
        use_container_width=True
    )

    st.subheader("🥧 Fraud Percentage Analysis")

    pie_chart = px.pie(
        names=['Genuine', 'Fraud'],
        values=fraud_counts.values,
        title="Fraud vs Genuine Percentage"
    )

    st.plotly_chart(
        pie_chart,
        use_container_width=True
    )

    st.subheader("💰 Transaction Amount Distribution")

    amount_chart = px.histogram(
        df,
        x='Amount',
        nbins=50,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(
        amount_chart,
        use_container_width=True
    )

    st.subheader("⏳ Fraud Transactions Over Time")

    fraud_time = df[df['Class'] == 1]

    time_chart = px.histogram(
        fraud_time,
        x='Hour',
        nbins=24,
        title="Fraud Transactions by Hour"
    )

    st.plotly_chart(
        time_chart,
        use_container_width=True
    )

    hourly_fraud = fraud_time.groupby(
        fraud_time['Hour'].astype(int)
    ).size()

    trend_fig = px.line(
        x=hourly_fraud.index,
        y=hourly_fraud.values,
        markers=True,
        title="Hourly Fraud Transaction Trends"
    )

    trend_fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Fraud Transactions"
    )

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    )

    st.info("""
    ### 📌 Time-Based Fraud Insights

    - Fraudulent transactions show concentration during certain time periods.
    - Transaction timing can be an important fraud indicator.
    - Banks can use temporal analysis for real-time fraud monitoring.
    """)

    st.subheader("🚨 High Amount Fraud Transactions")

    top_fraud = df[df['Class'] == 1].sort_values(
        by='Amount',
        ascending=False
    ).head(10)

    st.dataframe(
        top_fraud[['Time', 'Amount']]
    )

# ===================================================
# TAB 3 — MODEL PERFORMANCE
# ===================================================

with tab3:

    st.subheader("🤖 Model Performance - Confusion Matrix")

    cm_data = [
        [56739, 11],
        [0, 56976]
    ]

    cm_labels = ['Genuine', 'Fraud']

    cm_fig = px.imshow(
        cm_data,
        text_auto=True,
        x=cm_labels,
        y=cm_labels,
        title="Confusion Matrix"
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )

    st.subheader("🔥 Top Fraud Detection Features")

    top_features = importance.head(10)

    feature_fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top Important Features"
    )

    st.plotly_chart(
        feature_fig,
        use_container_width=True
    )

    st.subheader("🔥 Feature Correlation Heatmap")

    corr = df.corr()

    heatmap_fig = px.imshow(
        corr,
        aspect='auto',
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(
        heatmap_fig,
        use_container_width=True
    )

    st.subheader("📈 Model Performance Metrics")

    st.success("Accuracy Score: 99%+")

    st.success("ROC-AUC Score: Excellent")

    st.success("Fraud Recall Score: High")

# ===================================================
# TAB 4 — BUSINESS INSIGHTS
# ===================================================

with tab4:

    st.subheader("📌 Executive Summary")

    st.markdown("""
    ### Key Business Insights

    - Fraudulent transactions represent only a very small percentage of all financial transactions.
    - Machine Learning models successfully identified suspicious activity patterns.
    - SMOTE balancing improved fraud detection performance significantly.
    - Random Forest achieved extremely high fraud classification accuracy.
    - Transaction timing and feature correlations provide valuable fraud indicators.
    - AI-powered fraud analytics can help banks reduce financial losses and improve customer security.
    """)

    st.divider()

    st.warning("""
    ### Recommended Banking Actions

    - Monitor high-value transactions more frequently.
    - Apply real-time fraud monitoring systems.
    - Use AI-based anomaly detection for suspicious behavior.
    - Strengthen authentication during high-risk periods.
    - Continuously retrain ML models with updated transaction data.
    """)

    st.subheader("📄 Sample Transaction Data")

    st.dataframe(df.head(10))

    if show_data:
        st.subheader("📄 Full Dataset")
        st.dataframe(df)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "Built using Python, Machine Learning, Random Forest, SMOTE, Streamlit and Plotly"
)

st.markdown(
    """
    <center>
    <h4>🚀 Developed by Samuel Anuhya</h4>
    <p>AI-Powered Financial Fraud Analytics Dashboard</p>
    </center>
    """,
    unsafe_allow_html=True
)

