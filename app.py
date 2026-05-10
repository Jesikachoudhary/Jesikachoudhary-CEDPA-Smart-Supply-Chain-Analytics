import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from app.ml_models.risk_prediction import train_risk_model

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="CEDPA Smart Supply Chain Analytics",
    layout="wide",
    page_icon="📊"
)

# --------------------------------
# Custom CSS Styling
# --------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #050816;
    }

    h1, h2, h3 {
        color: white;
    }

    p {
        color: #cbd5e1;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    .metric-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 18px;
    }

    .metric-value {
        color: white;
        font-size: 38px;
        font-weight: bold;
    }

    .hero-title {
        font-size: 52px;
        font-weight: 800;
        color: white;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Hero Section
# --------------------------------

st.markdown(
    """
    <div class="hero-title">
    CEDPA Smart Supply Chain Analytics Platform
    </div>

    <div class="hero-subtitle">
    AI Powered Predictive Analytics • Demand Forecasting • Inventory Intelligence
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.title("📌 CEDPA Navigation")

section = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Dashboard Overview",
        "AI Risk Prediction",
        "Demand Analysis",
        "Inventory Insights",
        "Demand Forecasting"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain CSV",
    type=["csv"]
)

# --------------------------------
# Load Data
# --------------------------------

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

else:
    data = pd.read_csv(
        "app/data/sample_data/complete_dataset_large.csv"
    )

# --------------------------------
# KPI Cards
# --------------------------------

st.subheader("📈 Business KPI Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{len(data)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Columns</div>
            <div class="metric-value">{len(data.columns)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-title">System Status</div>
            <div class="metric-value">Active</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------
# Dashboard Overview
# --------------------------------

if section == "Dashboard Overview":

    st.subheader("📦 Supply Chain Dataset")

    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("📋 Dataset Columns")

    st.write(data.columns.tolist())

# --------------------------------
# AI Risk Prediction
# --------------------------------

elif section == "AI Risk Prediction":

    st.subheader("🤖 AI Supply Chain Risk Prediction")

    model, accuracy = train_risk_model(data)

    st.success(
        f"Model Accuracy: {accuracy:.2f}"
    )

    risk_counts = data["risk_level"].value_counts()

    risk_df = pd.DataFrame({
        "Risk": ["Low Risk", "High Risk"],
        "Count": risk_counts.values
    })

    fig_risk = px.pie(
        risk_df,
        names="Risk",
        values="Count",
        title="Supply Chain Risk Distribution",
        hole=0.5
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # Feature Importance

    st.subheader("📊 Feature Importance Analysis")

    importance_df = pd.DataFrame({
        "Feature": [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
            "lead-time",
            "quantity_on_hand",
            "backlog"
        ],
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

# --------------------------------
# Demand Analysis
# --------------------------------

elif section == "Demand Analysis":

    st.subheader("📉 Monthly Demand Analysis")

    monthly_columns = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    monthly_demand = data[monthly_columns].sum()

    demand_df = pd.DataFrame({
        "Month": monthly_columns,
        "Demand": monthly_demand.values
    })

    fig = px.line(
        demand_df,
        x="Month",
        y="Demand",
        markers=True,
        title="Monthly Demand Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Inventory Insights
# --------------------------------

elif section == "Inventory Insights":

    st.subheader("📦 Inventory Distribution")

    fig2 = px.histogram(
        data,
        x="quantity_on_hand",
        nbins=20,
        title="Inventory Quantity Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚚 Lead Time Analysis")

    fig3 = px.box(
        data,
        y="lead-time",
        title="Lead Time Distribution"
    )

    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------
# Demand Forecasting
# --------------------------------

elif section == "Demand Forecasting":

    st.subheader("📈 Demand Forecasting")

    monthly_columns = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    monthly_avg = data[monthly_columns].mean()

    forecast_df = pd.DataFrame({
        "Month": monthly_columns,
        "Average Demand": monthly_avg.values
    })

    forecast_df["Forecast"] = (
        forecast_df["Average Demand"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    fig_forecast = px.line(
        forecast_df,
        x="Month",
        y=["Average Demand", "Forecast"],
        markers=True,
        title="Demand Forecasting Analysis"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.subheader("📋 Forecast Data")

    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast Report",
        data=csv,
        file_name="forecast_report.csv",
        mime="text/csv"
    )
