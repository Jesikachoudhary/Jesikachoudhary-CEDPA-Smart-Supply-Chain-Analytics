import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from app.ml_models.risk_prediction import train_risk_model

# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(
    page_title="CEDPA Smart Supply Chain Analytics",
    layout="wide"
)

# --------------------------------
# Title
# --------------------------------

st.title("CEDPA Smart Supply Chain Analytics Platform")

st.markdown("Cloud Enabled Distributed Predictive Analytics Dashboard")

# --------------------------------
# Sidebar Navigation
# --------------------------------

st.sidebar.title("CEDPA Navigation")

section = st.sidebar.radio(
    "Go To",
    [
        "Dashboard Overview",
        "AI Risk Prediction",
        "Demand Analysis",
        "Inventory Insights",
        "Demand Forecasting"
    ]
)

# --------------------------------
# File Upload
# --------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain CSV",
    type=["csv"]
)

# --------------------------------
# Load Dataset
# --------------------------------

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

else:
    data = pd.read_csv(
        "app/data/sample_data/complete_dataset_large.csv"
    )

# --------------------------------
# KPI Section
# --------------------------------

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("Total Columns", len(data.columns))

with col3:
    st.metric("Dataset Loaded", "Yes")

# --------------------------------
# Dashboard Overview
# --------------------------------

if section == "Dashboard Overview":

    st.subheader("Supply Chain Dataset")

    st.dataframe(data.head())

    st.subheader("Dataset Columns")

    st.write(data.columns.tolist())

# --------------------------------
# AI Risk Prediction
# --------------------------------

elif section == "AI Risk Prediction":

    st.subheader("AI-Based Supply Chain Risk Prediction")

    # Train Model
    model, accuracy = train_risk_model(data)

    st.success(
        f"Model Trained Successfully | Accuracy: {accuracy:.2f}"
    )

    # High Risk Count
    high_risk = data[
        (data["quantity_on_hand"] < 1000) &
        (data["backlog"] > 0)
    ]

    st.metric("High Risk Products", len(high_risk))

    # Risk Distribution
    risk_counts = data["risk_level"].value_counts()

    risk_df = pd.DataFrame({
        "Risk": ["Low Risk", "High Risk"],
        "Count": risk_counts.values
    })

    fig_risk = px.pie(
        risk_df,
        names="Risk",
        values="Count",
        title="Supply Chain Risk Distribution"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # --------------------------------
    # Feature Importance
    # --------------------------------

    st.subheader("Feature Importance Analysis")

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
        title="ML Feature Importance"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    # --------------------------------
    # User Prediction
    # --------------------------------

    st.subheader("Predict Supply Chain Risk")

    col1, col2 = st.columns(2)

    with col1:
        jan = st.number_input("January Demand", value=2000)
        feb = st.number_input("February Demand", value=1800)
        mar = st.number_input("March Demand", value=2200)
        apr = st.number_input("April Demand", value=2100)
        may = st.number_input("May Demand", value=1900)
        jun = st.number_input("June Demand", value=2300)

    with col2:
        jul = st.number_input("July Demand", value=2400)
        aug = st.number_input("August Demand", value=2500)
        sep = st.number_input("September Demand", value=2100)
        octo = st.number_input("October Demand", value=2600)
        nov = st.number_input("November Demand", value=2700)
        dec = st.number_input("December Demand", value=3000)

    lead_time = st.number_input("Lead Time", value=5)

    quantity_on_hand = st.number_input(
        "Quantity On Hand",
        value=500
    )

    backlog = st.number_input("Backlog", value=1)

    # Predict Button
    if st.button("Predict Risk"):

        input_data = [[
            jan, feb, mar, apr, may, jun,
            jul, aug, sep, octo, nov, dec,
            lead_time,
            quantity_on_hand,
            backlog
        ]]

        prediction = model.predict(np.array(input_data))

        if prediction[0] == 1:
            st.error("High Supply Chain Risk Detected")

        else:
            st.success("Low Supply Chain Risk")

# --------------------------------
# Demand Analysis
# --------------------------------

elif section == "Demand Analysis":

    st.subheader("Monthly Demand Analysis")

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

    st.subheader("Inventory Distribution")

    fig2 = px.histogram(
        data,
        x="quantity_on_hand",
        nbins=20,
        title="Inventory Quantity Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Lead Time Analysis")

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

    st.subheader("Demand Forecasting")

    monthly_columns = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    # Average Monthly Demand
    monthly_avg = data[monthly_columns].mean()

    forecast_df = pd.DataFrame({
        "Month": monthly_columns,
        "Average Demand": monthly_avg.values
    })

    # Rolling Forecast
    forecast_df["Forecast"] = (
        forecast_df["Average Demand"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # Forecast Chart
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

    st.subheader("Forecast Data")

    st.dataframe(forecast_df)

    # --------------------------------
    # Download Forecast Report
    # --------------------------------

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="Download Forecast Report",
        data=csv,
        file_name="forecast_report.csv",
        mime="text/csv"
    )