import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

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
        "Demand Forecasting",
        "Geo Analytics"
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
    # Executive Insights
    # --------------------------------

    st.subheader("📊 Executive Business Insights")

    monthly_columns = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    monthly_totals = data[monthly_columns].sum()

    top_month = monthly_totals.idxmax()

    top_value = monthly_totals.max()

    avg_lead_time = data["lead-time"].mean()

    low_stock = data[data["quantity_on_hand"] < 1000]

    high_backlog = data[data["backlog"] > 0]

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"📈 Highest Demand Month: {top_month.upper()} "
            f"({top_value:.0f} units)"
        )

        st.warning(
            f"⚠️ Low Stock Products: {len(low_stock)}"
        )

    with col2:

        st.success(
            f"🚚 Average Lead Time: {avg_lead_time:.2f} days"
        )

        st.error(
            f"📦 Products With Backlog: {len(high_backlog)}"
        )

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

    # --------------------------------
    # Feature Importance
    # --------------------------------

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
    # User Prediction
    # --------------------------------

    st.subheader("🔮 Predict Supply Chain Risk")

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
            st.error("⚠️ High Supply Chain Risk Detected")

        else:
            st.success("✅ Low Supply Chain Risk")

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
    # Correlation Heatmap
    # --------------------------------

    st.subheader("🔥 Correlation Heatmap")

    numeric_data = data.select_dtypes(
        include=["int64", "float64"]
    )

    corr_matrix = numeric_data.corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Feature Correlation Heatmap"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

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

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    # --------------------------------
    # Manual Forecast Prediction
    # --------------------------------

    st.subheader("🔮 Manual Demand Forecast Prediction")

    user_demand = st.number_input(
        "Enter Current Average Demand",
        value=2000
    )

    forecast_value = (
        forecast_df["Forecast"].mean()
        + (user_demand * 0.05)
    )

    st.metric(
        "Predicted Future Demand",
        f"{forecast_value:.2f}"
    )

    # --------------------------------
    # ML Forecast Prediction
    # --------------------------------

    st.subheader("🤖 AI Demand Forecasting")

    month_numbers = np.array(
        range(1, 13)
    ).reshape(-1, 1)

    demand_values = monthly_avg.values

    forecast_model = LinearRegression()

    forecast_model.fit(
        month_numbers,
        demand_values
    )

    future_month = st.slider(
        "Select Future Month",
        13,
        24,
        15
    )

    future_prediction = forecast_model.predict(
        [[future_month]]
    )[0]

    st.success(
        f"Predicted Demand for Month "
        f"{future_month}: "
        f"{future_prediction:.2f}"
    )

    future_df = pd.DataFrame({
        "Month": list(range(1, 13)) + [future_month],
        "Demand": list(demand_values) + [future_prediction]
    })

    fig_future = px.line(
        future_df,
        x="Month",
        y="Demand",
        markers=True,
        title="AI Forecasted Demand Trend"
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )

    # --------------------------------
    # Download Forecast Report
    # --------------------------------

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast Report",
        data=csv,
        file_name="forecast_report.csv",
        mime="text/csv"
    )

# --------------------------------
# Geo Analytics
# --------------------------------

elif section == "Geo Analytics":

    st.subheader("🌍 India Supply Chain Geo Analytics")

    geo_data = pd.DataFrame({
        "City": [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Chennai",
            "Hyderabad",
            "Kolkata",
            "Pune"
        ],

        "Latitude": [
            28.6139,
            19.0760,
            12.9716,
            13.0827,
            17.3850,
            22.5726,
            18.5204
        ],

        "Longitude": [
            77.2090,
            72.8777,
            77.5946,
            80.2707,
            78.4867,
            88.3639,
            73.8567
        ],

        "Demand": [
            5000,
            7000,
            6500,
            4000,
            5500,
            3500,
            4800
        ]
    })

    fig_map = px.scatter_mapbox(
        geo_data,
        lat="Latitude",
        lon="Longitude",
        size="Demand",
        color="Demand",
        hover_name="City",
        hover_data=["Demand"],
        zoom=4.5,
        height=750,
        title="Supply Chain Demand Across India"
    )

    fig_map.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    st.dataframe(
        geo_data,
        use_container_width=True
    )

    # --------------------------------
# Footer
# --------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:gray;'>
    Developed by Jesika Choudhary • AI Powered Supply Chain Analytics
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)